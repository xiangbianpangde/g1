"""
agent.py — 基于 LangGraph/LangChain 的 G1 人形机器人 Agent。

核心设计：
  - LLM：Ollama 本地模型（默认 qwen3:8b），复用项目中 core/chat.py 的惯用模型
  - Tool：全部调用 core.Bridge 的成熟 UDP 协议（MOVE / ARM / TTS / POSTURE）
  - Agent：LangGraph create_react_agent (ReAct)，通过 Tool 路由控制机器人
  - Vision：可选 RealSense + YOLO-pose 视觉感知（--vision 启用）

用法：
    # Windows 开发（模拟模式，不连实机）
    python agent.py --sim

    # 实机部署（需先在机器人上跑 g1_node）
    python agent.py --bridge-host 192.168.123.161

    # 实机 + 视觉感知
    python agent.py --bridge-host 192.168.123.161 --vision

    # 交互模式（默认）
    python agent.py --sim

    # 单次指令模式
    python agent.py --sim --command "向前走两米然后挥手"

    # 换模型
    python agent.py --sim --model qwen3:8b
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

from langgraph.prebuilt import create_react_agent
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.checkpoint.sqlite import SqliteSaver

from core.bridge import Bridge
from core.chat import Chat
from tools import create_robot_tools, CoreComponents

load_dotenv()

# ═══════════════════════════════════════════════════════════════════════
# System Prompt（多层安全防护 L1）
# ═══════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """你是一个控制宇树 G1 人形机器人的 AI 助手。你可以通过调用工具来控制机器人的运动和感知。

## 安全规则（最高优先级，必须严格遵守）

1. 移动速度不得超过 0.5 m/s（使用 fast 等级），默认使用 normal（0.3 m/s）
2. 每次连续移动距离不得超过 5 米
3. 旋转速度不得超过 0.5 rad/s，单次旋转不超过 360 度
4. 执行动作前必须确认周围环境安全，如有疑问先调用 detect_objects
5. 如果用户指令模糊（如"走过去"），应先询问具体方向和距离
6. 检测到障碍物时应立即停止并告知用户
7. 任何情况下，当用户说"停"、"停止"、"急停"时必须立即调用 emergency_stop
8. 机器人站起后才能行走；蹲下后不能行走

## 可用能力

- 行走控制：move_robot（前进/后退/左移/右移）、turn_robot（旋转）、stop_moving（停止）
- 姿态控制：stand_up（站起）、squat_down（蹲下）、low_stand（低站姿）、high_stand（高站姿）、damp_mode（阻尼模式）
- 手臂动作：wave_hand（挥手）、shake_hand（握手）、hands_up（双手举起）、arms_down（手臂收回）
- 语音播报：speak（让机器人说话）
- 环境感知：detect_objects（物体检测）、get_robot_status（状态查询）
- 安全：emergency_stop（紧急停止）

## 交互风格

- 用简洁中文回复用户，一句话即可
- 执行动作后报告简明结果
- 遇到异常立即告知用户并建议处理方案
- 如果用户说了多个动作，按顺序逐一执行"""


# ═══════════════════════════════════════════════════════════════════════
# Agent 初始化
# ═══════════════════════════════════════════════════════════════════════

def create_g1_agent(
    bridge: Bridge,
    *,
    simulation: bool = False,
    model_name: str = "qwen3:8b",
    db_path: str | None = None,
    chat: Chat | None = None,
    core: CoreComponents | None = None,
):
    """
    创建完整的 G1 Agent。

    Args:
        bridge: UDP 通信客户端
        simulation: 是否模拟模式（True=仅打印，不发送 UDP）
        model_name: Ollama 模型名（默认 qwen3:8b，与 core/chat.py 一致）
        db_path: SQLite 持久化路径（默认 ./data/agent_checkpoint.db）
        chat: 可选的 core.Chat 实例（用于复用已有的对话历史/模型配置）
        core: 可选的 CoreComponents（如 Vision），供感知类 Tool 使用
    Returns:
        (agent, tools) 元组
    """
    # ── 模型（Ollama 本地）───────────────────────────────────────────
    # 若提供了 Chat 实例，使用其 model 名；否则使用函数参数默认值
    effective_model = chat.model if chat else model_name
    print(f"[AGENT] 初始化 Ollama 模型: {effective_model}")
    model = init_chat_model(
        model=effective_model,
        model_provider="ollama",   # 本地 Ollama，不走 API
        temperature=0.1,           # 低温度保证控制指令的确定性
    )

    # ── 工具 ──────────────────────────────────────────────────────────
    tools = create_robot_tools(bridge, simulation=simulation, core=core)
    print(f"[AGENT] 已加载 {len(tools)} 个工具:")
    for t in tools:
        desc = t.description.split("\n")[0][:60]
        print(f"        - {t.name}: {desc}")

    # ── Checkpointer（对话记忆持久化到 SQLite）────────────────────────
    db_path = db_path or str(Path(__file__).parent / "data" / "agent_checkpoint.db")
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    checkpointer = SqliteSaver.from_conn_string(db_path)

    # ── 创建 Agent（LangGraph create_react_agent）─────────────────────
    agent = create_react_agent(
        model=model,
        tools=tools,
        prompt=SYSTEM_PROMPT,
        checkpointer=checkpointer,
    )

    return agent, tools


# ═══════════════════════════════════════════════════════════════════════
# 交互模式
# ═══════════════════════════════════════════════════════════════════════

def run_interactive(agent, thread_id: str = "g1-agent-session"):
    """命令行交互循环。"""
    config = {"configurable": {"thread_id": thread_id}}
    print("\n" + "=" * 60)
    print("  G1 Agent 已就绪 — 输入指令控制机器人（q 退出）")
    print("  示例: '向前走一米' / '挥手打招呼' / '检测前方有什么'")
    print("=" * 60 + "\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break

        if user_input.lower() == "q":
            print("再见！")
            break
        if not user_input:
            continue

        print("G1: ", end="", flush=True)
        try:
            t0 = time.time()
            result = agent.invoke(
                {"messages": [HumanMessage(content=user_input)]},
                config=config,
            )
            messages = result.get("messages", [])
            reply = ""
            for msg in reversed(messages):
                if isinstance(msg, AIMessage) and msg.content:
                    reply = msg.content
                    break
            if reply:
                print(reply)
            else:
                print("(已执行)")
            print(f"     ({time.time() - t0:.1f}s)\n")
        except Exception as e:
            print(f"\n[ERROR] {e}\n")


def run_single(agent, command: str, thread_id: str = "g1-agent-once"):
    """单次指令模式"""
    config = {"configurable": {"thread_id": thread_id}}
    print(f"[AGENT] 执行: {command}")
    try:
        result = agent.invoke(
            {"messages": [HumanMessage(content=command)]},
            config=config,
        )
        messages = result.get("messages", [])
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and msg.content:
                print(msg.content)
                return
        print("(已执行)")
    except Exception as e:
        print(f"[ERROR] {e}")
        sys.exit(1)


# ═══════════════════════════════════════════════════════════════════════
# 主入口
# ═══════════════════════════════════════════════════════════════════════

def main() -> None:
    p = argparse.ArgumentParser(
        description="G1 人形机器人 LangGraph Agent（Ollama 本地推理）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python agent.py --sim                              # 模拟模式交互
  python agent.py --sim --command "前进一米然后挥手"   # 模拟模式单指令
  python agent.py --bridge-host 192.168.123.161       # 实机模式
  python agent.py --bridge-host 192.168.123.161 --vision  # 实机 + 视觉
  python agent.py --sim --model qwen3:8b              # 换 Ollama 模型
        """,
    )
    p.add_argument("--bridge-host", default="127.0.0.1", help="C++ g1_node 地址")
    p.add_argument("--bridge-port", type=int, default=9870, help="C++ g1_node 端口")
    p.add_argument("--sim", action="store_true", help="模拟模式（不连实机，仅打印动作）")
    p.add_argument("--model", default="qwen3:8b", help="Ollama 模型名（默认 qwen3:8b）")
    p.add_argument("--command", default=None, help="单次指令（不进入交互模式）")
    p.add_argument("--thread-id", default="g1-agent-session", help="会话 ID")
    p.add_argument("--db", default=None, help="SQLite checkpoint 路径")
    p.add_argument("--verbose", action="store_true", help="Bridge 调试输出")
    p.add_argument(
        "--vision", action="store_true", default=False,
        help="启用 RealSense + YOLO-pose 视觉感知（需要 RealSense D435i 相机）",
    )
    args = p.parse_args()

    # ── 连接 Bridge ───────────────────────────────────────────────────
    bridge = Bridge(
        host=args.bridge_host,
        port=args.bridge_port,
        verbose=args.verbose,
    )

    if args.sim:
        print("[AGENT] 模拟模式 — 不连实机，仅打印动作")
    else:
        print(f"[AGENT] 实机模式 — bridge → {args.bridge_host}:{args.bridge_port}")

    # ── Vision（可选感知模块）────────────────────────────────────────
    vision = None
    core_components = None

    if args.vision and not args.sim:
        print("[AGENT] 视觉感知已启用 — 初始化 RealSense + YOLO-pose ...")
        try:
            from core.vision import Vision
            vision = Vision(bridge)
            vision.start()
            core_components = CoreComponents(vision=vision)
            print("[AGENT] 视觉模块就绪")
        except Exception as e:
            print(f"[AGENT] 视觉初始化失败: {e}")
            print("[AGENT] 将继续运行，但视觉感知不可用")
            vision = None  # 确保不会在 finally 中重复 close
    elif args.vision and args.sim:
        print("[AGENT] --vision 在模拟模式下被忽略（模拟模式不需要相机）")

    # ── 创建 Agent ────────────────────────────────────────────────────
    agent, _tools = create_g1_agent(
        bridge,
        simulation=args.sim,
        model_name=args.model,
        db_path=args.db,
        core=core_components,
    )
    print(f"[AGENT] Agent 创建完成，会话 ID: {args.thread_id}")

    # ── 执行 ──────────────────────────────────────────────────────────
    try:
        if args.command:
            run_single(agent, args.command, thread_id=args.thread_id)
        else:
            run_interactive(agent, thread_id=args.thread_id)
    finally:
        if vision is not None:
            try:
                vision.close()
                print("[AGENT] 视觉模块已关闭")
            except Exception as e:
                print(f"[AGENT] 视觉关闭异常: {e}")
        bridge.close()


if __name__ == "__main__":
    main()
