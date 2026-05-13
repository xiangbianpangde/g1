"""
tools.py — G1 机器人控制 LangChain Tool 集。

基于 core.Bridge（UDP → C++ g1_node）实现：
  - 运动控制：前进/后退/左移/右移/旋转/停止
  - 姿态控制：站起/蹲下/高低站姿/阻尼模式
  - 手臂动作：挥手/握手/双手举起/手臂收回
  - 语音播报：通过 G1 喇叭说话
  - 安全：紧急停止
  - 感知：物体检测 / 机器人状态查询（模拟模式，需实机扩展）

用法：
    from tools import create_robot_tools
    bridge = Bridge(host="127.0.0.1", port=9870)
    tools = create_robot_tools(bridge, simulation=True)  # Windows 开发用模拟
    tools = create_robot_tools(bridge, simulation=False) # 实机部署

安全设计（四层防护）：
    L1: System Prompt 规则约束（速度/距离上限）
    L2: Pydantic Field 参数校验（范围、类型）
    L3: Tool 内部运行时检查（方向合法性、模拟/实机分支）
    L4: SDK/硬件急停（倾倒/过热/断连自动停止）
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

from langchain_core.tools import tool, ToolException
from pydantic import BaseModel, Field

from core.bridge import Bridge

if TYPE_CHECKING:
    from core.vision import Vision


# ═══════════════════════════════════════════════════════════════════════
# CoreComponents 容器
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class CoreComponents:
    """核心组件容器，将 Vision 等实例传递给 Tool 集。"""
    vision: Vision | None = None


# ═══════════════════════════════════════════════════════════════════════
# Pydantic 输入模型
# ═══════════════════════════════════════════════════════════════════════

class SpeedLevel(str, Enum):
    SLOW = "slow"
    NORMAL = "normal"
    FAST = "fast"


class MoveInput(BaseModel):
    direction: str = Field(
        description="移动方向: forward(前进) / backward(后退) / left(左移) / right(右移)",
    )
    distance: float = Field(
        description="移动距离(米)，范围 0.1 ~ 5.0",
        gt=0.0,
        le=5.0,
    )
    speed_level: SpeedLevel = Field(
        default=SpeedLevel.NORMAL,
        description="速度等级: slow=0.2m/s, normal=0.3m/s, fast=0.5m/s",
    )


class TurnInput(BaseModel):
    angle: float = Field(
        description="旋转角度(度)，正=右转，范围 15 ~ 360",
        ge=15.0,
        le=360.0,
    )
    speed_level: SpeedLevel = Field(
        default=SpeedLevel.NORMAL,
        description="旋转速度等级: slow=0.2rad/s, normal=0.3rad/s, fast=0.5rad/s",
    )


class SayInput(BaseModel):
    text: str = Field(
        description="要让机器人说出来的文字（中文），不超过 200 字",
        min_length=1,
        max_length=200,
    )


# ═══════════════════════════════════════════════════════════════════════
# 常量
# ═══════════════════════════════════════════════════════════════════════

SPEED_MAP = {"slow": 0.2, "normal": 0.3, "fast": 0.5}
DIRECTION_VEC = {
    "forward":  (1.0,  0.0,  0.0),
    "backward": (-1.0, 0.0,  0.0),
    "left":     (0.0,  1.0,  0.0),
    "right":    (0.0,  -1.0, 0.0),
}
POSTURE_IDS = {"stand_up": 1, "squat_down": 2, "low_stand": 3, "high_stand": 4, "damp": 5}
ARM_IDS = {"wave": 26, "shake_hand": 27, "hands_up": 15, "arms_down": 99}


def _sim_log(action: str, detail: str = "") -> str:
    msg = f"[模拟] {action}"
    if detail:
        msg += f" — {detail}"
    print(msg)
    return msg


# ═══════════════════════════════════════════════════════════════════════
# Tool 工厂
# ═══════════════════════════════════════════════════════════════════════

def create_robot_tools(bridge: Bridge, *, simulation: bool = False,
                       core: CoreComponents | None = None):
    """创建所有 G1 机器人控制 Tool。simulation=True 时仅打印不发送 UDP。

    Args:
        bridge: UDP 通信客户端
        simulation: 是否模拟模式
        core: 可选核心组件（如 Vision），供感知类 Tool 使用
    """

    # ── 运动控制 ──────────────────────────────────────────────────────

    @tool(args_schema=MoveInput)
    def move_robot(direction: str, distance: float,
                   speed_level: SpeedLevel = SpeedLevel.NORMAL) -> str:
        """控制机器人向指定方向移动指定距离。完成后自动停止。"""
        speed = SPEED_MAP[speed_level.value]
        if direction not in DIRECTION_VEC:
            raise ToolException(
                f"不支持的方向 '{direction}'，请使用: forward/backward/left/right"
            )
        sign, lat, _ = DIRECTION_VEC[direction]
        vx, vy = sign * speed, lat * speed
        duration = distance / speed

        if simulation:
            return _sim_log("移动", f"{direction} {distance}m @ {speed}m/s ({duration:.1f}s)")

        try:
            steps = int(duration / 0.1)
            for _ in range(steps):
                bridge.send_move(vx, vy, 0.0)
                time.sleep(0.1)
            bridge.send_move(0.0, 0.0, 0.0)
            return f"已完成：向{direction}移动 {distance:.1f} 米（速度 {speed} m/s）"
        except Exception as e:
            bridge.send_move(0.0, 0.0, 0.0)
            raise ToolException(f"移动失败: {e}")

    # ── 旋转控制 ──────────────────────────────────────────────────────

    @tool(args_schema=TurnInput)
    def turn_robot(angle: float,
                   speed_level: SpeedLevel = SpeedLevel.NORMAL) -> str:
        """控制机器人原地旋转。正角度=右转。"""
        import math
        speed = SPEED_MAP[speed_level.value]
        angle_rad = math.radians(abs(angle))
        duration = angle_rad / speed
        direction = "右转" if angle > 0 else "左转"

        if simulation:
            return _sim_log("旋转", f"{direction} {abs(angle):.0f}° @ {speed}rad/s ({duration:.1f}s)")

        try:
            vyaw = speed if angle > 0 else -speed
            steps = int(duration / 0.1)
            for _ in range(steps):
                bridge.send_move(0.0, 0.0, vyaw)
                time.sleep(0.1)
            bridge.send_move(0.0, 0.0, 0.0)
            return f"已完成：{direction} {abs(angle):.0f} 度"
        except Exception as e:
            bridge.send_move(0.0, 0.0, 0.0)
            raise ToolException(f"旋转失败: {e}")

    # ── 停止 ──────────────────────────────────────────────────────────

    @tool
    def stop_moving() -> str:
        """停止机器人当前所有移动，保持站立姿态。"""
        if simulation:
            return _sim_log("停止移动")
        bridge.send_move(0.0, 0.0, 0.0)
        return "已停止所有移动"

    # ── 姿态控制 ──────────────────────────────────────────────────────

    @tool
    def stand_up() -> str:
        """让机器人从蹲下状态站起。行走前必须先站立。"""
        if simulation:
            return _sim_log("站起", "Squat2StandUp")
        bridge.send_posture(POSTURE_IDS["stand_up"])
        time.sleep(2.0)
        return "机器人已站起"

    @tool
    def squat_down() -> str:
        """让机器人从站立状态蹲下。蹲下后无法行走。"""
        if simulation:
            return _sim_log("蹲下", "StandUp2Squat")
        bridge.send_posture(POSTURE_IDS["squat_down"])
        time.sleep(2.0)
        return "机器人已蹲下"

    @tool
    def low_stand() -> str:
        """切换到低站姿（重心更低，更稳定）。"""
        if simulation:
            return _sim_log("低站姿")
        bridge.send_posture(POSTURE_IDS["low_stand"])
        return "已切换到低站姿"

    @tool
    def high_stand() -> str:
        """切换到高站姿（重心更高，视野更好）。"""
        if simulation:
            return _sim_log("高站姿")
        bridge.send_posture(POSTURE_IDS["high_stand"])
        return "已切换到高站姿"

    @tool
    def damp_mode() -> str:
        """进入阻尼模式（关节放松，可被外力推动）。"""
        if simulation:
            return _sim_log("阻尼模式")
        bridge.send_posture(POSTURE_IDS["damp"])
        return "已进入阻尼模式"

    # ── 手臂动作 ──────────────────────────────────────────────────────

    @tool
    def wave_hand() -> str:
        """让机器人挥手。用于打招呼或告别。"""
        if simulation:
            return _sim_log("手臂动作", "挥手 (id=26)")
        bridge.send_arm(ARM_IDS["wave"])
        return "正在挥手"

    @tool
    def shake_hand() -> str:
        """让机器人做出握手动作。用户伸出手时使用。"""
        if simulation:
            return _sim_log("手臂动作", "握手 (id=27)")
        bridge.send_arm(ARM_IDS["shake_hand"])
        return "正在握手"

    @tool
    def hands_up() -> str:
        """让机器人双手举起（投降/欢呼/吸引注意）。"""
        if simulation:
            return _sim_log("手臂动作", "双手举起 (id=15)")
        bridge.send_arm(ARM_IDS["hands_up"])
        return "双手已举起"

    @tool
    def arms_down() -> str:
        """让机器人手臂收回默认位置。"""
        if simulation:
            return _sim_log("手臂动作", "手臂收回 (id=99)")
        bridge.send_arm(ARM_IDS["arms_down"])
        return "手臂已收回"

    # ── 语音播报 ──────────────────────────────────────────────────────

    @tool(args_schema=SayInput)
    def speak(text: str) -> str:
        """让机器人通过喇叭说出指定文字（中文，不超过200字）。"""
        if simulation:
            return _sim_log("语音播报", f"→ G1 喇叭: {text!r}")
        bridge.send_tts(text)
        est = 1.3 + 0.22 * len(text)
        time.sleep(min(est, 5.0))
        return f"已播报: {text}"

    # ── 紧急停止 ──────────────────────────────────────────────────────

    @tool
    def emergency_stop() -> str:
        """【最高优先级】立即停止所有运动并进入阻尼模式。检测到危险时立即调用。"""
        if simulation:
            return _sim_log("🚨 紧急停止", "所有运动停止 + 阻尼模式")
        bridge.send_move(0.0, 0.0, 0.0)
        bridge.send_posture(POSTURE_IDS["damp"])
        return "🚨 紧急停止已执行！机器人已停止所有运动并进入阻尼模式"

    # ── 感知 ──────────────────────────────────────────────────────────

    @tool
    def detect_objects() -> str:
        """检测前方物体，返回障碍距离信息。实机使用 RealSense D435i 深度相机。"""
        if not simulation:
            if core is None or core.vision is None:
                return ("视觉模块未就绪，请确认 RealSense D435i 已连接"
                        "（可通过 --no-vision 参数跳过视觉初始化）")
            vision = core.vision
            if vision._pipeline is None:
                return ("视觉模块已加载但相机未启动 (RealSense D435i)，"
                        "请先调用 vision.start() 打开相机")
            try:
                frames = vision._pipeline.wait_for_frames(timeout_ms=5000)
                aligned = vision._align.process(frames)
                depth = aligned.get_depth_frame()
                if not depth:
                    return "无法获取深度帧 (RealSense D435i)"
                w, h = depth.get_width(), depth.get_height()
                dist_l, dist_c, dist_r = vision._check_obstacles(depth, w, h)
                result = (f"前方检测: 左{dist_l:.1f}m / 中{dist_c:.1f}m / "
                          f"右{dist_r:.1f}m (RealSense D435i)")
                warnings = []
                if dist_l < 0.6:
                    warnings.append(f"左侧仅{dist_l:.1f}m")
                if dist_c < 0.6:
                    warnings.append(f"中央仅{dist_c:.1f}m")
                if dist_r < 0.6:
                    warnings.append(f"右侧仅{dist_r:.1f}m")
                if warnings:
                    result += "\n⚠️ 警告：障碍物过近！" + "、".join(warnings)
                return result
            except Exception as e:
                return f"视觉检测失败: {e}"
        return _sim_log("物体检测",
                        "前方 2.3m 椅子(85%) | 左侧 3.1m 桌子(72%)")

    @tool
    def get_robot_status() -> str:
        """查询机器人状态（电量/温度/连接/IMU）。实机需订阅 DDS LowState。"""
        if not simulation:
            host, port = bridge.addr
            return (
                f"Bridge 连接: {host}:{port}\n"
                f"状态数据为占位值(需 DDS LowState 订阅)\n"
                f"- 电池: ~85%\n"
                f"- 关节温度: ~38°C\n"
                f"- IMU: 正常\n"
                f"- 姿态: 站立"
            )
        return _sim_log("状态查询",
                        "电池 85% | 关节 38°C | 连接正常 | IMU 正常 | 站立")

    return [
        move_robot,
        turn_robot,
        stop_moving,
        stand_up,
        squat_down,
        low_stand,
        high_stand,
        damp_mode,
        wave_hand,
        shake_hand,
        hands_up,
        arms_down,
        speak,
        emergency_stop,
        detect_objects,
        get_robot_status,
    ]
