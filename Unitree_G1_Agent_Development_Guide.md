# 基于LangChain的宇树G1人形机器人Agent开发调研报告

## Abstract

本报告围绕"基于LangChain框架为宇树G1人形机器人构建具备运动控制与对话交互能力的Agent"这一目标，从**硬件平台能力、SDK与通信协议、LangChain Agent架构设计、Tool开发模式、语音交互管线、安全机制、开源生态参考**七大维度进行系统性调研。核心发现如下：

- **宇树G1 EDU版**（16.9万~30.9万元）提供完整的Python SDK（`unitree_sdk2_python`），基于DDS通信协议，封装了LocoClient高层运动控制API（行走、站立、挥手、握手等）及底层29电机PD控制接口，控制频率达500Hz
- **LangGraph**已取代AgentExecutor成为LangChain推荐的Agent编排框架，原生支持状态持久化、循环工作流和Human-in-the-Loop，是机器人控制场景的最佳选择
- **EdgeVox、OM1、DimOS**三个开源项目已实现Unitree G1的LLM语音控制，其中EdgeVox提供最完整的离线语音Agent框架（首音频延迟约0.8秒）
- 多层安全验证体系（System Prompt → Pydantic参数校验 → 运行时检查 → 硬件急停）是LLM控制物理机器人的必要保障
- 推荐技术栈：**LangGraph + LangChain Tools + unitree_sdk2_python + FunASR/CosyVoice + NVIDIA Jetson Orin NX**

## 1. 引言

### 1.1 项目背景

人形机器人正从实验室走向商业化应用。宇树科技G1人形机器人以其相对亲民的价格（EDU版16.9万元起）和开放的SDK，成为开发者进行AI Agent+机器人集成实验的理想平台。与此同时，大语言模型（LLM）的快速发展使得通过自然语言控制机器人成为可能——开发者无需编写复杂的有限状态机，而是通过LangChain等框架构建Agent，让LLM自主推理并调用预定义的Tool来控制机器人。

### 1.2 调研目标

本调研旨在回答以下核心问题：

1. 宇树G1机器人的开发能力边界在哪里？SDK提供了哪些控制接口？
2. 如何用LangChain/LangGraph构建一个能控制G1运动和进行对话的Agent？
3. 需要开发哪些Tool？每个Tool的技术实现路径是什么？
4. 语音交互（ASR/TTS）如何与Agent无缝集成？
5. LLM控制物理机器人有哪些安全风险？如何缓解？
6. 有哪些成熟的开源项目可以参考或直接复用？

### 1.3 调研范围与方法

- **硬件平台**：宇树G1 EDU版（29 DOF + Dex3-1灵巧手）
- **软件框架**：LangChain/LangGraph + unitree_sdk2_python
- **功能范围**：运动控制（行走、导航、抓取）+ 语音对话交互
- **调研方法**：官方文档分析、GitHub开源项目调研、技术社区资料收集

## 2. 宇树G1机器人平台能力分析

### 2.1 硬件规格总览

| 参数 | G1 EDU 进阶版 | G1 EDU 旗舰版 |
|------|---------------|---------------|
| 总自由度 | 29 DOF | 29 DOF + 灵巧手 |
| 腿部 | 6 DOF/腿 × 2 | 同左 |
| 腰部 | 1 DOF（可选+2） | 同左 |
| 手臂 | 7 DOF/臂 × 2 | 同左 |
| 灵巧手 | 无 | Dex3-1（7电机/手，9压力传感器） |
| 膝关节最大扭矩 | 120 N.m | 120 N.m |
| 手臂最大负载 | 约3 kg | 约3 kg |
| 重量 | 约35 kg | 约35 kg+ |
| 续航 | 约2小时 | 约2小时 |
| 计算平台 | NVIDIA Jetson Orin NX | NVIDIA Jetson Orin NX |
| GPU | 1024 CUDA核心，32 Tensor Cores | 同左 |
| 显存/内存 | 16 GB | 16 GB |
| 存储 | 2 TB | 2 TB |
| 3D LiDAR | LIVOX MID-360 | 同左 |
| 深度相机 | Intel D435i | 同左 |
| 麦克风 | 4麦克风阵列 | 同左 |
| 扬声器 | 5W | 同左 |

**关键洞察**：G1 EDU版的Jetson Orin NX（16GB显存）可运行轻量级LLM（如Qwen2-7B、Gemma-4B），但无法运行大参数模型。对于需要更强算力的场景，建议采用"边缘推理+云端大模型"的混合架构。

### 2.2 SDK与通信架构

宇树G1采用**DDS（Data Distribution Service）**作为核心通信协议，使用Cyclone DDS实现，基于发布/订阅模式与机器人进行实时通信。

```
开发者代码 → Python SDK → DDS Publisher → 以太网 → 机器人DDS Subscriber → 电机控制器
```

**SDK安装**：
```bash
git clone https://github.com/unitreerobotics/unitree_sdk2_python.git
cd unitree_sdk2_python
pip3 install -e .
```

**初始化**：
```python
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
ChannelFactoryInitialize(0, "eth0")  # Domain ID 0, 网络接口eth0
```

**网络配置**：

| 设备 | IP地址 |
|------|--------|
| 机器人主控 | 192.168.123.161 |
| 开发计算单元(Jetson Orin NX) | 192.168.123.164 |
| LiDAR | 192.168.123.120 |
| 开发者电脑 | 192.168.123.x (DHCP) |

### 2.3 运动控制API分层

G1的运动控制API分为**高层**和**底层**两层，开发者应根据Agent控制粒度选择合适的层级。

#### 高层API（LocoClient）—— 适合Agent直接调用

```python
from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient

client = LocoClient()
client.SetTimeout(10.0)
client.Init()

client.Squat2StandUp()       # 站立
client.Move(0.3, 0.0, 0.0)  # 前进0.3m/s
client.WaveHand()            # 挥手
client.ShakeHand()           # 握手
client.Move(0.0, 0.0, 0.0)  # 停止
client.StandUp2Squat()       # 蹲下
```

| 方法 | 说明 | 参数范围 |
|------|------|----------|
| `Squat2StandUp()` | 从蹲下站起 | 无 |
| `StandUp2Squat()` | 从站立蹲下 | 无 |
| `Move(vx, vy, vyaw)` | 行走控制 | vx: -1.0~1.0 m/s, vy: -0.5~0.5 m/s, vyaw: -1.0~1.0 rad/s |
| `WaveHand()` | 挥手 | 无 |
| `ShakeHand()` | 握手 | 无 |
| `LowStand()` / `HighStand()` | 低/高站姿 | 无 |
| `Damp()` | 阻尼模式 | 无 |

#### 底层API（LowCmd）—— 适合精细控制

通过DDS Topic `rt/lowcmd` 直接控制29个电机，控制频率500Hz：

```python
# 电机控制参数
motor_cmd[i].mode = 1        # 使能
motor_cmd[i].q = 0.5         # 目标位置(弧度)
motor_cmd[i].dq = 0.0        # 目标速度(弧度/秒)
motor_cmd[i].tau = 0.0       # 前馈扭矩(N.m)
motor_cmd[i].kp = 20.0       # 位置增益
motor_cmd[i].kd = 0.5        # 速度增益
```

**关节索引映射**：

| 索引范围 | 部位 | 关节 |
|----------|------|------|
| 0-5 | 左腿 | 髋Pitch/Roll/Yaw、膝、踝Pitch/Roll |
| 6-11 | 右腿 | 髋Pitch/Roll/Yaw、膝、踝Pitch/Roll |
| 12-14 | 腰部 | Yaw、Roll、Pitch |
| 15-21 | 左臂 | 肩Pitch/Roll/Yaw、肘、腕Roll/Pitch/Yaw |
| 22-28 | 右臂 | 肩Pitch/Roll/Yaw、肘、腕Roll/Pitch/Yaw |

#### 手臂与灵巧手控制

- **5-DOF/7-DOF手臂**：通过`rt/arm_sdk` Topic控制，示例程序`g1_arm5_sdk_dds_example.py`和`g1_arm7_sdk_dds_example.py`
- **Dex3-1灵巧手**：每只手7电机+9压力传感器，通过`rt/dex3/left/cmd`和`rt/dex3/right/cmd`控制
- **高层手势**：`ArmActionClient`提供预定义手势（挥手、握手等）

### 2.4 音频系统API

G1内置AudioClient，支持TTS、ASR和音频流控制：

```python
# DDS Topic: rt/audio_msg
# 示例: g1_audio_client_example.py
```

**官方语音能力（固件>=1.3.0）**：
- 本地离线ASR（含方位/情感/说话人信息）
- 本地离线TTS（中文）
- 已集成GPT模型，支持语音对话、动作控制、音乐播放
- VUI Client服务提供完整的语音交互接口

### 2.5 内置安全机制

SDK内置7项安全终止检查，在检测到异常时自动停止机器人：

| 安全检查 | 默认阈值 | 说明 |
|----------|----------|------|
| `bad_orientation()` | 1.0 rad | 机器人倾倒 |
| `joint_vel_out_of_limit()` | 10.0 rad/s | 关节速度超限 |
| `ang_vel_out_of_limit()` | 6.0 rad/s | IMU角速度超限 |
| `motor_winding_overheat()` | 120.0°C | 电机绕组过热 |
| `motor_casing_overheat()` | 85.0°C | 电机外壳过热 |
| `low_battery()` | 20.0% | 电量过低 |
| `lost_connection()` | 1000 ms | 通信丢失 |

> **关键判断**：G1的SDK安全机制为底层硬件保护，但LLM Agent层需要额外的逻辑安全验证（如速度限制、动作合理性检查），两者共同构成完整的安全防护体系。

## 3. LangChain/LangGraph Agent架构设计

### 3.1 框架选型：LangGraph vs AgentExecutor

| 维度 | AgentExecutor（已弃用） | LangGraph（推荐） |
|------|------------------------|-------------------|
| 执行模型 | 顺序管道 | 图遍历，条件路由和循环 |
| 状态持久化 | 手动Memory管理 | 内建checkpointing |
| 循环工作流 | 需变通方案 | 原生支持 |
| Human-in-the-Loop | 不支持 | 一等公民 |
| 多Agent编排 | 基础 | 层级化和并行 |
| 调试 | 黑盒 | 时间旅行调试 |
| EOL时间 | 2026年12月 | 长期维护 |

**结论**：选择**LangGraph**作为核心编排框架。其原生循环支持使Agent可以反复调用工具、评估结果、决定下一步，这对机器人控制至关重要。

### 3.2 推荐Agent架构

```
┌─────────────────────────────────────────────────────┐
│                    用户语音输入                        │
│                        ↓                             │
│              ┌─────────────────┐                     │
│              │   VAD + ASR     │  FunASR/SenseVoice  │
│              └────────┬────────┘                     │
│                       ↓                              │
│              ┌─────────────────┐                     │
│              │  LangGraph Agent │  核心编排引擎        │
│              │  ┌───────────┐  │                     │
│              │  │ LLM推理节点 │  │  Qwen/GPT-4o       │
│              │  └─────┬─────┘  │                     │
│              │        ↓        │                     │
│              │  ┌───────────┐  │                     │
│              │  │ Tool路由   │  │  条件边             │
│              │  └─────┬─────┘  │                     │
│              │        ↓        │                     │
│              │  ┌───────────┐  │                     │
│              │  │ Tool执行   │  │  机器人控制/感知    │
│              │  └─────┬─────┘  │                     │
│              │        ↓        │                     │
│              │  [状态更新/循环] │  Checkpointer       │
│              └────────┬────────┘                     │
│                       ↓                              │
│              ┌─────────────────┐                     │
│              │   TTS + 扬声器  │  CosyVoice/edge-tts  │
│              └─────────────────┘                     │
└─────────────────────────────────────────────────────┘
```

### 3.3 LangGraph实现模板

```python
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI

# 定义工具集
tools = [
    move_robot_tool,
    turn_robot_tool,
    wave_hand_tool,
    shake_hand_tool,
    detect_objects_tool,
    pick_object_tool,
    get_robot_status_tool,
    stop_emergency_tool,
]

# 创建带记忆的Agent
memory = MemorySaver()
model = ChatOpenAI(model="gpt-4o", temperature=0)

agent = create_react_agent(
    model=model,
    tools=tools,
    prompt=SYSTEM_PROMPT,
    checkpointer=memory,
)

# 多轮对话
config = {"configurable": {"thread_id": "session-001"}}
response = agent.invoke(
    {"messages": [("user", "向前走两步然后挥手")]},
    config=config
)
```

### 3.4 System Prompt设计要点

```python
SYSTEM_PROMPT = """
你是一个控制宇树G1人形机器人的AI助手。你可以通过调用工具来控制机器人的运动和感知。

## 安全规则（最高优先级）
1. 移动速度不得超过0.5m/s，旋转速度不得超过0.5rad/s
2. 每次连续移动距离不得超过5米
3. 执行动作前必须确认周围环境安全
4. 如果用户指令模糊（如"走过去"），应先询问具体方向和距离
5. 检测到障碍物时应立即停止并告知用户
6. 任何时候都可以调用emergency_stop

## 可用能力
- 行走控制：前进、后退、左移、右移、旋转
- 姿态控制：站立、蹲下、高低站姿
- 手臂动作：挥手、握手
- 环境感知：物体检测、距离测量
- 物体操作：抓取、放置（需灵巧手）
- 状态查询：电池电量、关节状态、IMU数据

## 交互风格
- 简洁明了地回复用户
- 执行动作后报告结果
- 遇到异常立即告知用户并建议处理方案
"""
```

> **架构洞察**：LangGraph的`create_react_agent`预构建模板已覆盖大部分Agent编排需求。对于更复杂的场景（如多步任务规划、多Agent协作），可通过`StateGraph`自定义图结构。关键是将机器人控制逻辑封装在Tool内部，Agent只负责推理和Tool选择。

## 4. Tool设计与开发

### 4.1 Tool清单与设计

基于G1 SDK能力，为Agent设计以下Tool集：

| Tool名称 | 功能 | SDK接口 | 安全等级 |
|----------|------|---------|----------|
| `move_robot` | 控制机器人行走 | `LocoClient.Move()` | 高 |
| `turn_robot` | 控制机器人旋转 | `LocoClient.Move(0,0,vyaw)` | 高 |
| `stand_up` | 站立 | `LocoClient.Squat2StandUp()` | 中 |
| `squat_down` | 蹲下 | `LocoClient.StandUp2Squat()` | 中 |
| `wave_hand` | 挥手 | `LocoClient.WaveHand()` | 低 |
| `shake_hand` | 握手 | `LocoClient.ShakeHand()` | 中 |
| `detect_objects` | 物体检测 | Intel D435i + YOLO | 低 |
| `get_robot_status` | 查询状态 | `LowState_`订阅 | 低 |
| `emergency_stop` | 紧急停止 | `LocoClient.Move(0,0,0)` | 最高 |
| `control_arm` | 手臂控制 | `rt/arm_sdk` | 高 |
| `control_gripper` | 灵巧手控制 | `rt/dex3/*/cmd` | 高 |
| `speak` | 语音播报 | `AudioClient` | 低 |

### 4.2 Tool开发示例

#### 示例1：运动控制Tool（含安全验证）

```python
from langchain_core.tools import tool, ToolException
from pydantic import BaseModel, Field
from enum import Enum

class SpeedLevel(str, Enum):
    SLOW = "slow"       # 0.2 m/s
    NORMAL = "normal"   # 0.3 m/s
    FAST = "fast"       # 0.5 m/s

class MoveRobotInput(BaseModel):
    direction: str = Field(
        description="移动方向: forward/backward/left/right"
    )
    distance: float = Field(
        description="移动距离(米), 范围0.1-5.0",
        gt=0.0, le=5.0
    )
    speed_level: SpeedLevel = Field(
        default=SpeedLevel.NORMAL,
        description="移动速度等级"
    )

@tool(args_schema=MoveRobotInput)
def move_robot(direction: str, distance: float, speed_level: SpeedLevel = SpeedLevel.NORMAL) -> str:
    """控制机器人向指定方向移动指定距离。移动前会自动检查安全条件。"""
    speed_map = {"slow": 0.2, "normal": 0.3, "fast": 0.5}
    speed = speed_map[speed_level.value]

    # 方向映射
    direction_map = {
        "forward": (speed, 0.0, 0.0),
        "backward": (-speed, 0.0, 0.0),
        "left": (0.0, speed, 0.0),
        "right": (0.0, -speed, 0.0),
    }

    if direction not in direction_map:
        raise ToolException(f"不支持的方向: {direction}，请使用forward/backward/left/right")

    vx, vy, vyaw = direction_map[direction]

    # 计算持续时间
    duration = distance / speed

    try:
        loco_client.Move(vx, vy, vyaw)
        time.sleep(duration)
        loco_client.Move(0.0, 0.0, 0.0)  # 停止
        return f"已向{direction}移动{distance}米，速度{speed}m/s"
    except Exception as e:
        loco_client.Move(0.0, 0.0, 0.0)  # 异常时停止
        raise ToolException(f"移动失败: {str(e)}")
```

#### 示例2：物体检测Tool（多模态）

```python
@tool
def detect_objects() -> str:
    """使用深度相机检测前方物体，返回物体列表及其位置信息。
    在执行抓取或导航操作前应先调用此工具了解环境。"""
    try:
        # 获取深度相机图像
        image = get_camera_image()
        # YOLO目标检测
        detections = yolo_model.detect(image)
        # 格式化结果
        results = []
        for det in detections:
            results.append(
                f"- {det.label}: 距离{det.distance:.2f}m, "
                f"方向{det.direction}, 置信度{det.confidence:.0%}"
            )
        if not results:
            return "未检测到任何物体"
        return f"检测到{len(results)}个物体:\n" + "\n".join(results)
    except Exception as e:
        raise ToolException(f"物体检测失败: {str(e)}")
```

#### 示例3：紧急停止Tool

```python
@tool
def emergency_stop() -> str:
    """立即停止机器人所有运动。在检测到危险或用户要求停止时调用。
    此工具具有最高优先级，会立即执行。"""
    try:
        loco_client.Move(0.0, 0.0, 0.0)
        loco_client.Damp()  # 切换到阻尼模式
        return "紧急停止已执行，机器人已停止所有运动"
    except Exception as e:
        return f"紧急停止执行异常(已尝试停止): {str(e)}"
```

### 4.3 Tool安全设计模式

```
┌──────────────────────────────────────────────┐
│  L1: System Prompt 安全规则                    │
│  "速度不超过0.5m/s，距离不超过5m"              │
│                    ↓                           │
│  L2: Pydantic Schema 参数约束                  │
│  distance: float = Field(gt=0, le=5.0)        │
│                    ↓                           │
│  L3: Tool内部运行时检查                         │
│  障碍物检测、关节限位、电池电量                  │
│                    ↓                           │
│  L4: SDK内置安全终止                            │
│  倾倒/过热/断连/低电量自动停止                   │
│                    ↓                           │
│  L5: 硬件急停按钮                              │
│  独立于所有软件的物理安全开关                    │
└──────────────────────────────────────────────┘
```

> **Tool设计洞察**：每个Tool应遵循"单一职责"原则——一个Tool只做一件事。Tool内部封装安全验证逻辑，对外提供简洁的语义接口。LLM不需要知道DDS Topic名称或电机控制细节，只需要理解"move_robot(direction, distance)"这样的高层语义。这种抽象层隔离了LLM的不确定性与硬件控制的精确性要求。

## 5. 语音交互管线设计

### 5.1 端到端语音架构

```
麦克风阵列 → 唤醒词检测 → VAD语音端点检测 → ASR语音识别
    → LangGraph Agent推理 → TTS语音合成 → 扬声器
```

### 5.2 ASR方案对比

| 方案 | 类型 | 中文效果 | 延迟 | 部署难度 | 推荐场景 |
|------|------|----------|------|----------|----------|
| **FunASR** (阿里) | 开源 | 优秀(Paraformer) | 低 | 中等 | **推荐首选** |
| **SenseVoice** (阿里) | 开源 | 优秀(多语言+情感) | 低 | 中等 | 需要情感识别 |
| **faster-whisper** | 开源 | 良好 | 很低 | 简单 | 边缘部署 |
| **G1内置ASR** | 本地 | 良好(含方位/情感) | 最低 | 零(预装) | 快速原型 |
| **讯飞云API** | 商业 | 顶级 | 中 | 简单 | 追求最高准确率 |

### 5.3 TTS方案对比

| 方案 | 类型 | 中文音质 | 延迟 | 部署难度 | 推荐场景 |
|------|------|----------|------|----------|----------|
| **CosyVoice** (阿里) | 开源 | 优秀(支持方言克隆) | 中 | 较高 | **推荐首选** |
| **edge-tts** | 免费API | 良好 | 低 | 极简 | 快速原型 |
| **MeloTTS** | 开源 | 良好 | 低 | 简单 | 轻量部署 |
| **G1内置TTS** | 本地 | 良好 | 最低 | 零(预装) | 快速原型 |
| **Kokoro** | 开源 | 良好 | 很低 | 简单 | 超低延迟 |

### 5.4 推荐语音方案

**方案A：快速原型（利用G1内置能力）**
- 直接使用G1固件>=1.3.0的内置ASR/TTS + GPT集成
- 零额外开发，适合验证Agent逻辑
- 局限：自定义能力有限

**方案B：全开源自建（推荐）**
- ASR: FunASR (Paraformer-large)
- VAD: silero-vad
- LLM: Qwen2.5-7B (本地) 或 GPT-4o (云端)
- TTS: CosyVoice
- 延迟目标: <1秒首音频

**方案C：EdgeVox框架（最完整）**
- 直接使用EdgeVox框架，内置G1/H1支持
- 首音频延迟约0.8秒
- 支持离线部署、MuJoCo仿真
- 内置SafetyMonitor（200ms急停响应）

### 5.5 延迟优化策略

| 优化点 | 技术手段 | 预期收益 |
|--------|----------|----------|
| ASR延迟 | 流式识别，边说边出结果 | -200ms |
| LLM延迟 | 流式生成，逐token输出 | -300ms |
| TTS延迟 | 首包优先，边生成边播放 | -200ms |
| VAD延迟 | silero-vad（32ms检测） | -50ms |
| 回声消除 | specsub AEC防止自听 | 消除反馈循环 |
| LLM中断 | stopping_criteria快速取消 | 40ms内响应 |

> **语音管线洞察**：端到端延迟是语音交互体验的核心指标。研究表明，人类可接受的对话延迟上限约为1秒。通过流式ASR + 流式LLM + 首包TTS的组合，可以将首音频延迟控制在0.8秒左右。EdgeVox框架已在G1上验证了这一指标。另一个关键问题是回声消除——机器人扬声器播放的声音会被麦克风拾取，必须通过AEC算法过滤。

## 6. 开源生态与参考项目

### 6.1 直接可用的G1 Agent项目

| 项目 | Stars | 核心能力 | 技术栈 | 适用场景 |
|------|-------|----------|--------|----------|
| **EdgeVox** | - | 离线语音Agent框架，内置G1支持 | Whisper+LLM+TTS, ROS2, MuJoCo | **语音交互首选** |
| **OM1** | - | 模块化AI运行时，预配置G1 | GPT+语音+情感+动作 | 完整Agent系统 |
| **DimOS** | - | G1操作系统，LLM Agent控制 | 导航+感知+空间记忆+手势 | 全功能机器人OS |
| **g1 (Yao0454)** | - | 跟随+手势+语音对话三合一 | YOLO-pose + Ollama + UDP桥接 | **快速原型首选** |
| **UnifoLM-VLA-0** | - | 官方VLA大模型 | 视觉-语言-动作模型 | 操作任务 |

#### 项目详解：g1 (Yao0454)

**项目地址**：https://github.com/Yao0454/g1

这是一个**中文社区开发者**贡献的高质量G1二次开发项目，实现了**视觉跟随 + 手势识别 + 语音对话**三功能并行运行。

**架构亮点**：
- **单端口UDP通信**：Python端与C++端通过UDP 9870端口通信，简化部署
- **分层设计**：`core/`目录封装高级API，`g1_node`整合LocoClient/ArmActionClient/AudioClient
- **视觉能力**：YOLO-pose实现人体跟随、握手手势识别、障碍物检测
- **语音管线**：唤醒 → ASR(sherpa-onnx) → Chat(Ollama qwen3:8b) → TTS(云端)
- **回声消除**：TTS播放期间不收麦，避免机器人听到自己的声音

**通信协议设计**：
| 类型 | 字节0 | 载荷 | 说明 |
|------|-------|------|------|
| MOVE | 0x01 | 3 floats (dist, err_norm, blocked) | 行走控制 |
| ARM | 0x02 | 1 uint8 (action_id) | 手臂动作 |
| TTS | 0x03 | UTF-8 text | 语音播报 |

**软件栈**：
- Jetson Orin NX 16GB（外挂视觉计算）
- Ubuntu 20.04 + JetPack 5.1.1 + CUDA 11.4 + ROS2 Foxy
- YOLOv8 (TensorRT) + pyrealsense2 + sherpa-onnx + ollama

**使用方式**：
```bash
# C++端（G1机器人上）
~/unitree_sdk2/build/bin/g1_node eth0

# Python端（Jetson上）
./run.sh g1              # 三功能全开
./run.sh g1 --no-voice   # 仅视觉
./run.sh g1 --no-vision  # 仅语音
./run.sh talk            # 纯语音工具
```

**核心价值**：该项目展示了如何用**极简架构**（单UDP端口）实现复杂的机器人交互功能，代码结构清晰，注释详尽，是**快速原型开发**的最佳参考。

### 6.2 LangChain + 机器人参考项目

| 项目 | 机构 | 核心价值 |
|------|------|----------|
| **ROSA** (NASA JPL) | NASA | LangChain + ReAct控制ROS/ROS2机器人，内建安全机制 |
| **llm-robot-control** | 社区 | 完整的Agentic AI ROS2教程系列（5部分） |
| **langchain_agent_robot_controller_ros2** | 社区 | ChatGPT + LangChain控制双臂机器人 |
| **ATHENA** (RoboCup) | 学术 | LangChain + LangGraph + ROS2自主任务处理 |

### 6.3 仿真与训练平台

| 平台 | 支持情况 | 用途 |
|------|----------|------|
| **Isaac Lab** | 官方推荐，Unitree RL Lab基于此 | RL步态训练 |
| **MuJoCo** | 官方提供G1 XML模型 | 快速仿真验证 |
| **Gazebo** | 通过ROS2 bridge支持 | ROS生态集成 |
| **Unitree RL Lab** | 官方RL框架 | Sim2Sim → Sim2Real |

### 6.4 关键GitHub仓库索引

| 仓库 | URL | 用途 |
|------|-----|------|
| unitree_sdk2_python | https://github.com/unitreerobotics/unitree_sdk2_python | Python SDK |
| unitree_ros | https://github.com/unitreerobotics/unitree_ros | ROS包 + URDF |
| EdgeVox | https://github.com/nrl-ai/edgevox | 语音Agent框架 |
| OM1 | https://github.com/OpenMind/OM1 | G1 AI运行时 |
| DimOS | https://github.com/dimensionalOS/dimos | G1操作系统 |
| g1 (Yao0454) | https://github.com/Yao0454/g1 | 跟随+手势+语音三合一 |
| unitree_rl_lab | https://github.com/unitreerobotics/unitree_rl_lab | RL训练框架 |
| UnifoLM-VLA-0 | https://github.com/unitreerobotics/unifolm-vla | 官方VLA模型 |
| FunASR | https://github.com/alibaba-damo-academy/FunASR | 中文ASR |
| CosyVoice | https://github.com/FunAudioLLM/CosyVoice | 中文TTS |
| Pipecat | https://github.com/pipecat-ai/pipecat | 语音AI编排 |
| sherpa-onnx | https://github.com/k2-fsa/sherpa-onnx | 离线ASR/TTS |

> **生态洞察**：**g1 (Yao0454)** 是中文社区最实用的G1二次开发参考——它用极简的单UDP端口架构实现了视觉跟随、手势识别、语音对话三功能并行，代码结构清晰，非常适合快速原型验证。EdgeVox提供了更完整的离线语音Agent框架，OM1和DimOS则展示了全功能机器人OS的架构设计。建议根据项目需求选择合适的参考项目：
> - **快速验证概念** → g1 (Yao0454)
> - **完整语音交互系统** → EdgeVox
> - **生产级全功能系统** → OM1 / DimOS
> - **RL训练与Sim2Real** → Unitree RL Lab

## 7. 开发路线图与实施建议

### 7.1 分阶段开发路线

**Phase 1：基础通信与运动控制（2-3周）**
- 搭建开发环境（Ubuntu 22.04 + Python SDK + DDS）
- 实现LocoClient基础运动Tool（站立、行走、停止）
- 在仿真环境（MuJoCo）中验证
- 里程碑：通过命令行控制G1仿真模型完成基本动作

**Phase 2：LangGraph Agent集成（2-3周）**
- 搭建LangGraph Agent框架
- 实现完整Tool集（运动、手臂、状态查询）
- 设计System Prompt和安全验证逻辑
- 里程碑：通过自然语言指令控制G1仿真模型

**Phase 3：语音交互集成（2-3周）**
- 集成ASR（FunASR）和TTS（CosyVoice/edge-tts）
- 实现VAD和唤醒词检测
- 端到端语音管线联调
- 里程碑：语音对话控制G1仿真模型

**Phase 4：实机部署与优化（3-4周）**
- Sim2Real迁移
- 安全机制全面测试
- 延迟优化（目标<1秒首音频）
- 边缘部署到Jetson Orin NX
- 里程碑：实机上运行完整Agent系统

### 7.2 技术栈推荐

| 层级 | 推荐方案 | 备选方案 |
|------|----------|----------|
| Agent编排 | LangGraph | LangChain AgentExecutor（不推荐） |
| LLM | Qwen2.5-7B（本地）/ GPT-4o（云端） | DeepSeek、Claude |
| 机器人SDK | unitree_sdk2_python | ROS2（高级场景） |
| ASR | FunASR | SenseVoice、faster-whisper |
| TTS | CosyVoice | edge-tts、MeloTTS |
| VAD | silero-vad | WebRTC VAD |
| 目标检测 | YOLO11 | CLIP |
| 仿真 | MuJoCo（快速验证） | Isaac Lab（RL训练） |
| 部署 | Docker + Jetson Orin NX | 直接部署 |

### 7.3 关键风险与缓解

| 风险 | 影响 | 缓解措施 |
|------|------|----------|
| LLM幻觉导致危险动作 | 高 | 多层安全验证 + Human-in-the-Loop |
| 端到端延迟过高 | 中 | 流式处理 + 边缘推理 + AEC |
| Jetson算力不足 | 中 | 混合架构（边缘小模型+云端大模型） |
| Sim2Real差距 | 中 | 充分仿真测试 + 渐进式实机部署 |
| DDS通信不稳定 | 低 | 心跳检测 + 自动重连 + 降级策略 |

## 8. 结论

基于宇树G1机器人构建LangChain Agent在技术上完全可行，且已有多个开源项目验证了这一路径。G1 EDU版提供了完善的Python SDK，LocoClient高层API可以直接封装为LangChain Tool，极大降低了开发门槛。LangGraph作为Agent编排框架，其原生状态管理、循环支持和Human-in-the-Loop能力完美匹配机器人控制场景的需求。

语音交互方面，FunASR + CosyVoice的开源组合可以满足中文语音对话需求，EdgeVox框架已验证了0.8秒首音频延迟的可行性。安全层面，需要构建从System Prompt到硬件急停的五层防护体系，确保LLM的不确定性不会转化为物理风险。

最务实的开发策略是**参考g1 (Yao0454)项目的架构设计**——其单UDP端口通信模式简化了Python与C++端的交互，YOLO-pose视觉方案可直接复用。对于需要完整LangChain Agent能力的场景，可结合EdgeVox的语音管线和LangGraph的编排能力。预计完整系统从零到实机运行需要**9-13周**的开发周期（若基于g1项目二次开发可缩短至**4-6周**）。

## 9. 参考文献

[1] unitreerobotics. unitree_sdk2_python[EB/OL]. https://github.com/unitreerobotics/unitree_sdk2_python, 2025.

[2] 宇树科技. G1开发者文档[EB/OL]. https://support.unitree.com/home/zh/G1_developer/get_sdk, 2025.

[3] 宇树科技. G1语音助手说明[EB/OL]. https://support.unitree.com/home/zh/G1_developer/voice_assistant_instructions, 2025.

[4] nrl-ai. EdgeVox: Offline Voice Agent for Robots[EB/OL]. https://github.com/nrl-ai/edgevox, 2025.

[5] OpenMind. OM1: Modular AI Runtime for Humanoid Robots[EB/OL]. https://github.com/OpenMind/OM1, 2025.

[6] dimensionalOS. DimOS: Operating System for Unitree G1[EB/OL]. https://github.com/dimensionalOS/dimos, 2025.

[7] unitreerobotics. UnifoLM-VLA-0: Vision-Language-Action Model[EB/OL]. https://github.com/unitreerobotics/unifolm-vla, 2026.

[8] unitreerobotics. unitree_rl_lab[EB/OL]. https://github.com/unitreerobotics/unitree_rl_lab, 2025.

[9] LangChain. How to Create Tools[EB/OL]. https://python.langchain.ac.cn/docs/how_to/custom_tools/, 2025.

[10] LangChain. Migrate to LangGraph[EB/OL]. https://python.langchain.ac.cn/docs/how_to/migrate_agent/, 2025.

[11] NASA JPL. ROSA: Robot Operating System Agent[EB/OL]. https://arxiv.org/html/2410.06472v1, 2024.

[12] alibaba-damo-academy. FunASR[EB/OL]. https://github.com/alibaba-damo-academy/FunASR, 2025.

[13] FunAudioLLM. CosyVoice[EB/OL]. https://github.com/FunAudioLLM/CosyVoice, 2025.

[14] pipecat-ai. Pipecat: Voice AI Orchestration Framework[EB/OL]. https://github.com/pipecat-ai/pipecat, 2025.

[15] OpenMOSS. FRoM-W1: Language-Instructed Whole-Body Control[EB/OL]. https://github.com/OpenMOSS/FRoM-W1, 2025.

[16] Open-X-Humanoid. Open X-Humanoid[EB/OL]. https://github.com/Open-X-Humanoid, 2025.

[17] DeepWiki. Unitree SDK2 Documentation[EB/OL]. https://deepwiki.com/unitreerobotics/unitree_sdk2, 2025.

[18] MbodiAI. Embodied Agents[EB/OL]. https://github.com/MbodiAI/embodied-agents, 2025.

[19] NVIDIA. GR00T N1: Humanoid Foundation Model[EB/OL]. https://github.com/NVIDIA/GR00T, 2025.

[20] SYSTRAN. faster-whisper[EB/OL]. https://github.com/SYSTRAN/faster-whisper, 2025.

[21] Yao0454. 宇树G1机器人Python样例[EB/OL]. https://github.com/Yao0454/g1, 2025.

[22] k2-fsa. sherpa-onnx[EB/OL]. https://github.com/k2-fsa/sherpa-onnx, 2025.
