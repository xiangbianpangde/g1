# G1 Agent 与 Tool 文件改进 Spec

## Why

当前 `agent.py` 和 `tools.py` 存在以下问题：
1. `agent.py` 使用 `langchain.agents.create_agent`（基于旧 AgentExecutor），而开发指南明确推荐使用 LangGraph 的 `create_react_agent`
2. `agent.py` 通过 `init_chat_model` 创建模型，未复用项目中成熟的 `core.Chat`（已有 ollama 对话封装、历史持久化、代理绕过）
3. `tools.py` 中的 `detect_objects` 工具在非模拟模式下仅返回占位信息，未利用项目中 `core.Vision` 已有的 YOLO-pose + RealSense 深度检测能力
4. `tools.py` 中的 `get_robot_status` 工具在非模拟模式下仅返回"需要 DDS 连接"，未提供有用信息
5. 缺少对 `core.Voice` 语音模块的工具集成（如语音唤醒/语音输入）

## What Changes

- **`agent.py`**：改用 `langgraph.prebuilt.create_react_agent` 替代 `langchain.agents.create_agent`；复用项目 `core.Chat` 做 LLM 调用；使用 LangGraph 原生 SqliteSaver
- **`tools.py`**：`detect_objects` 集成 `core.Vision` 的真实检测能力；`get_robot_status` 提供有意义的 bridge 健康检查；工具工厂增加 `CoreComponents` 参数以接收可选的核心组件

## Impact

- Affected specs: 无（新建项目）
- Affected code: `agent.py`, `tools.py`
- 不涉及 **BREAKING** 变更——现有 CLI 参数和模拟模式完全兼容

## ADDED Requirements

### Requirement: Agent 使用 LangGraph create_react_agent
Agent SHALL 使用 `langgraph.prebuilt.create_react_agent` 替代已弃用的 `langchain.agents.create_agent`，保持 SqliteSaver 检查点持久化。

#### Scenario: Agent 创建成功
- **WHEN** 调用 `create_g1_agent(bridge, simulation=False)`
- **THEN** 返回的 agent 对象是基于 `create_react_agent` 的 LangGraph 图，支持 tool calling 和状态持久化

#### Scenario: 模拟模式正常运行
- **WHEN** 以 `--sim` 参数启动
- **THEN** Agent 正确初始化，工具调用走模拟路径，交互模式正常工作

### Requirement: 复用 core.Chat 进行 LLM 调用
Agent SHALL 复用项目中成熟的 `core.Chat` 类（ollama 封装），而非单独通过 `init_chat_model` 创建模型实例。

#### Scenario: LLM 调用使用 core.Chat
- **WHEN** Agent 需要调用 LLM 进行推理
- **THEN** 底层使用 `core.Chat` 实例（复用其代理绕过、历史持久化等成熟逻辑）

### Requirement: detect_objects 工具使用 core.Vision 真实检测
`detect_objects` 工具在非模拟模式下 SHALL 利用 `core.Vision` 的 YOLO-pose 推理和 RealSense 深度数据检测前方物体，返回真实检测结果。

#### Scenario: 实机模式下物体检测
- **WHEN** Agent 调用 `detect_objects` 且 `simulation=False` 且 Vision 组件可用
- **THEN** 返回基于 RealSense 深度图 ROI 分析的真实障碍物距离信息

#### Scenario: 模拟模式下物体检测
- **WHEN** Agent 调用 `detect_objects` 且 `simulation=True`
- **THEN** 返回模拟数据，行为与当前一致

#### Scenario: Vision 不可用时的优雅降级
- **WHEN** Agent 调用 `detect_objects` 且 Vision 组件未初始化（如缺 RealSense）
- **THEN** 返回明确的降级消息"视觉模块未就绪，请连接 RealSense 相机"

### Requirement: get_robot_status 提供有意义的 bridge 健康检查
`get_robot_status` 工具在非模拟模式下 SHALL 至少提供 bridge 连接状态检查，而非仅返回"需要 DDS 连接"。

#### Scenario: 实机模式下的状态查询
- **WHEN** Agent 调用 `get_robot_status` 且 `simulation=False`
- **THEN** 返回 bridge 连接状态（主机/端口）+ 模拟的关节/电量信息（标注为占位数据）

#### Scenario: 模拟模式下的状态查询
- **WHEN** Agent 调用 `get_robot_status` 且 `simulation=True`
- **THEN** 返回占位模拟数据，行为不变

## MODIFIED Requirements

### Requirement: 工具集工厂函数参数扩展
`create_robot_tools(bridge, simulation)` SHALL 增加可选参数 `CoreComponents` 以接收 `core.Vision` 等核心组件，使 detect_objects 等工具能使用真实的感知能力。

#### Scenario: 不传 CoreComponents 时向后兼容
- **WHEN** 调用 `create_robot_tools(bridge, simulation=True)` 不传 CoreComponents
- **THEN** 行为与当前完全一致，检测工具走模拟路径

#### Scenario: 传入 CoreComponents 时启用真实检测
- **WHEN** 调用 `create_robot_tools(bridge, simulation=False, core=CoreComponents(vision=vision_instance))`
- **THEN** detect_objects 使用 vision_instance 进行真实感知
