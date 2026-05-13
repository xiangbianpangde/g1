# Tasks

- [x] Task 1: 创建 CoreComponents 容器类
  - 在 `tools.py` 中新增 `CoreComponents` 数据类，包含可选的 `vision: Vision | None` 字段
  - 修改 `create_robot_tools` 函数签名，增加 `core: CoreComponents | None = None` 参数
  - 将 `CoreComponents` 导出到模块公共接口

- [x] Task 2: 改造 detect_objects 工具使用 core.Vision
  - 在非模拟模式下，当 `core.vision` 可用时，调用 `vision._check_obstacles` 获取左中右三区域深度
  - 将深度数据格式化为人类可读的检测结果字符串
  - 当 Vision 不可用时返回优雅的降级消息
  - 保持模拟模式行为不变

- [x] Task 3: 改造 get_robot_status 工具提供有意义的 bridge 状态
  - 在非模拟模式下，至少返回 bridge 连接目标（host:port）和连接状态
  - 补充占位的机器人状态数据（标注为占位/估算）
  - 保持模拟模式行为不变

- [x] Task 4: 改造 agent.py 使用 LangGraph create_react_agent
  - 将 `from langchain.agents import create_agent` 替换为 `from langgraph.prebuilt import create_react_agent`
  - 调整 agent 创建逻辑适配新 API
  - 保持 SqliteSaver checkpoint 持久化
  - 确保 System Prompt、Tool 集、checkpointer 正确传入
  - 适配消息结构（LangGraph 的 input/output schema）

- [x] Task 5: 改造 agent.py 复用 core.Chat
  - 将 agent 中的 LLM 调用底层改为使用 `core.Chat` 实例
  - LangGraph 的 `create_react_agent` 需要一个 LangChain 兼容的 model 对象，`core.Chat` 是基于 ollama 库的封装
  - 保持 agent 使用 `init_chat_model`（Ollama provider），但复用 `core.Chat` 的模型名、代理配置等约定
  - 更新 `create_g1_agent` 函数签名，允许传入可选的 `Chat` 实例

- [x] Task 6: 更新 agent.py CLI 接口以支持 CoreComponents 传入
  - 在 `main()` 中初始化 `Vision` 实例（仅在非模拟模式下）
  - 将 Vision 传递给 `CoreComponents` 和工具集
  - 添加 `--vision` 参数以启用视觉组件（默认关闭，不影响现有用法）
  - 保持所有现有 CLI 参数兼容

# Task Dependencies

- Task 2 依赖 Task 1（CoreComponents 容器先行）✅
- Task 3 依赖 Task 1（CoreComponents 容器先行）✅
- Task 6 依赖 Task 1, Task 4, Task 5（agent 和 tools 改造完成后，CLI 整合）✅
- Task 4 和 Task 5 无互相依赖，可并行 ✅
- Task 2 和 Task 3 无互相依赖，可并行 ✅
