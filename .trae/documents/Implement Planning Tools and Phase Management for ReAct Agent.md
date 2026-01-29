## 1. 扩展数据模型与初始化

* 在 `ReActAgentStrategy` 类中添加 `plans` 列表和 `agent_state` 成员变量。

* 在 `_invoke` 开始时初始化 `self.plans = []` 和 `self.agent_state = "INITIAL"`。

## 2. 定义内置工具实体

* 创建内置工具的定义，包括 `generate_plan` (接收计划列表), `update_plans` (批量增删改), `complete_plan` (完成当前项)。

* 实现 `_get_built_in_tools()` 方法，根据当前状态返回对应的内置工具列表。

## 3. 实现计划管理逻辑

* **`_handle_generate_plan`**: 接收输入，创建带 ID 的计划项，状态设为 `pending`。

* **`_handle_complete_plan`**: 查找第一个 `pending` 项设为 `completed`；若全部完成，状态切至 `FINISHED`。

* **`_handle_update_plans`**: 实现对待办任务的批量操作。使用 ID 索引，操作后重新对 `pending` 任务进行排序和编号，确保 ID 连续且不影响已完成的任务。

## 4. 动态提示词与工具过滤

* 修改 `_system_prompt_message`:

  * 根据 `agent_state` 决定 `{{tools}}` 的内容。

  * 在 `EXECUTING` 状态下，拼接计划看板字符串（已完成/当前/待办）并注入。

* 修改 `_organize_prompt_messages`: 确保计划信息能正确传递到 LLM。

## 5. 拦截并执行工具

* 在 `_handle_invoke_action` 中添加逻辑：

  * 检查 `tool_name` 是否属于内置工具。

  * 若是，调用本地处理方法并返回 Observation 结果。

  * 若不是，继续执行原有的 Dify 工具调用逻辑。

## 6. 验证与测试

* 验证从 `INITIAL` 到 `EXECUTING` 再到 `FINISHED` 的完整流转。

* 测试 `update_plans` 在各种增删改组合下，编号是否依然保持连续且逻辑正确。

* 确认 Dify 原有工具在 `EXECUTING` 阶段仍能正常配合使用。

