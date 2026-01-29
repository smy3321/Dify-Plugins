import json
import time
import concurrent.futures
import threading
from collections.abc import Generator, Mapping
from typing import Any, Optional, cast
from enum import Enum

import pydantic
from pydantic import BaseModel, Field

from dify_plugin.entities.agent import AgentInvokeMessage
from dify_plugin.entities.model.llm import LLMModelConfig, LLMUsage
from dify_plugin.entities.model.message import (
    AssistantPromptMessage,
    PromptMessage,
    SystemPromptMessage,
    UserPromptMessage,
)
from dify_plugin.entities.tool import (
    ToolDescription,
    ToolIdentity,
    ToolInvokeMessage,
    ToolParameter,
    ToolProviderType,
)

from dify_plugin.interfaces.agent import (
    AgentModelConfig,
    AgentScratchpadUnit,
    AgentStrategy,
    AgentToolIdentity,
    ToolEntity,
)
from output_parser.cot_output_parser import ReactChunk, ReactState, CotAgentOutputParser
from prompt.template import REACT_PROMPT_TEMPLATES


class PlanStatus(str, Enum):
    """
    计划项状态枚举。
    """
    TODO = "todo"
    DOING = "doing"
    DONE = "done"


class PlanItem(BaseModel):
    """
    计划项模型。
    """
    id: int
    content: str
    status: PlanStatus = PlanStatus.TODO


class AgentWorkflowState(str, Enum):
    """
    Agent 工作流状态枚举。
    """
    START = "start"
    EXECUTING = "executing"
    FINISHED = "finished"


class LogMetadata:
    """
    日志元数据键名定义。
    这些常量用于在日志记录中标识特定的元数据字段，
    例如开始时间、耗时、Token消耗、价格等，方便后续的监控和计费统计。
    """
    STARTED_AT = "started_at"    # 开始时间
    PROVIDER = "provider"        # 模型提供商
    FINISHED_AT = "finished_at"  # 结束时间
    ELAPSED_TIME = "elapsed_time" # 耗时（秒）
    TOTAL_PRICE = "total_price"  # 总费用
    CURRENCY = "currency"        # 货币单位
    TOTAL_TOKENS = "total_tokens" # 总 Token 数

# 某些模型提供商（如 wenxin）可能不支持 Observation 作为停止词，需特殊处理
ignore_observation_providers = ["wenxin"]


class ReActParams(BaseModel):
    """
    ReAct 策略的输入参数模型。
    定义了执行 ReAct 策略所需的所有必要信息。
    """
    instruction: str            # 系统指令/角色设定
    model: AgentModelConfig     # 模型配置（包括模型名称、参数等）
    tools: list[ToolEntity] | None # 可用的工具列表
    maximum_iterations: int = 3 # 最大迭代次数，防止无限循环（默认3次）
    enable_planner: bool = True # 是否启用计划管理工具


class AgentPromptEntity(BaseModel):
    """
    Agent 提示词实体。
    用于存储 ReAct 策略中使用的提示词模板。
    """
    first_prompt: str   # 初始提示词模板
    next_iteration: str # 下一轮迭代的提示词模板（通常包含 scratchpad 占位符）


class ReActAgentStrategy(AgentStrategy):
    """
    ReAct (Reasoning and Acting) Agent 策略实现类。
    
    ReAct 是一种结合了推理（Reasoning）和行动（Acting）的 Agent 范式。
    它允许模型生成推理轨迹（Thought），基于推理决定调用工具（Action），
    并根据工具执行结果（Observation）进行下一步推理，直到得出最终答案。
    """
    instruction: str = ""
    # 历史对话消息，用于保持多轮对话的上下文
    history_prompt_messages: list[PromptMessage] = Field(default_factory=list)
    # 转换为 Prompt 格式的工具列表
    prompt_messages_tools: list[ToolEntity] = Field(default_factory=list)
    # 计划列表
    plans: list[PlanItem] = Field(default_factory=list)
    # 工作流状态
    workflow_state: AgentWorkflowState = Field(default=AgentWorkflowState.START)
    # 记忆压缩摘要
    compressed_summary: str = Field(default="")

    def _get_builtin_tools(self) -> list[ToolEntity]:
        """
        获取内置工具列表。
        """
        builtin_tools = []
        
        # 1. 计划生成工具
        plan_gen_tool = ToolEntity(
            identity=AgentToolIdentity(
                name="plan_generation_tool",
                author="builtin",
                label={"en_US": "Plan Generation", "zh_Hans": "计划生成"},
                provider="builtin"
            ),
            description=ToolDescription(
                llm="Generate a high-quality, executable work plan based on the task complexity. Use this tool directly when starting. Do not over-explain the planning process in Thought. Each step should be concrete and actionable. NOTE: Only include execution steps. DO NOT include steps like 'Question Summary' or 'Conclusion Generation', as Conclusion step will be auto-generated at the end. A good plan should avoid ambiguity and ensure task success.",
                human={"en_US": "Generate a work plan", "zh_Hans": "生成工作计划"}
            ),
            parameters=[
                ToolParameter(
                    name="plans",
                    label={"en_US": "Plans", "zh_Hans": "计划列表"},
                    human_description={"en_US": "A list of 2-5 strings representing the steps to complete the task.", "zh_Hans": "包含2-5个步骤的任务计划列表"},
                    type=ToolParameter.ToolParameterType.ARRAY,
                    form=ToolParameter.ToolParameterForm.LLM,
                    required=True,
                    llm_description="A list of 2-5 concrete, actionable execution steps. Ensure each step is valuable for achieving the goal. Exclude final response or summary steps."
                )
            ]
        )
        builtin_tools.append(plan_gen_tool)
        
        # 2. 计划批量修改工具
        plan_update_tool = ToolEntity(
            identity=AgentToolIdentity(
                name="plan_batch_update_tool",
                author="builtin",
                label={"en_US": "Plan Batch Update", "zh_Hans": "计划批量修改"},
                provider="builtin"
            ),
            description=ToolDescription(
                llm="Optimize or adjust future plans whenever you believe the current plan is no longer optimal or needs refinement. Use this tool proactively and silently; focus on the task content in Thought, not the act of updating the plan. Supports 'update', 'delete', and 'add' actions.",
                human={"en_US": "Batch update plans", "zh_Hans": "批量修改计划"}
            ),
            parameters=[
                ToolParameter(
                    name="actions",
                    label={"en_US": "Actions", "zh_Hans": "修改操作列表"},
                    human_description={
                        "en_US": "List of actions: \n- update: {'type': 'update', 'id': int, 'content': str}\n- delete: {'type': 'delete', 'id': int}\n- add: {'type': 'add', 'content': str, 'after_id'?: int}", 
                        "zh_Hans": "操作列表：\n- update: {'type': 'update', 'id': int, 'content': str}\n- delete: {'type': 'delete', 'id': int}\n- add: {'type': 'add', 'content': str, 'after_id'?: int}"
                    },
                    type=ToolParameter.ToolParameterType.ARRAY,
                    form=ToolParameter.ToolParameterForm.LLM,
                    required=True,
                    llm_description=(
                        "List of actions to refine the plan. Each action must be a dict with a 'type' field:\n"
                        "1. update: {'type': 'update', 'id': <plan_id>, 'content': <new_content>} - Update an existing plan item.\n"
                        "2. delete: {'type': 'delete', 'id': <plan_id>} - Delete an existing plan item (only pending items).\n"
                        "3. add: {'type': 'add', 'content': <new_content>, 'after_id': <id_to_insert_after>} - Add a new plan item. 'after_id' is optional and defaults to the current step ID.\n"
                        "Only execution steps are allowed in 'content'."
                    )
                )
            ]
        )
        builtin_tools.append(plan_update_tool)
        
        # 3. 计划完成工具
        plan_complete_tool = ToolEntity(
            identity=AgentToolIdentity(
                name="plan_completion_tool",
                author="builtin",
                label={"en_US": "Plan Completion", "zh_Hans": "计划完成"},
                provider="builtin"
            ),
            description=ToolDescription(
                llm="Mark the current plan item as completed and move to the next one. Use this tool IMMEDIATELY and SILENTLY after completing the current step. You MUST NOT mention or explain the transition in Thought; keep Thought block strictly focused on the task execution content.",
                human={"en_US": "Complete current plan", "zh_Hans": "完成当前计划"}
            ),
            parameters=[
                ToolParameter(
                    name="ids",
                    label={"en_US": "Plan IDs", "zh_Hans": "任务ID列表"},
                    human_description={"en_US": "List of plan IDs to mark as completed. If omitted, only the current plan is marked.", "zh_Hans": "要标记为完成的任务ID列表。若省略，仅标记当前任务。"},
                    type=ToolParameter.ToolParameterType.ARRAY,
                    form=ToolParameter.ToolParameterForm.LLM,
                    required=False,
                    llm_description="Optional list of plan IDs to mark as completed. Must include the current step ID if provided. Use this when you have completed multiple steps at once."
                )
            ]
        )
        builtin_tools.append(plan_complete_tool)

        # 4. 记忆压缩工具
        memory_compression_tool = ToolEntity(
            identity=AgentToolIdentity(
                name="memory_compression_tool",
                author="builtin",
                label={"en_US": "Memory Compression", "zh_Hans": "记忆压缩"},
                provider="builtin"
            ),
            description=ToolDescription(
                llm="Compress the previous reasoning scratchpad into a concise summary to save context space. Use this when the history is long or less relevant.",
                human={"en_US": "Compress memory", "zh_Hans": "压缩记忆"}
            ),
            parameters=[
                ToolParameter(
                    name="summary",
                    label={"en_US": "Summary", "zh_Hans": "摘要内容"},
                    human_description={"en_US": "A comprehensive summary of the previous thoughts, actions, and observations.", "zh_Hans": "对之前思考、行动和观察的全面摘要"},
                    type=ToolParameter.ToolParameterType.STRING,
                    form=ToolParameter.ToolParameterForm.LLM,
                    required=True,
                    llm_description="A comprehensive summary of the previous thoughts, actions, and observations. Keep important parts detailed and unimportant parts concise."
                )
            ]
        )
        builtin_tools.append(memory_compression_tool)
        
        return builtin_tools

    @property
    def _user_prompt_message(self) -> UserPromptMessage:
        """
        构造用户消息对象。
        
        将指令（instruction）封装为 UserPromptMessage 对象。
        """
        return UserPromptMessage(content=self.instruction)

    def _rebuild_plans(self):
        """
        重新编号并维护计划状态。
        """
        if not self.plans:
            return
        
        # 重新分配 ID
        for i, plan in enumerate(self.plans):
            plan.id = i + 1
            
        # 确保只有一个 DOING，或者如果没有 DOING 则设置第一个 TODO 为 DOING
        doing_count = sum(1 for p in self.plans if p.status == PlanStatus.DOING)
        if doing_count == 0:
            for plan in self.plans:
                if plan.status == PlanStatus.TODO:
                    plan.status = PlanStatus.DOING
                    break
        elif doing_count > 1:
            # 这种情况下只保留第一个 DOING
            found_first = False
            for plan in self.plans:
                if plan.status == PlanStatus.DOING:
                    if not found_first:
                        found_first = True
                    else:
                        plan.status = PlanStatus.TODO
        
        # 检查是否全部完成
        if all(p.status == PlanStatus.DONE for p in self.plans):
            self.workflow_state = AgentWorkflowState.FINISHED
        else:
            self.workflow_state = AgentWorkflowState.EXECUTING

    def _handle_plan_generation(self, plans: list[str]) -> str:
        """
        处理计划生成。
        """
        # 健壮性处理：若为字符串，尝试按逗号切分；若为单键值对 dict，则提取值
        if isinstance(plans, str):
            plans = [p.strip() for p in plans.split(",") if p.strip()]
        elif isinstance(plans, dict) and len(plans) == 1:
            # 单键值对，提取值
            plans = list(plans.values())[0]
            # 若提取后仍是字符串，继续切分
            if isinstance(plans, str):
                plans = [p.strip() for p in plans.split(",") if p.strip()]
        if not isinstance(plans, list):
            return "Failed to generate plans: invalid input format."

        self.plans = [
            PlanItem(id=i+1, content=content, status=PlanStatus.TODO)
            for i, content in enumerate(plans)
        ]
        if self.plans:
            self.plans[0].status = PlanStatus.DOING
            self.workflow_state = AgentWorkflowState.EXECUTING
            return f"Successfully generated {len(self.plans)} plans. Current plan is: {self.plans[0].content}"
        return "Failed to generate plans."

    def _handle_plan_batch_update(self, actions: list[dict]) -> str:
        """
        处理计划批量修改。
        """
        current_doing_id = 0
        for p in self.plans:
            if p.status == PlanStatus.DOING:
                current_doing_id = p.id
                break
        
        new_plans = list(self.plans)
        
        # 1. delete
        # 仅允许删除待办任务
        deleted_ids = {int(a["id"]) for a in actions if a["type"] == "delete" and "id" in a}
        new_plans = [p for p in new_plans if p.id not in deleted_ids or p.status == PlanStatus.DONE]
        
        # 2. update
        updates = {int(a["id"]): a["content"] for a in actions if a["type"] == "update" and "id" in a and "content" in a}
        for p in new_plans:
            if p.id in updates:
                p.content = updates[p.id]
        
        # 3. add
        # add 则在某个编号计划之后添加若干任务（必须大于等于在当前计划编号）
        adds = [a for a in actions if a["type"] == "add" and "content" in a]
        for add_action in adds:
            after_id = int(add_action.get("after_id", current_doing_id))
            if after_id < current_doing_id:
                after_id = current_doing_id # 强制纠正
                
            content = add_action["content"]
            # 寻找插入位置
            insert_pos = len(new_plans)
            for i, p in enumerate(new_plans):
                if p.id == after_id:
                    insert_pos = i + 1
                    break
            new_plans.insert(insert_pos, PlanItem(id=0, content=content, status=PlanStatus.TODO))
        
        self.plans = new_plans
        self._rebuild_plans()
        return "Plans updated successfully. Please check the new plan sequence in the next thought."

    def _handle_plan_completion(self, ids: list[int] | None = None) -> str:
        """
        处理计划完成。
        """
        # 找到当前正在进行的任务
        current_doing_plan = next((p for p in self.plans if p.status == PlanStatus.DOING), None)
        
        if not current_doing_plan:
            return "No current plan is being executed."

        if not ids:
            # 原有逻辑：完成当前任务
            current_doing_plan.status = PlanStatus.DONE
        else:
            # 兼容性处理：如果 ids 是字符串，尝试解析
            if isinstance(ids, str):
                try:
                    ids = json.loads(ids)
                except Exception:
                    pass

            # 再次确认 ids 是列表
            if not isinstance(ids, list):
                return "Error: 'ids' must be a list of integers."

            # 校验 ids
            # 1. 必须包含当前任务 ID
            if current_doing_plan.id not in ids:
                return f"Error: The provided 'ids' list must include the current step ID ({current_doing_plan.id})."
            
            # 2. 筛选有效 ID：DOING 或 TODO
            # 注意：DONE 的任务被视为无效（或者说是“已经完成”的），如果只有 DONE 的任务则报错
            valid_ids = []
            has_already_done = False
            
            # 建立 ID 映射以便快速查找
            plan_map = {p.id: p for p in self.plans}
            
            target_plans = []
            
            for plan_id in ids:
                if plan_id not in plan_map:
                    continue # 忽略不存在的 ID
                
                plan = plan_map[plan_id]
                if plan.status == PlanStatus.DONE:
                    has_already_done = True
                elif plan.status in (PlanStatus.DOING, PlanStatus.TODO):
                    target_plans.append(plan)
            
            if not target_plans:
                if has_already_done:
                    return "Error: All provided plan IDs are already completed."
                else:
                    return "Error: No valid plan IDs found to complete."
            
            # 执行完成操作
            for plan in target_plans:
                plan.status = PlanStatus.DONE

        self._rebuild_plans()
        
        if self.workflow_state == AgentWorkflowState.FINISHED:
            return "All plans completed!"
        
        current_plan = next((p for p in self.plans if p.status == PlanStatus.DOING), None)
        return f"Plans marked as completed. Next plan is: {current_plan.content if current_plan else 'None'}"

    def _handle_memory_compression(self, agent_scratchpad: list, summary: str) -> str:
        """
        处理记忆压缩。
        """
        self.compressed_summary = summary
        agent_scratchpad.clear()
        return "Memory compressed successfully. Previous scratchpad has been replaced by this summary."

    def _get_dynamic_system_prompt(self, tools: list[ToolEntity]) -> SystemPromptMessage:
        """
        根据当前状态和可用工具生成动态系统提示词。
        """
        prompt_entity = AgentPromptEntity(
            first_prompt=REACT_PROMPT_TEMPLATES["english"]["chat"]["prompt"],
            next_iteration=REACT_PROMPT_TEMPLATES["english"]["chat"][
                "agent_scratchpad"
            ],
        )
        
        # 核心指令
        instruction = "Respond to the human as helpfully and accurately as possible. All your thoughts and final answers MUST be in Chinese.\n\n"
        instruction += self.instruction
        
        # 注入计划信息
        if self.plans:
            completed = [p.content for p in self.plans if p.status == PlanStatus.DONE]
            current = [p.content for p in self.plans if p.status == PlanStatus.DOING]
            future = [f"{p.id}: {p.content}" for p in self.plans if p.status == PlanStatus.TODO]
            
            plan_info = "\n\n### Work Plan Progress\n"
            plan_info += f"- Completed: {', '.join(completed) if completed else 'None'}\n"
            plan_info += f"- CURRENT STEP: {current[0] if current else 'None'}\n"
            plan_info += f"- Future Steps: {', '.join(future) if future else 'None'}\n"
            
            plan_info += "\n### Planning Rules\n"
            plan_info += "1. FOCUS ON TASK: Your 'Thought' MUST focus strictly on the task content and execution. \n"
            plan_info += "2. NO META-PLANNING THOUGHTS: DO NOT output any meta-comments or thoughts about the plan state in your 'Thought' block. Specifically, AVOID phrases like 'I have finished the task', 'Need to complete current task', or 'Moving to next step'. \n"
            plan_info += "3. SILENT TRANSITION: When a step is finished, simply call 'plan_completion_tool' as an action SILENTLY. Your 'Thought' should only contain reasoning about the task itself, not the tool calling process.\n"
            instruction += plan_info

        # 注入状态特定的指导
        if self.workflow_state == AgentWorkflowState.START:
            instruction += "\n\nGUIDANCE: You are at the beginning. You MUST use 'plan_generation_tool' to create a 2-5 steps high-quality, executable plan first. Focus on concrete actions that lead to the solution.\n"
            instruction += "NOTE: All other available tools listed below are for your REFERENCE ONLY to help you design a better plan. You are PROHIBITED from calling any tools other than 'plan_generation_tool' in this stage. \n"
            instruction += "However, if the task is extremely straightforward and requires no planning, you may skip creating a plan and directly provide the FinalAnswer."
        elif self.workflow_state == AgentWorkflowState.EXECUTING:
            instruction += "\n\nGUIDANCE: Focus on the CURRENT STEP. You can use any available tools. Proactively use 'plan_batch_update_tool' to refine or optimize the remaining steps if you discover a better way. Once the current step is fully completed, you MUST use 'plan_completion_tool' first to move to the next step. \n"
            instruction += "CRITICAL: Perform the 'plan_completion_tool' call SILENTLY. Your 'Thought' must NOT contain any words about completing the task or calling the tool. Just output the task-related thoughts and then the tool call."
        elif self.workflow_state == AgentWorkflowState.FINISHED:
            instruction += "\n\nGUIDANCE: All plans are completed. Based on the previous observations, provide your FinalAnswer to the user."

        # 注入记忆压缩提示
        if self.compressed_summary:
            instruction += f"\n\n### Memory Note\nPrevious detailed reasoning has been compressed: {self.compressed_summary}\n"

         # 工具调用规则
        if self.workflow_state!=AgentWorkflowState.FINISHED:
            instruction += "\n\n### Tool Usage Rules\n"
            instruction += "- Use a json blob to specify a tool call with 'action' and 'action_input' keys.\n"
            if self.workflow_state==AgentWorkflowState.EXECUTING:
                instruction += "- You may call multiple tools in one turn by repeating the 'Action: $JSON_BLOB' pattern; this allows invoking the same tool with different parameters or several distinct tools in a single response to improve efficiency.\n"
                instruction += f"- Valid 'action' values: {', '.join([t.name for t in tools])}\n"

        system_prompt = (
            prompt_entity.first_prompt.replace("{{instruction}}", instruction)
            .replace(
                "{{tools}}",
                json.dumps(
                    [
                        tool.model_dump(mode="json")
                        for tool in tools
                    ]
                ),
            )
            .replace(
                "{{tool_names}}",
                ", ".join([tool.name for tool in tools]),
            )
        )

        return SystemPromptMessage(content=system_prompt)

    def _invoke(self, parameters: dict[str, Any]) -> Generator[AgentInvokeMessage]:
        """
        ReAct 运行入口。
        解析参数、组装提示词、循环调用模型并解析动作；
        当检测到工具调用时执行工具并写回观察结果，直到终止条件满足。
        """
        try:
            react_params = ReActParams(**parameters)
        except pydantic.ValidationError as e:
            raise ValueError(f"Invalid parameters: {e!s}") from e

        # 初始化运行时参数，保存本轮指令与 scratchpad
        self.instruction = react_params.instruction
        self.plans = []
        self.compressed_summary = ""
        self.workflow_state = AgentWorkflowState.START
        
        # agent_scratchpad 用于记录 ReAct 的推理链（Thought-Action-Observation 序列）
        agent_scratchpad: list[AgentScratchpadUnit] = []
        iteration_step = 1
        max_iteration_steps = react_params.maximum_iterations
        
        # run_agent_state 控制是否继续执行循环。
        # 初始为 True，每次循环开始设为 False，只有当需要执行工具调用时才重新设为 True。
        run_agent_state = True
        
        llm_usage: dict[str, Optional[LLMUsage]] = {"usage": None}
        final_answer = ""
        prompt_messages: list[PromptMessage] = []

        # 初始化模型配置与 stop 条件
        model = react_params.model
        stop = (
            react_params.model.completion_params.get("stop", [])
            if react_params.model.completion_params
            else []
        )
        # 多数 ReAct 提示依赖 Observation 作为分隔符，模型生成 Observation: 后应立即停止，
        # 等待 Agent 执行工具并将结果填入，否则模型会自己编造 Observation。
        if (
            "Observation" not in stop
            and model.provider not in ignore_observation_providers
        ):
            stop.append("Observation")

        # 初始化历史消息（用于保持多轮上下文）
        self.history_prompt_messages = model.history_prompt_messages

        # 将工具列表转成运行时索引，便于通过名称快速定位
        tools = react_params.tools or []
        if react_params.enable_planner:
            builtin_tools = self._get_builtin_tools()
            tools = list(tools) + builtin_tools
        
        tool_instances = {tool.identity.name: tool for tool in tools} if tools else {}
        react_params.model.completion_params = (
            react_params.model.completion_params or {}
        )
        # 转换为模型可理解的工具描述格式，用于系统提示词
        prompt_messages_tools = self._init_prompt_tools(tools)
        self._prompt_messages_tools = prompt_messages_tools

        while run_agent_state and iteration_step <= max_iteration_steps:
            # 默认只跑一轮；一旦检测到工具调用会设置为 True 进入下一轮
            # 如果模型直接生成 Final Answer，则 run_agent_state 保持 False，循环结束。
            run_agent_state = False
            round_started_at = time.perf_counter()
            # 创建 Round 日志，用于在 UI 上展示当前的推理轮次
            round_log = self.create_log_message(
                label=f"ROUND {iteration_step}",
                data={},
                metadata={
                    LogMetadata.STARTED_AT: round_started_at,
                },
                status=ToolInvokeMessage.LogMessage.LogStatus.START,
            )
            yield round_log
            message_file_ids: list[str] = []

            # 1. 组织构建本轮对话的 Prompt Messages
            prompt_messages = self._organize_prompt_messages(
                agent_scratchpad=agent_scratchpad,
            )
            
            # 根据模型上下文窗口限制，重新计算本次请求允许生成的最大 Token 数。
            # 避免因 Prompt 过长导致 Max Tokens 超出模型总上限而报错。
            if model.entity and model.completion_params:
                self.recalc_llm_max_tokens(
                    model.entity, prompt_messages, model.completion_params
                )
            
            # 调用 LLM 模型
            # stream=True: 启用流式输出，以便实时获取 Thought 和 Action
            # stop=stop: 设置停止词（如 Observation:），让模型在需要外部输入时停下来
            chunks = self.session.model.llm.invoke(
                model_config=LLMModelConfig(**model.model_dump(mode="json")),
                prompt_messages=prompt_messages,
                stream=True,
                stop=stop,
            )

            usage_dict: dict[str, Optional[LLMUsage]] = {"usage": None}
            # 使用 CoT 解析器处理流式输出块。
            # 解析器会识别文本是属于 Thought（思考过程）、Action（工具调用）还是 Answer（最终答案）。
            react_chunks = CotAgentOutputParser.handle_react_stream_output(
                chunks, usage_dict
            )
            
            # 初始化本轮的 scratchpad 单元，用于暂存模型输出的各个部分
            scratchpad = AgentScratchpadUnit(
                agent_response="", # 模型的原始完整响应
                thought="",        # 解析出的思考过程
                action_str="",     # 解析出的动作字符串（JSON）
                observation="",    # 工具执行后的观察结果（稍后填入）
                action=None,       # 解析后的 Action 对象
            )
            # 用于存储本轮的所有 Action
            actions: list[AgentScratchpadUnit.Action] = []

            model_started_at = time.perf_counter()
            # 创建 Model Thought 日志，展示模型的思考过程
            model_log = self.create_log_message(
                label=f"{model.model} Thought",
                data={},
                metadata={
                    LogMetadata.STARTED_AT: model_started_at,
                    LogMetadata.PROVIDER: model.provider,
                },
                parent=round_log,
                status=ToolInvokeMessage.LogMessage.LogStatus.START,
            )
            yield model_log

            # 迭代处理解析后的流式块
            for react_chunk in react_chunks:
                if isinstance(react_chunk, AgentScratchpadUnit.Action):
                    # 如果是 Action 对象，说明模型决定调用工具
                    actions.append(react_chunk)
                    assert scratchpad.agent_response is not None
                    # 将 Action JSON 拼接到完整响应中
                    scratchpad.agent_response += json.dumps(react_chunk.model_dump())
                    
                    # 兼容性处理：保留最后一个 action 到 scratchpad.action，
                    # 但我们会使用 actions 列表来执行所有工具。
                    scratchpad.action = react_chunk
                else:
                    # 如果是文本块（Thought 或 Answer）
                    assert isinstance(react_chunk, ReactChunk)
                    chunk_state = react_chunk.state
                    chunk = react_chunk.content
                    
                    # 将内容实时流式传输给客户端
                    yield self.create_text_message(chunk)
                    
                    if chunk_state == ReactState.ANSWER:
                        # 如果解析器判定为 Final Answer 的一部分
                        final_answer += chunk
                    elif chunk_state == ReactState.THINKING:
                        # 如果是 Thought 部分
                        scratchpad.agent_response = scratchpad.agent_response or ""
                        scratchpad.thought = scratchpad.thought or ""
                        scratchpad.agent_response += chunk
                        scratchpad.thought += chunk
            
            # 汇总所有 action 的字符串表示，用 "\nAction: " 分隔
            # 这样在 _organize_prompt_messages 中拼接 "Action: " 前缀时，
            # 能够正确生成多个 "Action: ..." 行。
            if actions:
                scratchpad.action_str = "\nAction: ".join([json.dumps(a.model_dump()) for a in actions])
            
            # 如果没有生成 Thought，给一个默认值（通常不应发生）
            scratchpad.thought = (
                scratchpad.thought.strip()
                if scratchpad.thought
                else "I am thinking about how to help you"
            )
            # 将本轮 scratchpad 加入历史列表，供下一轮 Prompt 使用
            agent_scratchpad.append(scratchpad)

            # 统计并累加本轮 Token 用量
            if "usage" in usage_dict:
                if usage_dict["usage"] is not None:
                    self.increase_usage(llm_usage, usage_dict["usage"])
            else:
                usage_dict["usage"] = LLMUsage.empty_usage()

            # 准备 Action 日志数据
            action_dict = (
                {"actions": [a.to_dict() for a in actions]}
                if actions
                else {"action": scratchpad.agent_response}
            )

            # 结束 Model Thought 日志记录
            yield self.finish_log_message(
                log=model_log,
                data={"thought": scratchpad.thought, **action_dict},
                metadata={
                    LogMetadata.STARTED_AT: model_started_at,
                    LogMetadata.FINISHED_AT: time.perf_counter(),
                    LogMetadata.ELAPSED_TIME: time.perf_counter() - model_started_at,
                    LogMetadata.PROVIDER: model.provider,
                    LogMetadata.TOTAL_PRICE: usage_dict["usage"].total_price
                    if usage_dict["usage"]
                    else 0,
                    LogMetadata.CURRENCY: usage_dict["usage"].currency
                    if usage_dict["usage"]
                    else "",
                    LogMetadata.TOTAL_TOKENS: usage_dict["usage"].total_tokens
                    if usage_dict["usage"]
                    else 0,
                },
            )
            if not actions:
                # 没有 Action，说明模型直接给出了最终答案 (Final Answer)
                # 这种情况下，scratchpad.thought 通常就是答案内容
                final_answer = scratchpad.thought
            else:
                # 如果有 Action，检查是否是 Final Answer 动作
                is_final_answer_action = any(a.action_name.lower() == "final answer" for a in actions)
                if is_final_answer_action:
                    # 如果包含 Final Answer 动作，取第一个 Final Answer 动作的结果
                    final_action = next(a for a in actions if a.action_name.lower() == "final answer")
                    try:
                        if isinstance(final_action.action_input, dict):
                            final_answer = json.dumps(final_action.action_input)
                        elif isinstance(final_action.action_input, str):
                            final_answer = final_action.action_input
                        else:
                            final_answer = f"{final_action.action_input}"
                    except json.JSONDecodeError:
                        final_answer = f"{final_action.action_input}"
                else:
                    # 触发普通工具调用
                    # 首先检查是否到达最大迭代次数
                    if iteration_step == max_iteration_steps:
                        # 达到上限：不再执行工具，返回错误说明
                        error_messages = []
                        for action in actions:
                            tool_name = action.action_name
                            tool_call_started_at = time.perf_counter()

                            # 为被跳过的工具调用创建日志
                            tool_call_log = self.create_log_message(
                                label=f"CALL {tool_name}",
                                data={},
                                metadata={
                                    LogMetadata.STARTED_AT: tool_call_started_at,
                                    LogMetadata.PROVIDER: tool_instances[
                                        tool_name
                                    ].identity.provider
                                    if tool_instances.get(tool_name)
                                    else "",
                                },
                                parent=round_log,
                                status=ToolInvokeMessage.LogMessage.LogStatus.START,
                            )
                            yield tool_call_log

                            # 观测值设置为错误信息，供下一轮（或最终）返回
                            error_message = (
                                f"Maximum iteration limit ({max_iteration_steps}) reached. "
                                f"Cannot call tool '{tool_name}'."
                            )
                            error_messages.append(error_message)

                            # 结束工具调用日志（记录错误）
                            yield self.finish_log_message(
                                log=tool_call_log,
                                data={
                                    "tool_name": tool_name,
                                    "tool_call_args": action.action_input,
                                    "output": error_message,
                                },
                                metadata={
                                    LogMetadata.STARTED_AT: tool_call_started_at,
                                    LogMetadata.PROVIDER: tool_instances[
                                        tool_name
                                    ].identity.provider
                                    if tool_instances.get(tool_name)
                                    else "",
                                    LogMetadata.FINISHED_AT: time.perf_counter(),
                                    LogMetadata.ELAPSED_TIME: time.perf_counter()
                                    - tool_call_started_at,
                                },
                            )
                        
                        scratchpad.observation = "\n".join(error_messages)
                        scratchpad.agent_response = scratchpad.observation
                        final_answer = scratchpad.observation
                    else:
                        # 未达到最大迭代次数，继续执行
                        # 设置 run_agent_state = True，确保 loop 继续进行下一轮
                        run_agent_state = True
                        
                        observations = []
                        # 1. 预先生成并 yield 所有工具的 START 日志
                        action_logs = []
                        for action in actions:
                            tool_name = action.action_name
                            tool_call_started_at = time.perf_counter()
                            
                            tool_call_log = self.create_log_message(
                                label=f"CALL {tool_name}",
                                data={},
                                metadata={
                                    LogMetadata.STARTED_AT: tool_call_started_at,
                                    LogMetadata.PROVIDER: tool_instances[
                                        tool_name
                                    ].identity.provider
                                    if tool_instances.get(tool_name)
                                    else "",
                                },
                                parent=round_log,
                                status=ToolInvokeMessage.LogMessage.LogStatus.START,
                            )
                            yield tool_call_log
                            action_logs.append({
                                "log": tool_call_log,
                                "started_at": tool_call_started_at,
                                "tool_name": tool_name,
                                "action": action
                            })

                        # 2. 定义并发执行的包装函数
                        # 内部工具需要加锁，外部工具并行执行
                        internal_lock = threading.Lock()
                        
                        def safe_invoke_action(action_item):
                            tool_name = action_item.action_name
                            # 定义需要串行执行的内部工具
                            internal_tools = {
                                "plan_generation_tool", 
                                "plan_batch_update_tool", 
                                "plan_completion_tool", 
                                "memory_compression_tool"
                            }
                            
                            start_time = time.perf_counter()
                            try:
                                if tool_name in internal_tools:
                                    with internal_lock:
                                        res = self._handle_invoke_action(
                                            action=action_item,
                                            tool_instances=tool_instances,
                                            message_file_ids=message_file_ids,
                                            agent_scratchpad=agent_scratchpad,
                                        )
                                else:
                                    res = self._handle_invoke_action(
                                        action=action_item,
                                        tool_instances=tool_instances,
                                        message_file_ids=message_file_ids,
                                        agent_scratchpad=agent_scratchpad,
                                    )
                            except Exception as e:
                                # 捕获所有异常，避免线程崩溃导致整个流程卡死
                                error_msg = f"Error executing {tool_name}: {str(e)}"
                                res = (error_msg, {}, [])
                            
                            end_time = time.perf_counter()
                            return res, end_time

                        # 3. 使用线程池并发执行
                        results = []
                        with concurrent.futures.ThreadPoolExecutor() as executor:
                            # 提交所有任务
                            futures = [executor.submit(safe_invoke_action, action) for action in actions]
                            # 等待所有任务完成，并按提交顺序收集结果
                            for future in futures:
                                results.append(future.result())

                        # 4. 处理结果并 yield FINISH 日志
                        # 注意 results 的顺序与 actions 和 action_logs 是一致的
                        for i, (execution_result, actual_end_time) in enumerate(results):
                            tool_invoke_response, tool_invoke_parameters, additional_messages = execution_result
                            
                            log_info = action_logs[i]
                            tool_call_log = log_info["log"]
                            started_at = log_info["started_at"]
                            tool_name = log_info["tool_name"]
                            
                            observations.append(f"Observation for {tool_name}: {tool_invoke_response}")

                            # 额外消息（如生成的图片）需要直接向上层透传，以便前端展示
                            yield from additional_messages
                            
                            # 结束工具调用日志
                            yield self.finish_log_message(
                                log=tool_call_log,
                                data={
                                    "tool_name": tool_name,
                                    "tool_call_args": tool_invoke_parameters,
                                    "output": tool_invoke_response,
                                },
                                metadata={
                                    LogMetadata.STARTED_AT: started_at,
                                    LogMetadata.PROVIDER: tool_instances[
                                        tool_name
                                    ].identity.provider
                                    if tool_instances.get(tool_name)
                                    else "",
                                    LogMetadata.FINISHED_AT: actual_end_time,
                                    LogMetadata.ELAPSED_TIME: actual_end_time - started_at,
                                },
                            )
                        
                        scratchpad.observation = "\n".join(observations)
                        scratchpad.agent_response = scratchpad.observation

                # 更新工具在提示词中的参数描述，保持工具状态同步
                # 某些工具可能有动态参数或状态变化，需要更新 prompt_messages_tools
                for prompt_tool in self._prompt_messages_tools:
                    self.update_prompt_message_tool(
                        tool_instances[prompt_tool.name], prompt_tool
                    )
            
            # 结束本轮（Round）日志
            yield self.finish_log_message(
                log=round_log,
                data={
                    "actions": [a.to_dict() for a in actions],
                    "thought": scratchpad.thought,
                    "observation": scratchpad.observation,
                },
                metadata={
                    LogMetadata.STARTED_AT: round_started_at,
                    LogMetadata.FINISHED_AT: time.perf_counter(),
                    LogMetadata.ELAPSED_TIME: time.perf_counter() - round_started_at,
                    LogMetadata.TOTAL_PRICE: usage_dict["usage"].total_price
                    if usage_dict["usage"]
                    else 0,
                    LogMetadata.CURRENCY: usage_dict["usage"].currency
                    if usage_dict["usage"]
                    else "",
                    LogMetadata.TOTAL_TOKENS: usage_dict["usage"].total_tokens
                    if usage_dict["usage"]
                    else 0,
                },
            )
            iteration_step += 1

        # yield self.create_text_message(final_answer)

        # 返回本次执行的用量统计，便于上层计费或展示
        yield self.create_json_message(
            {
                "execution_metadata": {
                    LogMetadata.TOTAL_PRICE: llm_usage["usage"].total_price
                    if llm_usage["usage"] is not None
                    else 0,
                    LogMetadata.CURRENCY: llm_usage["usage"].currency
                    if llm_usage["usage"] is not None
                    else "",
                    LogMetadata.TOTAL_TOKENS: llm_usage["usage"].total_tokens
                    if llm_usage["usage"] is not None
                    else 0,
                }
            }
        )

    def _organize_prompt_messages(
        self, agent_scratchpad: list
    ) -> list[PromptMessage]:
        """
        组织构建完整的 Prompt Messages 列表。
        """
        # 始终获取所有工具，但在 START 阶段会有特殊的提示说明
        if self.workflow_state == AgentWorkflowState.START:
            current_tools = self._prompt_messages_tools
        elif self.workflow_state == AgentWorkflowState.EXECUTING:
            # EXECUTING 阶段隐藏初步计划生成工具
            current_tools = [t for t in self._prompt_messages_tools if t.name != "plan_generation_tool"]
        elif self.workflow_state == AgentWorkflowState.FINISHED:
            # FINISHED 状态下不显示任何工具
            current_tools = []
        else:
            current_tools = self._prompt_messages_tools
        
        # 1. 系统提示词：包含动态状态指导和计划看板
        system_message = self._get_dynamic_system_prompt(current_tools)

        # 2. 处理 Scratchpad (之前的思考与行动记录)
        # 将当前轮之前的 Thought/Action/Observation 串联为一条 Assistant 消息
        # 这样模型就能"看到"自己之前的思考过程和工具执行结果
        if not agent_scratchpad and not self.compressed_summary:
            assistant_messages = []
        else:
            assistant_message = AssistantPromptMessage(content="")
            if self.compressed_summary:
                assert isinstance(assistant_message.content, str)
                assistant_message.content += f"[Memory Compression Summary]: {self.compressed_summary}\n\n"
            
            for unit in agent_scratchpad:
                if unit.is_final():
                    # 如果是最终答案，添加 Final Answer 标记
                    assert isinstance(assistant_message.content, str)
                    assistant_message.content += f"Final Answer: {unit.agent_response}"
                else:
                    # 如果是中间步骤，添加 Thought, Action, Observation
                    assert isinstance(assistant_message.content, str)
                    assistant_message.content += f"Thought: {unit.thought}\n\n"
                    if unit.action_str:
                        assistant_message.content += f"Action: {unit.action_str}\n\n"
                    if unit.observation:
                        assistant_message.content += (
                            f"Observation: {unit.observation}\n\n"
                        )

            assistant_messages = [assistant_message]

        # 3. 准备当前用户问题消息 (使用 instruction 替代 query)
        query_messages = [self._user_prompt_message]

        if assistant_messages:
            # 4. 组装最终消息列表
            # 顺序: System -> History -> User Query -> Assistant Scratchpad -> "continue"
            historic_messages = self.history_prompt_messages
            messages = [
                system_message,
                *historic_messages,
                *query_messages,
                *assistant_messages,
                UserPromptMessage(content="continue"),
            ]
        else:
            # 如果没有 scratchpad（第一轮），顺序: System -> History -> User Query
            historic_messages = self.history_prompt_messages
            messages = [system_message, *historic_messages, *query_messages]

        # 按顺序返回模型输入消息序列
        return messages

    def _handle_invoke_action(
        self,
        action: AgentScratchpadUnit.Action,
        tool_instances: Mapping[str, ToolEntity],
        message_file_ids: list[str],
        agent_scratchpad: list,
    ) -> tuple[str, dict[str, Any] | str, list[ToolInvokeMessage]]:
        """
        处理工具调用动作并返回观察结果。
        
        负责解析工具参数、执行工具调用、拼接返回文本与补充消息。
        
        Args:
            action: 包含工具名称和输入的 Action 对象
            tool_instances: 工具名称到工具实体的映射
            message_file_ids: 消息相关的文件 ID 列表
            agent_scratchpad: 推理草稿本列表
            
        Returns:
            tuple: (工具执行结果字符串, 调用使用的参数, 额外的消息列表)
        """
        # 获取工具名称与参数
        tool_call_name = action.action_name
        tool_call_args = action.action_input
        tool_instance = tool_instances.get(tool_call_name)

        if not tool_instance:
            # 工具不存在时直接返回错误说明
            answer = f"there is not a tool named {tool_call_name}"
            return answer, tool_call_args, []

        # 参数解析：支持 JSON 字符串或直接的字典
        if isinstance(tool_call_args, str):
            try:
                # LLM 返回字符串时尝试解析为 JSON
                tool_call_args = json.loads(tool_call_args)
            except json.JSONDecodeError as e:
                # 若不是合法 JSON，则尝试映射到单参数工具
                # 例如：如果工具只有一个参数，且模型直接返回了值而不是 JSON 对象
                params = [
                    param.name
                    for param in tool_instance.parameters
                    if param.form == ToolParameter.ToolParameterForm.LLM
                ]
                if len(params) > 1:
                    # 如果工具需要多个参数但提供了非 JSON 字符串，则报错
                    raise ValueError("tool call args is not a valid json string") from e
                # 单参数情况，构造字典
                tool_call_args = {params[0]: tool_call_args} if len(params) == 1 else {}
        
        tool_call_args = cast(dict[str, Any], tool_call_args)
        
        # 拦截内置工具调用
        if tool_call_name == "plan_generation_tool":
            if self.workflow_state != AgentWorkflowState.START:
                answer = f"Error: '{tool_call_name}' can only be used during the initial planning stage (START)."
                return answer, tool_call_args, []
            observation = self._handle_plan_generation(**tool_call_args)
            return observation, tool_call_args, []
        elif tool_call_name == "plan_batch_update_tool":
            observation = self._handle_plan_batch_update(**tool_call_args)
            return observation, tool_call_args, []
        elif tool_call_name == "plan_completion_tool":
            observation = self._handle_plan_completion(**tool_call_args)
            return observation, tool_call_args, []
        elif tool_call_name == "memory_compression_tool":
            observation = self._handle_memory_compression(agent_scratchpad, **tool_call_args)
            return observation, tool_call_args, []
        
        # 运行时参数覆盖工具默认参数，优先级以调用输入为准
        tool_invoke_parameters = {**tool_instance.runtime_parameters, **tool_call_args}
        
        try:
            # 调用工具并获取流式响应（可能包含文本、图片、文件等）
            tool_invoke_responses = self.session.tool.invoke(
                provider_type=ToolProviderType(tool_instance.provider_type),
                provider=tool_instance.identity.provider,
                tool_name=tool_instance.identity.name,
                parameters=tool_invoke_parameters,
            )
            
            result = ""
            additional_messages = []
            
            # 处理工具返回的各种类型的消息
            for response in tool_invoke_responses:
                if response.type == ToolInvokeMessage.MessageType.TEXT:
                    # 文本直接拼接为 Observation
                    result += cast(ToolInvokeMessage.TextMessage, response.message).text
                elif response.type == ToolInvokeMessage.MessageType.LINK:
                    # 链接提示用户查看
                    result += (
                        f"result link: {cast(ToolInvokeMessage.TextMessage, response.message).text}."
                        + " please tell user to check it."
                    )
                elif response.type in {
                    ToolInvokeMessage.MessageType.IMAGE_LINK,
                    ToolInvokeMessage.MessageType.IMAGE,
                }:
                    # 图片类消息透传给上层，同时在文本中说明文件路径
                    # 这样前端可以显示图片，而模型能"看到"文件路径
                    additional_messages.append(response)
                    image_link_text = cast(
                        ToolInvokeMessage.TextMessage, response.message
                    ).text
                    result += (
                        f"Image has been successfully generated and saved to: {image_link_text}. "
                        + "The image file is now available for download. "
                        + "Please inform the user that the image has been created successfully."
                    )
                elif response.type == ToolInvokeMessage.MessageType.JSON:
                    # JSON 转为字符串，保证模型可读
                    text = json.dumps(
                        cast(
                            ToolInvokeMessage.JsonMessage, response.message
                        ).json_object,
                        ensure_ascii=False,
                    )
                    result += f"tool response: {text}."
                elif response.type == ToolInvokeMessage.MessageType.BLOB:
                    # 二进制文件交给上层处理，仅给出简要文本说明
                    result += "Generated file with ... "
                    additional_messages.append(response)
                else:
                    # 未知类型做兜底输出
                    result += f"tool response: {response.message!r}."
        except Exception as e:
            # 工具调用失败时返回错误文本，避免中断主流程，让 Agent 知道出错了
            result = f"tool invoke error: {e!s}"
            additional_messages = []

        return result, tool_invoke_parameters, additional_messages

    def _convert_dict_to_action(self, action: dict) -> AgentScratchpadUnit.Action:
        """
        将字典格式的 action 转换为标准的 Action 对象。
        """
        return AgentScratchpadUnit.Action(
            action_name=action["action"], action_input=action["action_input"]
        )

    def _format_assistant_message(
        self, agent_scratchpad: list[AgentScratchpadUnit]
    ) -> str:
        """
        将 scratchpad 中的 Thought/Action/Observation/Final Answer 串联为文本。
        主要用于调试或日志记录，展示完整的推理链。
        """
        message = ""
        for scratchpad in agent_scratchpad:
            if scratchpad.is_final():
                # 最终答案只输出 Final Answer
                message += f"Final Answer: {scratchpad.agent_response}"
            else:
                # 中间步骤依次输出 Thought/Action/Observation
                message += f"Thought: {scratchpad.thought}\n\n"
                if scratchpad.action_str:
                    message += f"Action: {scratchpad.action_str}\n\n"
                if scratchpad.observation:
                    message += f"Observation: {scratchpad.observation}\n\n"

        return message
