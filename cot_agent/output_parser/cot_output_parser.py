import json
from collections.abc import Generator
from enum import Enum
from typing import Union

from dify_plugin.entities.model.llm import LLMResultChunk
from dify_plugin.interfaces.agent import AgentScratchpadUnit

# 定义可能出现在关键词前缀之前的分隔符集合，用于判断是否是独立的单词开始
PREFIX_DELIMITERS = frozenset({"\n", " ", ""})


class ReactState(Enum):
    """
    ReAct 状态枚举。
    定义了 Agent 在推理过程中的不同状态，以及对应的关键词前缀。
    """
    THINKING = ("Thought:", "THINKING")      # 思考状态
    ANSWER = ("FinalAnswer:", "ANSWER")      # 最终回答状态

    def __init__(self, prefix: str, state: str):
        self.prefix = prefix
        self.prefix_lower = prefix.lower()
        self.state = state


class ReactChunk:
    """
    ReAct 流式输出块。
    用于封装从 LLM 流式输出中解析出的文本片段及其所属的状态。
    """
    def __init__(self, state: ReactState, content: str):
        self.state = state      # 当前块所属的状态 (Thinking/Answer)
        self.content = content  # 文本内容


class CotAgentOutputParser:
    """
    CoT (Chain of Thought) Agent 输出解析器。
    负责处理 LLM 的流式输出，识别 Thought、Action 和 Final Answer。
    """
    @classmethod
    def handle_react_stream_output(
        cls, llm_response: Generator[LLMResultChunk, None, None], usage_dict: dict
    ) -> Generator[Union[ReactChunk, AgentScratchpadUnit.Action], None, None]:
        """
        处理 ReAct 模式的流式输出。
        
        这是一个生成器函数，它逐个消费 LLM 的输出 chunk，
        维护一个状态机来识别当前是在输出 Thought、Action 还是 Answer，
        并尝试从流中提取完整的 Action JSON。

        Args:
            llm_response: LLM 输出的流式生成器
            usage_dict: 用于回传 token 使用量的字典

        Yields:
            ReactChunk: 包含文本内容和状态的块
            AgentScratchpadUnit.Action: 解析出的完整动作对象
        """
        def parse_action(json_str):
            """
            尝试将提取出的 JSON 字符串解析为 Action 对象。

            示例输入:
                json_str = '{"tool": "calculator", "input": "3+5"}'
            返回:
                AgentScratchpadUnit.Action(
                    action_name="calculator",
                    action_input="3+5"
                )
            """
            try:
                # 尝试解析 JSON，strict=False 允许一些非标准格式（如控制字符）
                action = json.loads(json_str, strict=False)
                action_name = None
                action_input = None

                # 兼容处理：cohere 模型可能返回列表格式
                if isinstance(action, list) and len(action) == 1:
                    action = action[0]

                # 遍历字典寻找 input 和 action name
                # 这种模糊匹配是为了兼容不同模型可能输出略有不同的 key
                for key, value in action.items():
                    if "input" in key.lower():
                        action_input = value
                    else:
                        action_name = value

                if action_name is not None and action_input is not None:
                    return AgentScratchpadUnit.Action(
                        action_name=action_name,
                        action_input=action_input,
                    )
                else:
                    # 如果缺少必要字段，返回原字符串（当作普通文本处理）
                    return json_str or ""
            except Exception:
                # 解析失败，返回原字符串
                return json_str or ""

        # JSON 解析相关的状态变量
        json_cache = ""         # 缓存正在累积的 JSON 字符串
        in_json = False         # 标记是否正在读取 JSON 内容
        got_json = False        # 标记是否已完整读取一个 JSON 对象

        # JSON 语法分析辅助变量
        json_in_string = False      # 是否在 JSON 字符串内部（双引号内）
        json_escape = False         # 是否遇到转义字符
        pending_action_json = False # 是否检测到了 Action 前缀，正在等待 JSON 开始
        json_stack: list[str] = []  # 用于匹配花括号和方括号的栈

        cur_state = ReactState.THINKING # 初始状态默认为思考中
        last_character = ""

        class PrefixMatcher:
            """
            前缀匹配器。
            用于在流式输出中检测是否出现了特定的关键词（如 "Action:", "FinalAnswer:"）。
            """
            __slots__ = ("prefix", "state_on_full_match", "cache", "idx")

            def __init__(self, spec: ReactState | str):
                if isinstance(spec, ReactState):
                    self.prefix = spec.prefix_lower
                    self.state_on_full_match = spec
                else:
                    self.prefix = spec.lower()
                    self.state_on_full_match = None
                self.cache = ""
                self.idx = 0

            def step(self, delta: str) -> tuple[bool, ReactChunk | None, bool, bool]:
                """
                处理流中的下一个字符。
                
                Returns:
                    tuple: (
                        yield_raw_delta: 是否应该直接输出当前字符（未匹配或匹配中断）,
                        emitted_chunk: 如果匹配中断，返回之前缓存的已匹配部分作为普通文本块,
                        delta_consumed: 当前字符是否被匹配器消耗（缓存）,
                        matched_full_prefix: 是否完全匹配了前缀
                    )
                """
                nonlocal last_character, cur_state

                yield_raw_delta = False
                emitted_chunk = None
                delta_consumed = False
                matched_full_prefix = False

                if delta.lower() == self.prefix[self.idx]:
                    # 字符匹配成功
                    # 检查单词边界：如果是第一个字符，前一个字符必须是分隔符
                    if self.idx == 0 and last_character not in PREFIX_DELIMITERS:
                        yield_raw_delta = True
                    else:
                        last_character = delta
                        self.cache += delta
                        self.idx += 1
                        if self.idx == len(self.prefix):
                            # 完全匹配前缀
                            self.cache = ""
                            self.idx = 0
                            if self.state_on_full_match is not None:
                                # 如果关联了状态（如 FinalAnswer），则切换解析器状态
                                cur_state = self.state_on_full_match
                            matched_full_prefix = True
                        delta_consumed = True
                elif self.cache:
                    # 匹配中断，说明之前的缓存不是前缀的一部分，需要吐出来
                    last_character = delta
                    emitted_chunk = ReactChunk(cur_state, self.cache)
                    self.cache = ""
                    self.idx = 0
                    # 注意：当前字符 delta 没有被消耗，需要在外层循环继续处理

                return yield_raw_delta, emitted_chunk, delta_consumed, matched_full_prefix

        # 初始化三个匹配器
        action_matcher = PrefixMatcher("action:")
        answer_matcher = PrefixMatcher(ReactState.ANSWER)
        thought_matcher = PrefixMatcher(ReactState.THINKING)

        for response in llm_response:
            if response.delta.usage:
                usage_dict["usage"] = response.delta.usage
            response_content = response.delta.message.content
            if not isinstance(response_content, str):
                continue

            # stream 处理逻辑
            index = 0
            while index < len(response_content):
                steps = 1
                delta = response_content[index: index + steps]
                yield_delta = False

                if not in_json:
                    # 1. 尝试匹配 Action 前缀
                    yield_raw_delta, emitted_chunk, delta_consumed, matched_action_prefix = action_matcher.step(delta)
                    if emitted_chunk is not None:
                        yield emitted_chunk
                    yield_delta = yield_delta or yield_raw_delta
                    if matched_action_prefix:
                        # 匹配到 "Action:"，准备开始接收 JSON
                        pending_action_json = True
                    if delta_consumed:
                        index += steps
                        continue

                    # 2. 尝试匹配 Answer 前缀
                    yield_raw_delta, emitted_chunk, delta_consumed, _ = answer_matcher.step(delta)
                    if emitted_chunk is not None:
                        yield emitted_chunk
                    yield_delta = yield_delta or yield_raw_delta
                    if delta_consumed:
                        index += steps
                        continue

                    # 3. 尝试匹配 Thought 前缀
                    yield_raw_delta, emitted_chunk, delta_consumed, _ = thought_matcher.step(delta)
                    if emitted_chunk is not None:
                        yield emitted_chunk
                    yield_delta = yield_delta or yield_raw_delta
                    if delta_consumed:
                        index += steps
                        continue

                    # 如果没有被任何匹配器消耗，说明是普通内容，直接输出
                    if yield_delta:
                        index += steps
                        last_character = delta
                        yield ReactChunk(cur_state, delta)
                        continue

                # JSON 提取逻辑：如果正在等待 JSON 开始（已匹配到 Action:）
                if not in_json and pending_action_json:
                    if delta in {"{", "["}:
                        # 遇到 JSON 开始符号
                        in_json = True
                        got_json = False
                        json_cache = delta
                        json_in_string = False
                        json_escape = False
                        json_stack = ["}" if delta == "{" else "]"]
                        last_character = delta
                        index += steps
                        continue
                    if not delta.isspace():
                        # 如果遇到非空白字符且不是 JSON 开始符，说明不是合法的 Action JSON
                        # 取消等待状态
                        pending_action_json = False

                # JSON 内容累积逻辑
                if in_json:
                    last_character = delta
                    json_cache += delta

                    if json_in_string:
                        # 在字符串内部，处理转义
                        if json_escape:
                            json_escape = False
                        elif delta == "\\":
                            json_escape = True
                        elif delta == '"':
                            json_in_string = False
                    else:
                        # 在字符串外部，处理结构
                        if delta == '"':
                            json_in_string = True
                        elif delta in {"{", "["}:
                            json_stack.append("}" if delta == "{" else "]")
                        elif delta in {"}", "]"} and json_stack and delta == json_stack[-1]:
                            json_stack.pop()
                            if not json_stack:
                                # 栈空了，说明完整的 JSON 对象结束
                                in_json = False
                                got_json = True
                                pending_action_json = False
                                index += steps
                                continue

                # JSON 解析与输出
                if got_json:
                    got_json = False
                    last_character = delta
                    parsed_result = parse_action(json_cache)
                    if isinstance(parsed_result, AgentScratchpadUnit.Action):
                        yield parsed_result
                    else:
                        # 解析失败，当作普通文本输出
                        yield ReactChunk(cur_state, json_cache)
                    json_cache = ""
                    in_json = False
                    json_in_string = False
                    json_escape = False
                    json_stack = []

                if not in_json:
                    # 如果不是 JSON 内容，则作为当前状态的普通文本块输出
                    last_character = delta
                    yield ReactChunk(cur_state, delta)

                index += steps

        # 循环结束后，如果还有缓存的 JSON 字符串（可能不完整），尝试解析或输出
        if json_cache:
            parsed_result = parse_action(json_cache)
            if isinstance(parsed_result, AgentScratchpadUnit.Action):
                yield parsed_result
            else:
                yield ReactChunk(cur_state, json_cache)
