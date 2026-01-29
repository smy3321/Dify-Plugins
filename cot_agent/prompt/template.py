ENGLISH_REACT_COMPLETION_PROMPT_TEMPLATES = """{{instruction}}

You have access to the following tools:
{{tools}}

Your output should follow this format:
Thought: briefly plan the next step. DO NOT summarize previous information or observations unless necessary for the immediate next step.
Action:
{
  "action": "$TOOL_NAME",
  "action_input": $INPUT
}
Action:
{
  "action": "$TOOL_NAME",
  "action_input": $INPUT
}
(Optional if you have multiple actions)
... (repeat Thought/Action N times)
Thought: brief reason about the final answer
FinalAnswer: final response to human

Begin!
{{historic_messages}}
Question: {{query}}
{{agent_scratchpad}}
Thought:"""  # noqa: E501


ENGLISH_REACT_COMPLETION_AGENT_SCRATCHPAD_TEMPLATES = """Observation: {{observation}}
Thought:"""

ENGLISH_REACT_CHAT_PROMPT_TEMPLATES = """{{instruction}}

You have access to the following tools:
{{tools}}

Your outpur should follow this format:
Thought: briefly plan the next step. DO NOT summarize previous information or observations unless necessary for the immediate next step.
Action:
{
  "action": "$TOOL_NAME",
  "action_input": $INPUT
}

Action:
{
  "action": "$TOOL_NAME",
  "action_input": $INPUT
}
(Optional if you have multiple actions)
... (repeat Thought/Action N times)
Thought: brief reason about the final answer
FinalAnswer: final response to human

Begin!
"""  # noqa: E501


ENGLISH_REACT_CHAT_AGENT_SCRATCHPAD_TEMPLATES = ""

REACT_PROMPT_TEMPLATES = {
    "english": {
        "chat": {
            "prompt": ENGLISH_REACT_CHAT_PROMPT_TEMPLATES,
            "agent_scratchpad": ENGLISH_REACT_CHAT_AGENT_SCRATCHPAD_TEMPLATES,
        },
        "completion": {
            "prompt": ENGLISH_REACT_COMPLETION_PROMPT_TEMPLATES,
            "agent_scratchpad": ENGLISH_REACT_COMPLETION_AGENT_SCRATCHPAD_TEMPLATES,
        },
    }
}
