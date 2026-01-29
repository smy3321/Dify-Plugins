ENGLISH_REACT_COMPLETION_PROMPT_TEMPLATES = """{{instruction}}

You have access to the following tools:
{{tools}}

Your outpur should follow this format:
Thought: consider previous and subsequent steps
Action: $JSON_BLOB
Action: $JSON_BLOB (Optional if you have multiple actions) 
... (repeat Thought/Action N times)
Thought: I know what to respond
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
Thought: consider previous and subsequent steps
Action: $JSON_BLOB
Action: $JSON_BLOB (Optional if you have multiple actions) 
... (repeat Thought/Action N times)
Thought: I know what to respond
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
