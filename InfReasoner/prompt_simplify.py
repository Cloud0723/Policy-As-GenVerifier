# prompts.py

FIRST_TURN_SYSTEM_PROMPT = """You are a mathematics expert. Please reason step by step, and put your final answer within \\boxed{{}}.
Question: {question}
"""


SECOND_TURN_SUMMARY_PROMPT = """Please list what you have achieved in your last response. Note that you should only output the summarization. You should list all the key steps and important intermediate conclusion. Please list them.

Your last response:
{first_turn_response}

Please provide a summary of the key steps and important intermediate conclusions from your last response.
"""


SECOND_TURN_REASONING_PROMPT = """You are a mathematics expert. Please reason step by step, and put your final answer within \\boxed{{}}.

Question: {question}

Summary from the first turn:
{summary}
"""


FINAL_TURN_SYSTEM_PROMPT = """Please Try again based on the summary from the last turn.
Question: {question}
Summary from the last turn: {summary}
"""


SINGLE_TURN_SYSTEM_PROMPT = """You are a mathematics expert. Please reason step by step, and put your final answer within \\boxed{{}}.
Question: {question}
"""
