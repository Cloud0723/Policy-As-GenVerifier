FIRST_TURN_REASONING_PROMPT = """Question: {question}
Please reason step by step. You have sufficient context to reason and do not need to give the final answer if you do not finish reasoning. If the reasoning is not sufficient, please provide a concise summary based on the above reasoning and do not include \\boxed{{}} in the summary. 
"""


MIDDLE_TURN_REASONING_PROMPT = """Question: {question}
Summary of previous reasoning: {summary_reasoning}
Please continue reasoning step by step. You have sufficient context to reason and do not need to give the final answer if you do not finish reasoning. If the reasoning is sufficient, you can directly give the final answer in \\boxed{{}} without any other context.
"""

