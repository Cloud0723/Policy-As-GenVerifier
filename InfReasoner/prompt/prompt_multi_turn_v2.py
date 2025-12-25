FIRST_TURN_REASONING_PROMPT = """Question: {question}
Please reason step by step. You have sufficient context to reason and do not need to give the final answer if you do not finish reasoning.
"""


SUMMARY_ANSWER_PROMPT = """Context: {reasoning_context}
If the reasoning is not sufficient, please provide a concise summary based on the above reasoning and do not include \\boxed{{}} in the summary. If the reasoning is sufficient, you can directly give the final answer in \\boxed{{}} without any other context.
"""

MIDDLE_TURN_REASONING_PROMPT = """Question: {question}
Summary of previous reasoning: {summary_reasoning}
Please continue reasoning step by step. You have sufficient context to reason and do not need to give the final answer if you do not finish reasoning.
"""


FINAL_ANSWER_PROMPT = """Context: {reasoning_context}
Please directly provide the final answer based on the above reasoning in \\boxed{{}} without any other context.
"""
