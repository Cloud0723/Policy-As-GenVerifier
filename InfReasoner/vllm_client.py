#!/usr/bin/env python3
"""
vLLM Client for text generation.
Connects to the vLLM server started by vllm_serve.sh
"""

from openai import OpenAI
from prompt import FIRST_TURN_SYSTEM_PROMPT, MIDDLE_TURN_SYSTEM_PROMPT, FINAL_TURN_SYSTEM_PROMPT
from datasets import load_dataset

# Server configuration (matches vllm_serve.sh defaults)
HOST = "localhost"  # Change to server IP if running remotely
PORT = 9000         # Default port from vllm_serve.sh
MODEL = "Qwen/Qwen2.5-14B-Instruct"  # Default model from vllm_serve.sh

# Initialize OpenAI client pointing to vLLM server
client = OpenAI(
    base_url=f"http://{HOST}:{PORT}/v1",
    api_key="EMPTY"  # vLLM doesn't require an API key
)

def generate_text(prompt: str, max_tokens: int = 512, temperature: float = 0.7) -> str:
    """
    Generate text using the vLLM server.

    Args:
        prompt: The input prompt for text generation
        max_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature (0.0 to 1.0)

    Returns:
        Generated text string
    """
    response = client.completions.create(
        model=MODEL,
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
    )

    return response.choices[0].text


def load_aime_2025_data(dataset_name: str = "AI-MO/aimo-validation-aime", split: str = "train"):
    """
    Load AIME 2025 dataset from Hugging Face.

    Args:
        dataset_name: Hugging Face dataset name
        split: Dataset split to load

    Returns:
        Dataset object from Hugging Face
    """
    dataset = load_dataset(dataset_name, split=split)
    return dataset


def multi_turn_reasoning(question: str, max_turns: int = 5, max_tokens: int = 4096, temperature: float = 0.7) -> dict:
    """
    Perform multi-turn reasoning on a mathematical question.

    Args:
        question: The mathematical question to solve
        max_turns: Maximum number of reasoning turns
        max_tokens: Maximum tokens per turn
        temperature: Sampling temperature

    Returns:
        Dictionary containing the answer, reasoning history, and turn count
    """
    history = []
    summary = None

    for turn in range(1, max_turns + 1):
        # Select appropriate prompt based on turn number
        if turn == 1:
            prompt = FIRST_TURN_SYSTEM_PROMPT.format(question=question)
        elif turn == max_turns:
            prompt = FINAL_TURN_SYSTEM_PROMPT.format(question=question, summary=summary)
        else:
            prompt = MIDDLE_TURN_SYSTEM_PROMPT.format(turn_number=turn, question=question, summary=summary)

        # Generate response
        response = generate_text(prompt, max_tokens=max_tokens, temperature=temperature)
        history.append({
            'turn': turn,
            'prompt_type': 'first' if turn == 1 else ('final' if turn == max_turns else 'intermediate'),
            'prompt': prompt,
            'response': response
        })

        # Check if answer is provided
        if '<answer>' in response:
            # Extract answer
            answer_start = response.find('<answer>') + len('<answer>')
            answer_end = response.find('</answer>')
            answer = response[answer_start:answer_end].strip() if answer_end != -1 else response[answer_start:].strip()

            return {
                'answer': answer,
                'turns': turn,
                'history': history,
                'completed': True
            }

        # Extract summary for next turn
        if '<summary>' in response:
            summary_start = response.find('<summary>') + len('<summary>')
            summary_end = response.find('</summary>')
            summary = response[summary_start:summary_end].strip() if summary_end != -1 else response[summary_start:].strip()
        else:
            # If no summary provided, use reasoning as summary
            if '<reasoning>' in response:
                reasoning_start = response.find('<reasoning>') + len('<reasoning>')
                reasoning_end = response.find('</reasoning>')
                summary = response[reasoning_start:reasoning_end].strip() if reasoning_end != -1 else response[reasoning_start:].strip()

    # Max turns reached without answer
    return {
        'answer': None,
        'turns': max_turns,
        'history': history,
        'completed': False
    }


if __name__ == "__main__":
    # Load AIME 2025 dataset from Hugging Face
    print("=" * 80)
    print("Loading AIME 2025 Dataset from Hugging Face")
    print("=" * 80)

    try:
        dataset = load_aime_2025_data()
        print(f"Dataset loaded successfully!")
        print(f"Number of questions: {len(dataset)}")

        # Get the first question
        first_item = dataset[1]
        print(f"\nDataset fields: {list(first_item.keys())}")

        # Extract question (adjust field name based on actual dataset structure)
        if 'problem' in first_item:
            math_question = first_item['problem']
        elif 'question' in first_item:
            math_question = first_item['question']
        else:
            math_question = str(first_item)

        # Extract ground truth answer
        ground_truth = None
        if 'answer' in first_item:
            ground_truth = first_item['answer']
        elif 'solution' in first_item:
            ground_truth = first_item['solution']
        elif 'ground_truth' in first_item:
            ground_truth = first_item['ground_truth']

        print("\n" + "=" * 80)
        print("AIME 2025 Question #1")
        print("=" * 80)
        print(f"Question: {math_question}")
        if ground_truth:
            print(f"\nGround Truth Answer: {ground_truth}")
        print("-" * 80)

        # Perform multi-turn reasoning
        result = multi_turn_reasoning(math_question, max_turns=5, max_tokens=2048, temperature=0.7)

        print(f"\nCompleted: {result['completed']}")
        print(f"Total turns: {result['turns']}")

        print("\n" + "=" * 80)
        print("Results Comparison:")
        print("=" * 80)
        if ground_truth:
            print(f"Ground Truth: {ground_truth}")
        if result['answer']:
            print(f"Model Answer: {result['answer']}")
        else:
            print("Model Answer: No answer reached within maximum turns.")

        print("\n" + "=" * 80)
        print("Reasoning Trajectory (Full History):")
        print("=" * 80)
        for entry in result['history']:
            print(f"\n{'=' * 80}")
            print(f"Turn {entry['turn']} ({entry['prompt_type']})")
            print("=" * 80)
            print("\n[PROMPT]")
            print("-" * 80)
            print(entry['prompt'])
            print("\n[GENERATION]")
            print("-" * 80)
            print(entry['response'])
            print("=" * 80)

    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("\nFalling back to example question...")

        # Fallback to example question
        math_question = "What is the sum of all prime numbers between 1 and 100?"
        print(f"Question: {math_question}")
        print("-" * 80)

        result = multi_turn_reasoning(math_question, max_turns=5, max_tokens=4096, temperature=1)

        print(f"\nCompleted: {result['completed']}")
        print(f"Total turns: {result['turns']}")
        if result['answer']:
            print(f"\nFinal Answer:\n{result['answer']}")
        else:
            print("\nNo answer reached within maximum turns.")
