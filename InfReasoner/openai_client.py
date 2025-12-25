#!/usr/bin/env python3
"""
OpenAI Client for Mathematical Reasoning.

Connects to OpenAI's API and performs mathematical reasoning using:
- Single-turn: Direct answer generation with parallel sampling
- Multi-turn: Iterative refinement with multiple reasoning turns

Both modes support parallel sampling (n samples per question).
"""

import argparse
import json
import os
from datetime import datetime
from typing import Dict, List, Union

from openai import OpenAI
from datasets import load_dataset
from tqdm import tqdm

from prompt import (
    SINGLE_TURN_SYSTEM_PROMPT,
    FIRST_TURN_SYSTEM_PROMPT,
    MIDDLE_TURN_SYSTEM_PROMPT,
    FINAL_TURN_SYSTEM_PROMPT
)


class OpenAIReasoningClient:
    """
    Unified client for single-turn and multi-turn mathematical reasoning using OpenAI.

    This class provides methods for:
    - Connecting to OpenAI's API
    - Generating text completions
    - Single-turn reasoning with parallel sampling
    - Multi-turn reasoning with iterative refinement
    - Evaluating performance on datasets
    """

    def __init__(self, api_key: str = None, model: str = "gpt-4o"):
        """
        Initialize the OpenAI reasoning client.

        Args:
            api_key: OpenAI API key (defaults to OPENAI_API_KEY environment variable)
            model: Model name (e.g., "gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo")
        """
        self.model = model

        # Initialize OpenAI client
        self.client = OpenAI(
            api_key=api_key or os.environ.get("OPENAI_API_KEY")
        )

    # ============================================================================
    # Text Generation Methods
    # ============================================================================

    def generate_text(self, prompt: str, max_tokens: int = 512, temperature: float = 0.6,
                     top_p: float = 0.95, n: int = 1) -> Union[str, List[str]]:
        """
        Generate text using OpenAI's API.

        Args:
            prompt: The input prompt for text generation
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (0.0 to 2.0)
            top_p: Top-p (nucleus) sampling parameter
            n: Number of completions to generate

        Returns:
            Generated text string if n=1, otherwise list of strings
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            n=n
        )

        if n == 1:
            return response.choices[0].message.content
        else:
            return [choice.message.content for choice in response.choices]

    # ============================================================================
    # Reasoning Methods
    # ============================================================================

    def single_turn_reasoning(self, question: str, max_tokens: int = 8192, temperature: float = 0.6,
                            top_p: float = 0.95, n: int = 1) -> List[Dict]:
        """
        Perform single-turn reasoning on a mathematical question.

        Uses a single prompt with parallel sampling to generate multiple answers.
        Expects answers in LaTeX \\boxed{} format.

        Args:
            question: The mathematical question to solve
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p (nucleus) sampling parameter
            n: Number of samples to generate

        Returns:
            List of dictionaries, each containing the answer, response, and token counts
        """
        # Create prompt using single turn template
        prompt = SINGLE_TURN_SYSTEM_PROMPT.format(question=question)

        # Generate n responses in parallel
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            n=n
        )

        # Process each response
        results = []
        for choice in response.choices:
            response_text = choice.message.content

            # Get token counts from response
            prompt_tokens = response.usage.prompt_tokens if response.usage else 0
            response_tokens = len(response_text.split())  # Rough estimate per response
            total_tokens = response.usage.total_tokens if response.usage else 0

            # Extract answer from \boxed{}
            answer = self._extract_boxed_answer(response_text)

            results.append({
                'answer': answer,
                'prompt': prompt,
                'response': response_text,
                'prompt_tokens': prompt_tokens,
                'response_tokens': response_tokens,
                'total_tokens': total_tokens,
                'completed': answer is not None
            })

        return results

    def multi_turn_reasoning(self, question: str, max_turns: int = 5, max_tokens: int = 2048,
                           temperature: float = 0.6, top_p: float = 0.95, n: int = 1) -> List[Dict]:
        """
        Perform multi-turn reasoning on a mathematical question.

        Uses iterative refinement with multiple turns of thinking.
        Expects responses with XML tags: <think>, <summary>, <answer>.
        Supports parallel sampling - generates n independent reasoning chains.

        Args:
            question: The mathematical question to solve
            max_turns: Maximum number of reasoning turns
            max_tokens: Maximum tokens per turn
            temperature: Sampling temperature
            top_p: Top-p (nucleus) sampling parameter
            n: Number of independent reasoning chains to generate

        Returns:
            List of dictionaries, each containing the answer, response, token counts, and completion status
        """
        all_results = []

        # Generate n independent reasoning chains
        for sample_idx in range(n):
            history = []
            summary = None
            total_tokens = 0

            for turn in range(1, max_turns + 1):
                # Select appropriate prompt based on turn number
                prompt = self._get_turn_prompt(turn, max_turns, question, summary)

                # Generate response
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    n=1
                )

                response_text = response.choices[0].message.content

                # Get token counts
                prompt_tokens = response.usage.prompt_tokens if response.usage else 0
                response_tokens = response.usage.completion_tokens if response.usage else 0
                turn_total_tokens = response.usage.total_tokens if response.usage else 0
                total_tokens += turn_total_tokens

                # Store turn history
                history.append({
                    'turn': turn,
                    'prompt_type': 'first' if turn == 1 else ('final' if turn == max_turns else 'intermediate'),
                    'prompt': prompt,
                    'response': response_text,
                    'prompt_tokens': prompt_tokens,
                    'response_tokens': response_tokens,
                    'total_tokens': turn_total_tokens
                })

                # Check if answer is provided
                answer = self._extract_xml_answer(response_text)
                if answer is not None:
                    all_results.append(self._create_result_dict(answer, question, history, total_tokens, turn, completed=True))
                    break

                # Extract summary for next turn
                summary = self._extract_summary(response_text)
            else:
                # Max turns reached without answer (for-else executes if loop completes without break)
                all_results.append(self._create_result_dict(None, question, history, total_tokens, max_turns, completed=False))

        return all_results

    # ============================================================================
    # Evaluation Methods
    # ============================================================================

    def evaluate(self, dataset, mode: str = "multi-turn", num_questions: int = 30,
                temperature: float = 0.6, top_p: float = 0.95,
                # Sampling parameters
                num_samples: int = 64,
                # Single-turn parameters
                max_tokens_single: int = 8192,
                # Multi-turn parameters
                max_turns: int = 5, max_tokens_multi: int = 2048,
                # Output parameters
                save_trajectories: bool = True, output_dir: str = "trajectories") -> Dict:
        """
        Unified evaluation method for both single-turn and multi-turn reasoning.

        Args:
            dataset: Dataset to evaluate on
            mode: Reasoning mode ("single-turn" or "multi-turn")
            num_questions: Number of questions to process from end of dataset
            temperature: Sampling temperature
            top_p: Top-p parameter
            num_samples: Number of samples per question (used in both modes)
            max_tokens_single: Maximum tokens to generate (single-turn)
            max_turns: Maximum number of reasoning turns (multi-turn)
            max_tokens_multi: Maximum tokens per turn (multi-turn)
            save_trajectories: Whether to save trajectories to JSON file
            output_dir: Directory to save trajectory files

        Returns:
            Dictionary containing all results and statistics
        """
        # Print header
        self._print_header(mode, num_questions, dataset, max_turns, max_tokens_multi,
                          num_samples, max_tokens_single, temperature)

        # Get questions to process
        questions_to_process = self._get_questions_to_process(dataset, num_questions)

        # Initialize tracking variables
        all_question_results = []
        total_correct = 0
        total_completed = 0  # Questions with all samples completed
        total_samples = 0     # Total samples across all questions

        # Process questions with progress bar
        pbar = tqdm(questions_to_process, desc="Processing questions")
        for question_idx in pbar:
            # Extract question and ground truth
            math_question, ground_truth = self._extract_question_and_truth(dataset[question_idx])

            # Perform reasoning based on mode (both return: question_result, correct_count, is_completed)
            if mode == "single-turn":
                question_result, correct_count, is_completed = self._evaluate_single_turn_question(
                    math_question, ground_truth, question_idx, num_samples,
                    max_tokens_single, temperature, top_p
                )
            else:  # multi-turn
                question_result, correct_count, is_completed = self._evaluate_multi_turn_question(
                    math_question, ground_truth, question_idx, num_samples,
                    max_turns, max_tokens_multi, temperature, top_p
                )

            # Update totals (unified for both modes)
            total_correct += correct_count
            total_samples += num_samples
            if is_completed:
                total_completed += 1

            # Calculate progress bar metrics (unified for both modes)
            current_acc = (total_correct / total_samples * 100) if total_samples > 0 else 0
            q_acc = question_result['accuracy']
            questions_processed_so_far = len(all_question_results) + 1
            completion_rate = (total_completed / questions_processed_so_far * 100) if questions_processed_so_far > 0 else 0

            # Update progress bar description with metrics
            pbar.update(0)
            pbar.set_description(f"Acc: {current_acc:.2f}% | Q{len(all_question_results)+1}: {q_acc:.2f}% | Complete: {completion_rate:.2f}%")

            all_question_results.append(question_result)

            # Save trajectory for this question immediately
            if save_trajectories:
                self._save_question_trajectory(question_result, mode, output_dir)

        # Calculate final statistics and print results
        results = self._finalize_results(all_question_results, mode, questions_to_process,
                                        num_questions, total_correct, total_completed, total_samples)

        # Save final summary if requested
        if save_trajectories:
            summary_path = self._save_final_summary(results, output_dir)
            print(f"\nTrajectories saved to: {output_dir}/")
            print(f"Summary saved to: {summary_path}")

        return results

    # ============================================================================
    # Helper Methods - Answer Extraction
    # ============================================================================

    def _extract_boxed_answer(self, response: str) -> Union[str, None]:
        """Extract answer from LaTeX \\boxed{} format."""
        if '\\boxed{' not in response:
            return None

        answer_start = response.find('\\boxed{') + len('\\boxed{')
        # Find matching closing brace
        brace_count = 1
        answer_end = answer_start
        while answer_end < len(response) and brace_count > 0:
            if response[answer_end] == '{':
                brace_count += 1
            elif response[answer_end] == '}':
                brace_count -= 1
            answer_end += 1

        if brace_count == 0:
            return response[answer_start:answer_end-1].strip()
        return None

    def _extract_xml_answer(self, response: str) -> Union[str, None]:
        """Extract answer from XML <answer> tags."""
        if '<answer>' not in response:
            return None

        answer_start = response.find('<answer>') + len('<answer>')
        answer_end = response.find('</answer>')
        if answer_end != -1:
            return response[answer_start:answer_end].strip()
        return response[answer_start:].strip()

    def _extract_summary(self, response: str) -> Union[str, None]:
        """Extract summary from XML <summary> tags or fall back to <think> tags."""
        if '<summary>' in response:
            summary_start = response.find('<summary>') + len('<summary>')
            summary_end = response.find('</summary>')
            return response[summary_start:summary_end].strip() if summary_end != -1 else response[summary_start:].strip()
        elif '<think>' in response:
            think_start = response.find('<think>') + len('<think>')
            think_end = response.find('</think>')
            return response[think_start:think_end].strip() if think_end != -1 else response[think_start:].strip()
        return None

    # ============================================================================
    # Helper Methods - Prompts and Data
    # ============================================================================

    def _get_turn_prompt(self, turn: int, max_turns: int, question: str, summary: str) -> str:
        """Get the appropriate prompt for the current turn."""
        if turn == 1:
            return FIRST_TURN_SYSTEM_PROMPT.format(turn_number=turn, question=question)
        elif turn == max_turns:
            return FINAL_TURN_SYSTEM_PROMPT.format(turn_number=turn, question=question, summary=summary)
        else:
            return MIDDLE_TURN_SYSTEM_PROMPT.format(turn_number=turn, question=question, summary=summary)

    def _create_result_dict(self, answer: Union[str, None], question: str, history: List[Dict],
                           total_tokens: int, turns: int, completed: bool) -> Dict:
        """Create a standardized result dictionary."""
        return {
            'answer': answer,
            'prompt': FIRST_TURN_SYSTEM_PROMPT.format(turn_number=1, question=question),
            'response': history[-1]['response'] if history else '',
            'prompt_tokens': sum(h['prompt_tokens'] for h in history),
            'response_tokens': sum(h['response_tokens'] for h in history),
            'total_tokens': total_tokens,
            'completed': completed,
            'turns': turns,
            'history': history
        }

    def _get_questions_to_process(self, dataset, num_questions: int) -> List[int]:
        """Get the indices of questions to process (last N questions)."""
        total_questions = len(dataset)
        start_idx = max(0, total_questions - num_questions)
        return list(range(start_idx, total_questions))

    def _extract_question_and_truth(self, item: Dict) -> tuple:
        """Extract question text and ground truth from dataset item."""
        # Extract question
        if 'problem' in item:
            math_question = item['problem']
        elif 'question' in item:
            math_question = item['question']
        else:
            math_question = str(item)

        # Extract ground truth answer
        ground_truth = None
        if 'answer' in item:
            ground_truth = item['answer']
        elif 'solution' in item:
            ground_truth = item['solution']
        elif 'ground_truth' in item:
            ground_truth = item['ground_truth']

        return math_question, ground_truth

    # ============================================================================
    # Helper Methods - Evaluation
    # ============================================================================

    def _evaluate_single_turn_question(self, question: str, ground_truth: str, question_idx: int,
                                      num_samples: int, max_tokens: int, temperature: float,
                                      top_p: float) -> tuple:
        """Evaluate a single question using single-turn reasoning.

        Returns:
            tuple: (question_result, correct_count, is_completed)
                - question_result: dict with evaluation details
                - correct_count: number of correct samples
                - is_completed: always True for single-turn (all samples complete)
        """
        results = self.single_turn_reasoning(
            question, max_tokens=max_tokens, temperature=temperature,
            top_p=top_p, n=num_samples
        )

        # Process results
        correct_count = 0
        total_tokens = 0
        all_answers = []

        for sample_idx, result in enumerate(results):
            is_correct = self._check_answer_correctness(result['answer'], ground_truth)
            if is_correct:
                correct_count += 1

            total_tokens += result['total_tokens']
            all_answers.append({
                'sample_idx': sample_idx,
                'answer': result['answer'],
                'is_correct': is_correct,
                'prompt': result['prompt'],
                'response': result['response'],
                'prompt_tokens': result['prompt_tokens'],
                'response_tokens': result['response_tokens'],
                'tokens': result['total_tokens']
            })

        # Calculate accuracy for this question
        accuracy = (correct_count / num_samples * 100) if num_samples > 0 else 0
        avg_tokens = total_tokens / num_samples if num_samples > 0 else 0

        question_result = {
            'question_idx': question_idx,
            'question': question,
            'ground_truth': ground_truth,
            'correct_count': correct_count,
            'total_samples': num_samples,
            'accuracy': accuracy,
            'total_tokens': total_tokens,
            'avg_tokens': avg_tokens,
            'all_answers': all_answers
        }

        # Return same format as multi-turn: (question_result, correct_count, is_completed)
        return question_result, correct_count, True  # Single-turn always completes

    def _evaluate_multi_turn_question(self, question: str, ground_truth: str, question_idx: int,
                                     num_samples: int, max_turns: int, max_tokens: int,
                                     temperature: float, top_p: float) -> tuple:
        """Evaluate a single question using multi-turn reasoning.

        Returns:
            tuple: (question_result, correct_count, is_completed)
                - question_result: dict with evaluation details
                - correct_count: number of correct samples
                - is_completed: whether all samples completed within max_turns
        """
        results = self.multi_turn_reasoning(
            question, max_turns=max_turns, max_tokens=max_tokens,
            temperature=temperature, top_p=top_p, n=num_samples
        )

        # Process results
        correct_count = 0
        total_tokens = 0
        completed_count = 0
        all_answers = []

        for sample_idx, result in enumerate(results):
            is_correct = self._check_answer_correctness(result['answer'], ground_truth)
            if is_correct:
                correct_count += 1

            if result['completed']:
                completed_count += 1

            total_tokens += result['total_tokens']
            all_answers.append({
                'sample_idx': sample_idx,
                'answer': result['answer'],
                'is_correct': is_correct,
                'completed': result['completed'],
                'turns': result['turns'],
                'prompt_tokens': result['prompt_tokens'],
                'response_tokens': result['response_tokens'],
                'tokens': result['total_tokens'],
                'history': result['history']  # Contains all turn-by-turn prompts and responses
            })

        # Calculate accuracy for this question
        accuracy = (correct_count / num_samples * 100) if num_samples > 0 else 0
        completion_rate = (completed_count / num_samples * 100) if num_samples > 0 else 0
        avg_tokens = total_tokens / num_samples if num_samples > 0 else 0
        avg_turns = sum(a['turns'] for a in all_answers) / num_samples if num_samples > 0 else 0

        question_result = {
            'question_idx': question_idx,
            'question': question,
            'ground_truth': ground_truth,
            'correct_count': correct_count,
            'total_samples': num_samples,
            'accuracy': accuracy,
            'completed_count': completed_count,
            'completion_rate': completion_rate,
            'total_tokens': total_tokens,
            'avg_tokens': avg_tokens,
            'avg_turns': avg_turns,
            'all_answers': all_answers
        }

        # All samples completed if all finished within max_turns
        all_completed = (completed_count == num_samples)

        # Return same format as single-turn: (question_result, correct_count, is_completed)
        return question_result, correct_count, all_completed

    def _check_answer_correctness(self, answer: Union[str, None], ground_truth: Union[str, None]) -> bool:
        """Check if the model answer matches the ground truth using mathematical equivalence."""
        if not answer or not ground_truth:
            return False

        # Normalize both answers
        def normalize_answer(ans):
            if ans is None:
                return None
            ans = str(ans).strip()
            # Remove common LaTeX formatting
            ans = ans.replace('\\boxed{', '').replace('}', '')
            ans = ans.replace('\\text{', '').replace('\\mathrm{', '').replace('\\mathbf{', '')
            ans = ans.replace('$', '').replace('\\', '')
            # Remove spaces and convert to lowercase for text comparison
            ans = ans.replace(' ', '').lower()
            # Remove trailing/leading punctuation
            ans = ans.strip('.,;:')
            return ans

        normalized_answer = normalize_answer(answer)
        normalized_truth = normalize_answer(ground_truth)

        if normalized_answer == normalized_truth:
            return True

        # Try to parse as numbers and compare numerically
        try:
            # Handle fractions like "1/2"
            def eval_fraction(s):
                if '/' in s:
                    parts = s.split('/')
                    if len(parts) == 2:
                        return float(parts[0]) / float(parts[1])
                return float(s)

            ans_val = eval_fraction(normalized_answer)
            truth_val = eval_fraction(normalized_truth)

            # Try to convert to int if both are whole numbers
            if ans_val == int(ans_val) and truth_val == int(truth_val):
                return int(ans_val) == int(truth_val)

            # Compare with small tolerance for floating point
            return abs(ans_val - truth_val) < 1e-6
        except (ValueError, ZeroDivisionError, TypeError):
            # If can't parse as numbers, fall back to string comparison
            return normalized_answer == normalized_truth

    def _finalize_results(self, all_question_results: List[Dict], mode: str,
                         questions_to_process: List[int], num_questions: int,
                         total_correct: int, total_completed: int, total_samples: int) -> Dict:
        """Calculate final statistics, print results, and return summary."""
        questions_processed = len(questions_to_process)
        accuracy = (total_correct / total_samples * 100) if total_samples > 0 else 0

        # Build common result dictionary
        results = {
            'mode': mode,
            'questions_processed': questions_processed,
            'total_samples': total_samples,
            'total_correct': total_correct,
            'accuracy': accuracy,
            'all_question_results': all_question_results
        }

        # Add mode-specific statistics
        if mode == "multi-turn":
            overall_completion_rate = (total_completed / questions_processed * 100) if questions_processed > 0 else 0
            total_tokens_all = sum(qr['total_tokens'] for qr in all_question_results)
            avg_tokens_per_question = total_tokens_all / questions_processed if questions_processed > 0 else 0
            avg_turns = sum(qr['avg_turns'] for qr in all_question_results) / questions_processed if questions_processed > 0 else 0

            results.update({
                'total_completed': total_completed,
                'completion_rate': overall_completion_rate,
                'avg_turns': avg_turns,
                'total_tokens': total_tokens_all,
                'avg_tokens_per_question': avg_tokens_per_question
            })

            self._print_results(all_question_results, mode, questions_processed, num_questions,
                              total_correct=total_correct, total_samples=total_samples,
                              total_completed=total_completed, completion_rate=overall_completion_rate,
                              avg_turns=avg_turns, total_tokens_all=total_tokens_all,
                              avg_tokens_per_question=avg_tokens_per_question)
        else:
            self._print_results(all_question_results, mode, questions_processed, num_questions,
                              total_correct=total_correct, total_samples=total_samples)

        return results

    def _save_question_trajectory(self, question_result: Dict, mode: str, output_dir: str = "trajectories") -> str:
        """
        Save a single question's trajectory to a JSON file.

        Args:
            question_result: Result dictionary for one question
            mode: Reasoning mode ("single-turn" or "multi-turn")
            output_dir: Directory to save trajectory files

        Returns:
            Path to the saved JSON file
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Generate filename with question index
        question_idx = question_result['question_idx']
        filename = f"traj_{mode}_q{question_idx}.json"
        filepath = os.path.join(output_dir, filename)

        # Save to JSON file
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(question_result, f, indent=2, ensure_ascii=False)

        return filepath

    def _save_final_summary(self, results: Dict, output_dir: str = "trajectories") -> str:
        """
        Save final summary statistics to a JSON file.

        Args:
            results: Final results dictionary from evaluate()
            output_dir: Directory to save trajectory files

        Returns:
            Path to the saved JSON file
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        mode = results['mode']
        filename = f"summary_{mode}_{timestamp}.json"
        filepath = os.path.join(output_dir, filename)

        # Create summary without individual question results
        summary = {k: v for k, v in results.items() if k != 'all_question_results'}
        summary['num_questions_saved'] = len(results['all_question_results'])

        # Save to JSON file
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        return filepath

    # ============================================================================
    # Helper Methods - Printing
    # ============================================================================

    def _print_header(self, mode: str, num_questions: int, dataset, max_turns: int = None,
                     max_tokens_multi: int = None, num_samples: int = None,
                     max_tokens_single: int = None, temperature: float = None):
        """Print evaluation header with configuration."""
        print("=" * 80)
        print(f"{mode.upper().replace('-', '-')} REASONING MODE")
        print("=" * 80)

        total_questions = len(dataset)
        start_idx = max(0, total_questions - num_questions)
        print(f"\nProcessing last {num_questions} questions (from {start_idx+1} to {total_questions})...")

        if mode == "multi-turn":
            print(f"Configuration: num_samples={num_samples}, max_turns={max_turns}, max_tokens={max_tokens_multi}, temp={temperature}")
        else:
            print(f"Configuration: num_samples={num_samples}, max_tokens={max_tokens_single}, temp={temperature}")
        print("=" * 80)

    def _print_results(self, all_question_results: List[Dict], mode: str,
                      questions_processed: int, num_questions: int, **kwargs):
        """Unified method to print results for both single-turn and multi-turn evaluation."""
        print("\n" + "=" * 80)
        print("OVERALL RESULTS")
        print("=" * 80)
        print(f"Questions Processed: {questions_processed} (last {num_questions} from dataset)")

        # Common metrics for both modes
        total_samples = kwargs['total_samples']
        total_correct = kwargs['total_correct']
        accuracy = (total_correct / total_samples * 100) if total_samples > 0 else 0

        print(f"Total Samples: {total_samples}")
        print(f"Total Correct: {total_correct}/{total_samples}")
        print(f"Overall Accuracy: {accuracy:.2f}%")

        # Mode-specific metrics
        if mode == "multi-turn":
            total_completed = kwargs['total_completed']
            completion_rate = kwargs['completion_rate']
            avg_turns = kwargs['avg_turns']
            total_tokens_all = kwargs['total_tokens_all']
            avg_tokens_per_question = kwargs['avg_tokens_per_question']

            print(f"Questions with all samples completed: {total_completed}/{questions_processed} ({completion_rate:.2f}%)")
            print(f"Average Turns per Sample: {avg_turns:.2f}")
            print(f"Total Tokens: {total_tokens_all}")
            print(f"Avg Tokens per Question: {avg_tokens_per_question:.2f}")

        print("=" * 80)


# ============================================================================
# Utility Functions
# ============================================================================
 
def load_aime24_data(dataset_name: str = "math-ai/aime24", split: str = "test"):
    """
    Load AIME24 dataset from Hugging Face.
 
    Args:
        dataset_name: Hugging Face dataset name
        split: Dataset split to load
 
    Returns:
        Dataset object from Hugging Face
    """
    dataset = load_dataset(dataset_name, split=split)
    return dataset


# ============================================================================
# Main Script
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OpenAI Reasoning Client")
    parser.add_argument("--mode", type=str, default="single-turn", choices=["single-turn", "multi-turn"],
                       help="Reasoning mode: single-turn or multi-turn")
    parser.add_argument("--api-key", type=str, default=None, help="OpenAI API key (defaults to OPENAI_API_KEY env var)")
    parser.add_argument("--model", type=str, default="gpt-4o", help="OpenAI model name (e.g., gpt-4o, gpt-4o-mini, gpt-3.5-turbo)")
    parser.add_argument("--num-questions", type=int, default=30, help="Number of questions to process")

    # Sampling parameter
    parser.add_argument("--num-samples", type=int, default=8, help="Number of samples per question")
    parser.add_argument("--temperature", type=float, default=0.6, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=0.95, help="Top-p parameter")

    # Single-turn specific arguments
    parser.add_argument("--max-tokens-single", type=int, default=16384, help="Max tokens (single-turn)")

    # Multi-turn specific arguments
    parser.add_argument("--max-turns", type=int, default=5, help="Max reasoning turns (multi-turn)")
    parser.add_argument("--max-tokens-multi", type=int, default=16384, help="Max tokens per turn (multi-turn)")

    # Output arguments
    parser.add_argument("--save-trajectories", action="store_true", default=True, help="Save trajectories to JSON file")
    parser.add_argument("--output-dir", type=str, default="./output/trajectories", help="Directory to save trajectory files")

    args = parser.parse_args()

    # Load AIME24 dataset from Hugging Face
    print("=" * 80)
    print("Loading AIME24 Dataset from Hugging Face")
    print("=" * 80)

    dataset = load_aime24_data()
    print(f"Dataset loaded successfully!")
    print(f"Number of questions: {len(dataset)}")

    # Show dataset fields
    first_item = dataset[0]
    print(f"\nDataset fields: {list(first_item.keys())}")

    # Initialize client
    client = OpenAIReasoningClient(api_key=args.api_key, model=args.model)

    # Create output directory with model name and timestamp as suffix
    model_name_clean = args.model.lower().replace("/", "_").replace("\\", "_")
    # If output_dir ends with '/', remove it first
    base_dir = args.output_dir.rstrip('/').replace("-", "_")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir_with_model = f"{base_dir}_{model_name_clean}_{timestamp}"

    # Run unified evaluation
    results = client.evaluate(
        dataset=dataset,
        mode=args.mode,
        num_questions=args.num_questions,
        temperature=args.temperature,
        top_p=args.top_p,
        num_samples=args.num_samples,
        max_tokens_single=args.max_tokens_single,
        max_turns=args.max_turns,
        max_tokens_multi=args.max_tokens_multi,
        save_trajectories=args.save_trajectories,
        output_dir=output_dir_with_model
    )
