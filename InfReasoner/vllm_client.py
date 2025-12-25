#!/usr/bin/env python3
"""
Unified Client for Mathematical Reasoning.

Supports both vLLM servers and OpenAI API for mathematical reasoning:
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
from transformers import AutoTokenizer
from tqdm import tqdm

from prompt.prompt_multi_turn_v2 import (
    FIRST_TURN_REASONING_PROMPT,
    MIDDLE_TURN_REASONING_PROMPT,
    SUMMARY_ANSWER_PROMPT,
    FINAL_ANSWER_PROMPT
)
from prompt.prompt_single_turn import SINGLE_TURN_SYSTEM_PROMPT


class SingleTurnRollout:
    """
    Handles single-turn reasoning rollout for a single question.

    This class manages:
    - Single prompt generation
    - Answer extraction from \\boxed{} format
    - Token usage tracking
    """

    def __init__(self, question: str):
        """
        Initialize a single-turn rollout session.

        Args:
            question: The mathematical question to solve
        """
        self.question = question
        self.prompt = SINGLE_TURN_SYSTEM_PROMPT.format(question=question)
        self.answer = None
        self.response = None
        self.prompt_tokens = 0
        self.response_tokens = 0
        self.total_tokens = 0

    def get_prompt(self) -> str:
        """
        Get the prompt for single-turn reasoning.

        Returns:
            Formatted prompt string
        """
        return self.prompt

    def process_response(self, response: str, prompt_tokens: int, response_tokens: int):
        """
        Process the response and extract the answer.

        Args:
            response: The model's response text
            prompt_tokens: Number of tokens in the prompt
            response_tokens: Number of tokens in the response
        """
        self.response = response
        self.prompt_tokens = prompt_tokens
        self.response_tokens = response_tokens
        self.total_tokens = prompt_tokens + response_tokens
        self.answer = self._extract_boxed_answer(response)

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

    def get_result(self) -> Dict:
        """
        Get the final result dictionary for this rollout.

        Returns:
            Dictionary containing answer, response, and token counts
        """
        return {
            'answer': self.answer,
            'prompt': self.prompt,
            'response': self.response,
            'prompt_tokens': self.prompt_tokens,
            'response_tokens': self.response_tokens,
            'total_tokens': self.total_tokens,
            'completed': self.answer is not None
        }


class MultiTurnRollout:
    """
    Handles multi-turn reasoning rollout for a single question.

    This class manages the iterative refinement process:
    - Each turn has TWO steps: reasoning generation, then summary/answer generation
    - Maintains conversation history across turns
    - Handles turn-based prompting (first, middle, final)
    - Extracts answers and summaries from responses
    - Tracks token usage
    """

    def __init__(self, question: str, max_turns: int = 5):
        """
        Initialize a multi-turn rollout session.

        Args:
            question: The mathematical question to solve
            max_turns: Maximum number of reasoning turns allowed
        """
        self.question = question
        self.max_turns = max_turns
        self.history = []
        self.summary_reasoning = None  # Summary from previous turn
        self.current_reasoning = None   # Current turn's reasoning
        self.total_tokens = 0
        self.current_turn = 0
        self.answer = None
        self.completed = False

    def get_reasoning_prompt(self) -> str:
        """
        Get the reasoning prompt for the current turn.

        Returns:
            Formatted reasoning prompt string
        """
        turn = self.current_turn + 1

        if turn == 1:
            return FIRST_TURN_REASONING_PROMPT.format(question=self.question)
        else:
            return MIDDLE_TURN_REASONING_PROMPT.format(
                question=self.question,
                summary_reasoning=self.summary_reasoning or ""
            )

    def get_summary_prompt(self, reasoning_context: str) -> str:
        """
        Get the summary/answer prompt based on current reasoning.

        Args:
            reasoning_context: The reasoning generated in this turn

        Returns:
            Formatted summary/answer prompt string
        """
        turn = self.current_turn + 1

        if turn == self.max_turns:
            # Final turn: must provide answer
            return FINAL_ANSWER_PROMPT.format(reasoning_context=reasoning_context)
        else:
            # Non-final turn: can provide summary or answer
            return SUMMARY_ANSWER_PROMPT.format(reasoning_context=reasoning_context)

    def process_reasoning_response(self, reasoning: str, prompt_tokens: int, response_tokens: int, prompt: str):
        """
        Process the reasoning response (first step of a turn).

        Args:
            reasoning: The model's reasoning text
            prompt_tokens: Number of tokens in the prompt
            response_tokens: Number of tokens in the response
            prompt: The prompt that was used to generate this response
        """
        self.current_reasoning = reasoning
        turn_total_tokens = prompt_tokens + response_tokens
        self.total_tokens += turn_total_tokens

        # Store reasoning in history (will be updated with summary later)
        if not hasattr(self, '_current_turn_data'):
            self._current_turn_data = {}

        self._current_turn_data = {
            'turn': self.current_turn + 1,
            'reasoning_prompt': prompt,
            'reasoning': reasoning,
            'reasoning_prompt_tokens': prompt_tokens,
            'reasoning_response_tokens': response_tokens,
            'reasoning_tokens': turn_total_tokens
        }

    def process_summary_response(self, summary_response: str, prompt_tokens: int, response_tokens: int, prompt: str) -> bool:
        """
        Process the summary/answer response (second step of a turn).

        Args:
            summary_response: The model's summary or answer text
            prompt_tokens: Number of tokens in the prompt
            response_tokens: Number of tokens in the response
            prompt: The prompt that was used to generate this response

        Returns:
            True if an answer was found (rollout complete), False otherwise
        """
        self.current_turn += 1
        turn_total_tokens = prompt_tokens + response_tokens
        self.total_tokens += turn_total_tokens

        # Complete the turn data and store in history
        turn_data = self._current_turn_data
        turn_data.update({
            'summary_prompt': prompt,
            'summary_response': summary_response,
            'summary_prompt_tokens': prompt_tokens,
            'summary_response_tokens': response_tokens,
            'summary_tokens': turn_total_tokens,
            'total_tokens': turn_data['reasoning_tokens'] + turn_total_tokens,
            'prompt_type': self._get_prompt_type()
        })
        self.history.append(turn_data)

        # Check if answer is provided
        self.answer = self._extract_xml_answer(summary_response)
        if self.answer is not None:
            self.completed = True
            return True

        # Extract summary for next turn
        self.summary_reasoning = self._extract_summary(summary_response)
        if self.summary_reasoning is None:
            # Fallback: use the entire summary response
            self.summary_reasoning = summary_response

        # Check if max turns reached
        if self.current_turn >= self.max_turns:
            self.completed = False
            return True

        return False

    def _get_prompt_type(self) -> str:
        """Get the prompt type for the current turn."""
        if self.current_turn == 1:
            return 'first'
        elif self.current_turn == self.max_turns:
            return 'final'
        else:
            return 'intermediate'

    def _extract_xml_answer(self, response: str) -> Union[str, None]:
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

    def _extract_summary(self, response: str) -> Union[str, None]:
        """Extract summary from response text (no XML tags expected in v2)."""
        # In v2, the summary is the entire response text (no special tags)
        return response.strip() if response.strip() else None

    def get_result(self) -> Dict:
        """
        Get the final result dictionary for this rollout.

        Returns:
            Dictionary containing answer, response, token counts, and completion status
        """
        # Calculate total tokens from history
        total_prompt_tokens = sum(
            h.get('reasoning_prompt_tokens', 0) + h.get('summary_prompt_tokens', 0)
            for h in self.history
        )
        total_response_tokens = sum(
            h.get('reasoning_response_tokens', 0) + h.get('summary_response_tokens', 0)
            for h in self.history
        )

        return {
            'answer': self.answer,
            'prompt': FIRST_TURN_REASONING_PROMPT.format(question=self.question),
            'response': self.history[-1]['summary_response'] if self.history else '',
            'prompt_tokens': total_prompt_tokens,
            'response_tokens': total_response_tokens,
            'total_tokens': self.total_tokens,
            'completed': self.completed,
            'turns': self.current_turn,
            'history': self.history
        }

    def should_continue(self) -> bool:
        """Check if the rollout should continue."""
        return not self.completed and self.current_turn < self.max_turns


class VLLMReasoningClient:
    """
    Unified client for single-turn and multi-turn mathematical reasoning.

    Supports both vLLM servers and OpenAI API.

    This class provides methods for:
    - Connecting to a vLLM server or OpenAI API
    - Generating text completions
    - Single-turn reasoning with parallel sampling
    - Multi-turn reasoning with iterative refinement
    - Evaluating performance on datasets
    """

    def __init__(self, host: str = "localhost", port: int = 9000, model: str = "Qwen/Qwen2.5-72B-Instruct",
                 use_openai: bool = False):
        """
        Initialize the reasoning client.

        Args:
            host: vLLM server host (ignored if use_openai=True)
            port: vLLM server port (ignored if use_openai=True)
            model: Model name/path (for vLLM) or OpenAI model name (e.g., "gpt-4o")
            use_openai: Whether to use OpenAI API instead of vLLM server (reads OPENAI_API_KEY from environment)
        """
        self.host = host
        self.port = port
        self.model = model
        self.use_openai = use_openai

        # Initialize OpenAI client
        if use_openai:
            # Use OpenAI API - read API key from environment variable
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable must be set when using --use-openai")
            self.client = OpenAI(api_key=api_key)
            self.tokenizer = None  # OpenAI handles tokenization internally
        else:
            # Use vLLM server
            self.client = OpenAI(
                base_url=f"http://{host}:{port}/v1",
                api_key="EMPTY"  # vLLM doesn't require an API key
            )
            # Initialize tokenizer for token counting
            self.tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)

    # ============================================================================
    # Text Generation Methods
    # ============================================================================

    def generate_text(self, prompt: str, max_tokens: int = 512, temperature: float = 0.6,
                     top_p: float = 0.95, n: int = 1) -> Union[str, List[str]]:
        """
        Generate text using vLLM server or OpenAI API.

        Args:
            prompt: The input prompt for text generation
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (0.0 to 2.0 for OpenAI, 0.0 to 1.0 for vLLM)
            top_p: Top-p (nucleus) sampling parameter
            n: Number of completions to generate

        Returns:
            Generated text string if n=1, otherwise list of strings
        """
        if self.use_openai:
            # Use chat completions API for OpenAI
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
        else:
            # Use completions API for vLLM
            response = self.client.completions.create(
                model=self.model,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                n=n,
            )

            if n == 1:
                return response.choices[0].text
            else:
                return [choice.text for choice in response.choices]

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
        # Create rollout for this question
        rollout = SingleTurnRollout(question)
        prompt = rollout.get_prompt()

        # Generate responses using the unified generate_text method
        responses = self.generate_text(prompt, max_tokens=max_tokens, temperature=temperature,
                                      top_p=top_p, n=n)

        # Ensure responses is a list
        if not isinstance(responses, list):
            responses = [responses]

        # Count prompt tokens
        if self.use_openai:
            # For OpenAI, estimate prompt tokens
            prompt_tokens = len(prompt.split())
        else:
            # For vLLM, use tokenizer
            prompt_tokens = len(self.tokenizer.encode(prompt))

        # Process each response with a separate rollout instance
        results = []
        for response_text in responses:
            # Count response tokens
            if self.use_openai:
                # For OpenAI, estimate response tokens
                response_tokens = len(response_text.split())
            else:
                # For vLLM, use tokenizer
                response_tokens = len(self.tokenizer.encode(response_text))

            # Create a rollout for each sample
            sample_rollout = SingleTurnRollout(question)
            sample_rollout.process_response(response_text, prompt_tokens, response_tokens)
            results.append(sample_rollout.get_result())

        return results

    def multi_turn_reasoning(self, question: str, max_turns: int = 5, max_tokens: int = 2048,
                           temperature: float = 0.6, top_p: float = 0.95, n: int = 1) -> List[Dict]:
        """
        Perform multi-turn reasoning on a mathematical question.

        Uses iterative refinement with multiple turns of thinking.
        Each turn has TWO steps:
        1. Generate reasoning (3/4 of max_tokens)
        2. Generate summary/answer based on reasoning (1/4 of max_tokens)

        Supports parallel sampling - generates n independent reasoning chains.

        Args:
            question: The mathematical question to solve
            max_turns: Maximum number of reasoning turns
            max_tokens: Maximum tokens per turn (total for both reasoning and summary)
            temperature: Sampling temperature
            top_p: Top-p (nucleus) sampling parameter
            n: Number of independent reasoning chains to generate

        Returns:
            List of dictionaries, each containing the answer, response, token counts, and completion status
        """
        all_results = []

        # Calculate token allocation: 3/4 for reasoning, 1/4 for summary
        max_tokens_reasoning = int(max_tokens * 3 / 4)
        max_tokens_summary = int(max_tokens * 1 / 4)

        # Generate n independent reasoning chains
        for _ in range(n):
            rollout = MultiTurnRollout(question, max_turns)

            while rollout.should_continue():
                # Step 1: Generate reasoning
                reasoning_prompt = rollout.get_reasoning_prompt()

                # Generate reasoning using the unified generate_text method
                reasoning_response = self.generate_text(
                    reasoning_prompt,
                    max_tokens=max_tokens_reasoning,
                    temperature=temperature,
                    top_p=top_p,
                    n=1
                )

                # Count tokens for reasoning
                if self.use_openai:
                    reasoning_prompt_tokens = len(reasoning_prompt.split())
                    reasoning_response_tokens = len(reasoning_response.split())
                else:
                    reasoning_prompt_tokens = len(self.tokenizer.encode(reasoning_prompt))
                    reasoning_response_tokens = len(self.tokenizer.encode(reasoning_response))

                # Process reasoning response
                rollout.process_reasoning_response(
                    reasoning_response,
                    reasoning_prompt_tokens,
                    reasoning_response_tokens,
                    reasoning_prompt
                )

                # Step 2: Generate summary/answer based on reasoning
                summary_prompt = rollout.get_summary_prompt(reasoning_response)

                # Generate summary/answer
                summary_response = self.generate_text(
                    summary_prompt,
                    max_tokens=max_tokens_summary,
                    temperature=temperature,
                    top_p=top_p,
                    n=1
                )

                # Count tokens for summary
                if self.use_openai:
                    summary_prompt_tokens = len(summary_prompt.split())
                    summary_response_tokens = len(summary_response.split())
                else:
                    summary_prompt_tokens = len(self.tokenizer.encode(summary_prompt))
                    summary_response_tokens = len(self.tokenizer.encode(summary_response))

                # Process summary response and check if rollout is complete
                is_done = rollout.process_summary_response(
                    summary_response,
                    summary_prompt_tokens,
                    summary_response_tokens,
                    summary_prompt
                )

                if is_done:
                    break

            # Collect result from rollout
            all_results.append(rollout.get_result())

        return all_results

    # ============================================================================
    # Evaluation Methods
    # ============================================================================

    def evaluate(self, dataset, mode: str = "multi-turn",
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
            dataset: Dataset to evaluate on (already filtered to desired questions)
            mode: Reasoning mode ("single-turn" or "multi-turn")
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
        self._print_header(mode, dataset, max_turns, max_tokens_multi,
                          num_samples, max_tokens_single, temperature)

        # Initialize tracking variables
        all_question_results = []
        total_correct = 0
        total_completed = 0  # Questions with all samples completed
        total_samples = 0     # Total samples across all questions

        # Process questions with progress bar
        questions_to_process = range(len(dataset))
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
        results = self._finalize_results(all_question_results, mode, len(dataset),
                                        total_correct, total_completed, total_samples)

        # Save final summary if requested
        if save_trajectories:
            summary_path = self._save_final_summary(results, output_dir)
            print(f"\nTrajectories saved to: {output_dir}/")
            print(f"Summary saved to: {summary_path}")

        return results

    # ============================================================================
    # Helper Methods - Prompts and Data
    # ============================================================================

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
                'prompt': result['prompt'],
                'response': result['response'],
                'prompt_tokens': result['prompt_tokens'],
                'response_tokens': result['response_tokens'],
                'tokens': result['total_tokens']
            })

        # Calculate accuracy for this question
        accuracy = (correct_count / num_samples * 100) if num_samples > 0 else 0
        completion_rate = (completed_count / num_samples * 100) if num_samples > 0 else 0
        avg_tokens = total_tokens / num_samples if num_samples > 0 else 0

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
            'all_answers': all_answers
        }

        # All samples always complete in single-turn
        all_completed = (completed_count == num_samples)

        # Return same format as multi-turn: (question_result, correct_count, is_completed)
        return question_result, correct_count, all_completed

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
        total_response_tokens = sum(a['response_tokens'] for a in all_answers)
        avg_response_tokens = total_response_tokens / num_samples if num_samples > 0 else 0

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
            'avg_response_tokens': avg_response_tokens,
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
                         questions_processed: int,
                         total_correct: int, total_completed: int, total_samples: int) -> Dict:
        """Calculate final statistics, print results, and return summary."""
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
            avg_response_tokens_overall = sum(qr['avg_response_tokens'] for qr in all_question_results) / questions_processed if questions_processed > 0 else 0

            results.update({
                'total_completed': total_completed,
                'completion_rate': overall_completion_rate,
                'avg_turns': avg_turns,
                'total_tokens': total_tokens_all,
                'avg_tokens_per_question': avg_tokens_per_question,
                'avg_response_tokens': avg_response_tokens_overall
            })

            self._print_results(mode, questions_processed,
                              total_correct=total_correct, total_samples=total_samples,
                              total_completed=total_completed, completion_rate=overall_completion_rate,
                              avg_turns=avg_turns, total_tokens_all=total_tokens_all,
                              avg_tokens_per_question=avg_tokens_per_question,
                              avg_response_tokens=avg_response_tokens_overall)
        else:
            self._print_results(mode, questions_processed,
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

    def _print_header(self, mode: str, dataset, max_turns: int = None,
                     max_tokens_multi: int = None, num_samples: int = None,
                     max_tokens_single: int = None, temperature: float = None):
        """Print evaluation header with configuration."""
        print("=" * 80)
        print(f"{mode.upper().replace('-', '-')} REASONING MODE")
        print("=" * 80)

        num_questions = len(dataset)
        print(f"\nProcessing {num_questions} questions...")

        if mode == "multi-turn":
            print(f"Configuration: num_samples={num_samples}, max_turns={max_turns}, max_tokens={max_tokens_multi}, temp={temperature}")
        else:
            print(f"Configuration: num_samples={num_samples}, max_tokens={max_tokens_single}, temp={temperature}")
        print("=" * 80)

    def _print_results(self, mode: str, questions_processed: int, **kwargs):
        """Unified method to print results for both single-turn and multi-turn evaluation."""
        print("\n" + "=" * 80)
        print("OVERALL RESULTS")
        print("=" * 80)
        print(f"Questions Processed: {questions_processed}")

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
            avg_response_tokens = kwargs['avg_response_tokens']

            print(f"Questions with all samples completed: {total_completed}/{questions_processed} ({completion_rate:.2f}%)")
            print(f"Average Turns per Sample: {avg_turns:.2f}")
            print(f"Total Tokens: {total_tokens_all}")
            print(f"Avg Tokens per Question: {avg_tokens_per_question:.2f}")
            print(f"Avg Response Tokens per Question: {avg_response_tokens:.2f}")

        print("=" * 80)


# ============================================================================
# Utility Functions
# ============================================================================

def load_aime_2025_data(dataset_name: str = "AI-MO/aimo-validation-aime", split: str = "train", num_questions: int = None):
    """
    Load AIME 2025 dataset from Hugging Face.

    Args:
        dataset_name: Hugging Face dataset name
        split: Dataset split to load
        num_questions: Number of questions to load (loads from the end). If None, loads all.

    Returns:
        Dataset object from Hugging Face
    """
    dataset = load_dataset(dataset_name, split=split)

    # If num_questions is specified, only load that many from the end
    if num_questions is not None and num_questions < len(dataset):
        # Select the last num_questions samples
        start_idx = len(dataset) - num_questions
        dataset = dataset.select(range(start_idx, len(dataset)))

    return dataset


def create_output_directory(base_dir: str, model_name: str, mode: str,
                           max_len: int, num_samples: int, max_turns: int = None) -> str:
    """
    Create output directory path with model name, mode, max length, num samples, and timestamp.

    Args:
        base_dir: Base output directory path
        model_name: Model name/path to include in directory name
        mode: Reasoning mode ("single-turn" or "multi-turn")
        max_len: Maximum tokens (max_tokens_single or max_tokens_multi)
        num_samples: Number of samples per question
        max_turns: Maximum turns (only for multi-turn mode)

    Returns:
        Full output directory path with all parameters and timestamp
    """
    model_name_clean = model_name.lower().replace("/", "_").replace("\\", "_")
    base_dir_clean = base_dir.rstrip('/').replace("-", "_")
    mode_clean = mode.replace("-", "_")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Add max_turns to directory name if multi-turn mode
    if mode == "multi-turn" and max_turns is not None:
        return f"{base_dir_clean}_{model_name_clean}_{mode_clean}_turns{max_turns}_maxlen{max_len}_n{num_samples}_{timestamp}"
    else:
        return f"{base_dir_clean}_{model_name_clean}_{mode_clean}_maxlen{max_len}_n{num_samples}_{timestamp}"


# ============================================================================
# Main Script
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified Reasoning Client (vLLM/OpenAI)")
    parser.add_argument("--mode", type=str, default="multi-turn", choices=["single-turn", "multi-turn"],
                       help="Reasoning mode: single-turn or multi-turn")

    # Client configuration
    parser.add_argument("--use-openai", action="store_true", help="Use OpenAI API instead of vLLM server (requires OPENAI_API_KEY env var)")
    parser.add_argument("--host", type=str, default="localhost", help="vLLM server host (ignored if --use-openai)")
    parser.add_argument("--port", type=int, default=9000, help="vLLM server port (ignored if --use-openai)")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-4B",
                       help="Model name (vLLM model path or OpenAI model like 'gpt-4o')")
    parser.add_argument("--num-questions", type=int, default=30, help="Number of questions to process")

    # Sampling parameter
    parser.add_argument("--num-samples", type=int, default=64, help="Number of samples per question")
    parser.add_argument("--temperature", type=float, default=0.6, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=0.95, help="Top-p parameter")

    # Single-turn specific arguments
    parser.add_argument("--max-tokens-single", type=int, default=32000, help="Max tokens (single-turn)")

    # Multi-turn specific arguments
    parser.add_argument("--max-turns", type=int, default=5, help="Max reasoning turns (multi-turn)")
    parser.add_argument("--max-tokens-multi", type=int, default=16384, help="Max tokens per turn (multi-turn)")

    # Output arguments
    parser.add_argument("--save-trajectories", action="store_true", default=True, help="Save trajectories to JSON file")
    parser.add_argument("--output-dir", type=str, default="./output/trajectories", help="Directory to save trajectory files")

    args = parser.parse_args()

    # Print configuration
    print("=" * 80)
    if args.use_openai:
        print("Using OpenAI API")
        print(f"Model: {args.model}")
    else:
        print("Using vLLM Server")
        print(f"Server: {args.host}:{args.port}")
        print(f"Model: {args.model}")
    print("=" * 80)

    # Load AIME 2025 dataset from Hugging Face
    print("\n" + "=" * 80)
    print("Loading AIME 2025 Dataset from Hugging Face")
    print("=" * 80)

    dataset = load_aime_2025_data(num_questions=args.num_questions)
    print(f"Dataset loaded successfully!")
    print(f"Number of questions: {len(dataset)}")

    # Show dataset fields
    first_item = dataset[0]
    print(f"\nDataset fields: {list(first_item.keys())}")

    # Initialize client
    client = VLLMReasoningClient(
        host=args.host,
        port=args.port,
        model=args.model,
        use_openai=args.use_openai
    )

    # Create output directory with mode-specific max_len
    max_len = args.max_tokens_single if args.mode == "single-turn" else args.max_tokens_multi
    output_dir = create_output_directory(
        args.output_dir,
        args.model,
        args.mode,
        max_len,
        args.num_samples,
        max_turns=args.max_turns if args.mode == "multi-turn" else None
    )

    # Run unified evaluation
    results = client.evaluate(
        dataset=dataset,
        mode=args.mode,
        temperature=args.temperature,
        top_p=args.top_p,
        num_samples=args.num_samples,
        max_tokens_single=args.max_tokens_single,
        max_turns=args.max_turns,
        max_tokens_multi=args.max_tokens_multi,
        save_trajectories=args.save_trajectories,
        output_dir=output_dir
    )
