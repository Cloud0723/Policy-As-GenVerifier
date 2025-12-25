#!/usr/bin/env python3
"""
AIME24 Single Turn Evaluation
Reference: CURE/eval.sh evaluation pipeline

Single turn means: one prompt -> one response (no multi-turn self-correction)
"""

import argparse
import json
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from openai import OpenAI
import re
from datetime import datetime

# Import math_verify for answer verification
from math_verify.metric import math_metric
from math_verify.parser import LatexExtractionConfig, ExprExtractionConfig
from math_verify.errors import TimeoutException


def verify_answer(solution_str: str, ground_truth: str) -> dict:
    """Verify answer using math_verify (similar to CURE's compute_acc)"""
    verify_func = math_metric(
        gold_extraction_target=(LatexExtractionConfig(boxed_match_priority=0),),
        pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig(boxed_match_priority=0)),
    )
    ret_score = 0.
    pred = ""
    
    ground_truth_boxed = "\\boxed{" + str(ground_truth) + "}"
    solution_str = solution_str[-300:]  # Last 300 chars for efficiency
    
    try:
        ret_score, answers = verify_func([ground_truth_boxed], [solution_str])
        if answers and len(answers) > 1 and len(answers[1]) > 0:
            pred = answers[1][0]
    except (TimeoutException, Exception):
        pass
    
    return {
        "score": ret_score,
        "acc": ret_score,
        "pred": pred,
    }


def extract_boxed_answer(solution_str: str) -> str:
    """Extract answer from \\boxed{} format"""
    boxed_match = re.search(r'\\boxed\{([^}]+)\}', solution_str)
    if boxed_match:
        return boxed_match.group(1)
    return solution_str


def generate_single_turn(client, model, prompt, max_tokens, temperature, top_p, top_k):
    """Generate single turn response (no multi-turn)"""
    try:
        # Note: OpenAI API doesn't support top_k parameter
        # vLLM's OpenAI-compatible API also doesn't expose top_k through this interface
        kwargs = {
            'model': model,
            'prompt': prompt,
            'max_tokens': max_tokens,
            'temperature': temperature,
            'top_p': top_p,
        }
        # Only add top_k if it's not -1 (disabled), but note it may not work
        # Most OpenAI-compatible APIs don't support top_k
        # if top_k > 0:
        #     kwargs['top_k'] = top_k
        
        response = client.completions.create(**kwargs)
        return response.choices[0].text
    except Exception as e:
        print(f"Generation error: {e}")
        return ""


def build_single_turn_prompt(question: str) -> str:
    """Build single turn prompt (no multi-turn system prompt)"""
    # Simple direct prompt for single turn
    prompt = f"""Please solve the following math problem step by step and put your final answer within \\boxed{{}}.

Problem: {question}

Solution:"""
    return prompt


def evaluate_aime24_single_turn(
    host: str,
    port: int,
    model: str,
    n_samples: int,
    max_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    output_file: str
):
    """
    Evaluate AIME24 dataset with single turn generation
    
    Args:
        host: vLLM server host
        port: vLLM server port
        model: Model name
        n_samples: Number of samples per problem
        max_tokens: Max response tokens
        temperature: Sampling temperature
        top_p: Top-p sampling
        top_k: Top-k sampling
        output_file: Output JSON file path
    """
    # Initialize OpenAI client for vLLM
    client = OpenAI(
        base_url=f"http://{host}:{port}/v1",
        api_key="EMPTY"
    )
    
    # Load AIME24 dataset
    print("Loading AIME24 dataset...")
    dataset = load_dataset("math-ai/aime24", split="test")
    total_problems = len(dataset)
    print(f"✓ Loaded {total_problems} problems\n")
    
    # Storage for results
    all_results = []
    problem_scores = []
    problem_best_scores = []
    
    print(f"Starting evaluation ({n_samples} samples per problem)...")
    print("=" * 80)
    
    # Evaluate each problem
    for prob_idx, item in enumerate(tqdm(dataset, desc="Progress")):
        question = item['problem']
        ground_truth = extract_boxed_answer(item['solution'])
        
        print(f"\n[{prob_idx+1}/{total_problems}] Problem ID: {item['id']}")
        print(f"Question: {question[:100]}...")
        print(f"Ground Truth: {ground_truth}")
        
        # Generate n_samples for this problem
        sample_results = []
        sample_accs = []
        
        for sample_idx in range(n_samples):
            # Build single turn prompt
            prompt = build_single_turn_prompt(question)
            
            # Generate response (single turn)
            response = generate_single_turn(
                client=client,
                model=model,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k
            )
            
            # Verify answer
            verify_result = verify_answer(response, ground_truth)
            acc = verify_result['acc']
            sample_accs.append(acc)
            
            sample_results.append({
                'sample_idx': sample_idx,
                'response': response,
                'predicted_answer': verify_result['pred'],
                'accuracy': acc
            })
            
            # Progress update
            if (sample_idx + 1) % 8 == 0 or sample_idx == n_samples - 1:
                current_avg = np.mean(sample_accs)
                current_best = np.max(sample_accs)
                print(f"  [{sample_idx+1}/{n_samples}] Avg: {current_avg:.2%}, Best: {current_best:.2%}")
        
        # Compute statistics for this problem
        avg_score = np.mean(sample_accs)
        best_score = np.max(sample_accs)
        std_score = np.std(sample_accs)
        
        problem_scores.append(avg_score)
        problem_best_scores.append(best_score)
        
        problem_result = {
            'problem_idx': prob_idx,
            'problem_id': item['id'],
            'question': question,
            'ground_truth': ground_truth,
            'avg_score': avg_score,
            'best_score': best_score,
            'std_score': std_score,
            'n_samples': n_samples,
            'samples': sample_results
        }
        all_results.append(problem_result)
        
        print(f"  Result: Avg={avg_score:.2%}, Best={best_score:.2%}, Std={std_score:.4f}")
        print("-" * 80)
    
    # Compute global statistics
    final_avg_score = np.mean(problem_scores)
    final_best_score = np.mean(problem_best_scores)
    final_std = np.std(problem_scores)
    
    # Compute Pass@1 (at least one correct)
    pass_at_1 = np.mean([1 if s > 0 else 0 for s in problem_scores])
    
    # Compute Best-of-N for different N
    best_of_n_scores = {}
    problem_sample_accs = [[r['accuracy'] for r in prob['samples']] for prob in all_results]
    for n in [1, 2, 4, 8, 16, 32]:
        if n <= n_samples:
            best_scores = [np.max(accs[:n]) for accs in problem_sample_accs]
            best_of_n_scores[f"best@{n}"] = np.mean(best_scores)
    
    # Print final results
    print("\n" + "=" * 80)
    print("📊 Final Evaluation Results")
    print("=" * 80)
    print(f"\nMetrics:")
    print(f"  Avg@{n_samples}: {final_avg_score:.4f} ({final_avg_score*100:.2f}%)")
    print(f"  Best@{n_samples}: {final_best_score:.4f} ({final_best_score*100:.2f}%)")
    print(f"  Std@{n_samples}: {final_std:.4f}")
    print(f"  Pass@1: {pass_at_1:.4f} ({pass_at_1*100:.2f}%)")
    
    print(f"\nBest-of-N:")
    for metric_name, metric_value in best_of_n_scores.items():
        print(f"  {metric_name}: {metric_value:.4f} ({metric_value*100:.2f}%)")
    
    print(f"\nDataset:")
    print(f"  Total problems: {total_problems}")
    print(f"  Samples per problem: {n_samples}")
    print(f"  Total generations: {total_problems * n_samples}")
    
    # Save results
    summary = {
        'config': {
            'dataset': 'math-ai/aime24',
            'model': model,
            'n_samples': n_samples,
            'max_tokens': max_tokens,
            'temperature': temperature,
            'top_p': top_p,
            'top_k': top_k,
            'total_problems': total_problems,
            'evaluation_type': 'single_turn',
            'timestamp': datetime.now().isoformat()
        },
        'metrics': {
            f'avg@{n_samples}': final_avg_score,
            f'best@{n_samples}': final_best_score,
            f'std@{n_samples}': final_std,
            'pass@1': pass_at_1,
            **best_of_n_scores
        },
        'detailed_results': all_results
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 Results saved to: {output_file}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description='AIME24 Single Turn Evaluation (Reference: CURE/eval.sh)'
    )
    
    # Server configuration
    parser.add_argument('--host', type=str, default='localhost',
                        help='vLLM server host')
    parser.add_argument('--port', type=int, default=9001,
                        help='vLLM server port')
    parser.add_argument('--model', type=str, default='Qwen/Qwen2.5-Math-1.5B',
                        help='Model name')
    
    # Generation parameters (from CURE eval.sh)
    parser.add_argument('--n-samples', type=int, default=8,
                        help='Number of samples per problem (default: 32)')
    parser.add_argument('--max-tokens', type=int, default=4096,
                        help='Max response tokens (default: 4096-512)')
    parser.add_argument('--temperature', type=float, default=0.6,
                        help='Sampling temperature (default: 0.6)')
    parser.add_argument('--top-p', type=float, default=0.95,
                        help='Top-p sampling (default: 0.95)')
    parser.add_argument('--top-k', type=int, default=-1,
                        help='Top-k sampling (default: -1, disabled)')
    
    # Output configuration
    parser.add_argument('--output', type=str, required=True,
                        help='Output JSON file path')
    
    args = parser.parse_args()
    
    print(f"\n🚀 Starting AIME24 Single Turn Evaluation")
    print(f"Parameters: n={args.n_samples}, temp={args.temperature}, top_p={args.top_p}\n")
    
    evaluate_aime24_single_turn(
        host=args.host,
        port=args.port,
        model=args.model,
        n_samples=args.n_samples,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        output_file=args.output
    )
    
    print(f"\n✓ Evaluation completed!")


if __name__ == "__main__":
    main()
