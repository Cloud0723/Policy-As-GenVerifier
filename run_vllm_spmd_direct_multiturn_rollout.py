# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
vLLM Direct Multi-turn Rollout Test Script for AIME24 Dataset

This script evaluates AIME24 dataset using vLLM direct multi-turn rollout functionality.
Key features:
- AIME24 mathematical reasoning problems in multi-turn
- Generate 8 samples per problem (avg@8)
- Integrated reward model for result evaluation
- Distributed inference and tensor parallelism support
- Comprehensive result display and analysis
"""

import os
import re
import numpy as np
from types import SimpleNamespace
from transformers import AutoTokenizer, AutoConfig
import torch
from omegaconf import DictConfig
from tensordict import TensorDict
from datasets import load_dataset
from tqdm import tqdm

from verl.utils.distributed import initialize_global_process_group
from verl import DataProto
from verl.utils.torch_functional import pad_sequence_to_length
from verl.workers.rollout.vllm_rollout.vllm_multiturn_rollout_spmd import vLLMMutliTurnRollout
from verl.workers.reward_manager.multiturn import MultiTurnRewardManager

# Import math_verify for answer verification
try:
    from math_verify.metric import math_metric
    from math_verify.parser import LatexExtractionConfig, ExprExtractionConfig
    from math_verify.errors import TimeoutException
    MATH_VERIFY_AVAILABLE = True
except ImportError:
    print("Warning: math_verify not available, will use simple string matching")
    MATH_VERIFY_AVAILABLE = False


def extract_boxed_answer(solution_str: str) -> str:
    """Extract answer from \\boxed{} format"""
    boxed_match = re.search(r'\\boxed\{([^}]+)\}', solution_str)
    if boxed_match:
        return boxed_match.group(1)
    return solution_str


def verify_answer(solution_str: str, ground_truth: str) -> dict:
    """Verify answer using math_verify"""
    if not MATH_VERIFY_AVAILABLE:
        # Fallback: simple string matching
        solution_str = solution_str[-300:]  # Last 300 chars
        pred = extract_boxed_answer(solution_str)
        # Simple check if ground truth appears in prediction
        acc = 1.0 if ground_truth in pred or pred in ground_truth else 0.0
        return {"score": acc, "acc": acc, "pred": pred}
    
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


def prepare_aime24_data(max_problems=None):
    """Load AIME24 dataset and prepare messages"""
    print("Loading AIME24 dataset...")
    dataset = load_dataset("math-ai/aime24", split="test")
    
    if max_problems is not None:
        dataset = dataset.select(range(min(max_problems, len(dataset))))
    
    messages_list = []
    ground_truths = []
    problem_ids = []
    
    # System prompt for first turn (similar to prompt.py)
    first_turn_system_prompt = """You are a mathematics expert. Please provide step-by-step reasoning to solve the given question.
This is the first turn.

Instructions
You have sufficient context length in this turn and may reason freely.

- If the reasoning becomes too long to fit into the remaining context, you should first summarize
  the current reasoning for use in the next turn.
- If you have completed all necessary reasoning, you may directly provide the answer.
- If you produce a summary, it should capture all key assumptions, intermediate results, and
  conclusions needed to continue reasoning in the next turn.

Output Format
CRITICAL: Your response must ONLY contain XML tags and their content. Do NOT include any text outside the tags.
- Allowed tags: <think>, <summary>, <answer>
- Each tag appears at most ONCE
- No explanations, no additional text, ONLY the XML tags
- REQUIRED: You must provide <think> tag
- REQUIRED: You must provide either <summary> (if you need more turns) OR <answer> (if you're done), but NOT both

Question: {question}"""
    
    for item in dataset:
        question = item['problem']
        solution = item['solution']
        ground_truth = extract_boxed_answer(solution)
        
        # Build first turn message
        system_content = first_turn_system_prompt.format(question=question)
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": "Please solve this problem step by step."}
        ]
        
        messages_list.append(messages)
        ground_truths.append(ground_truth)
        problem_ids.append(item.get('id', len(messages_list)))
    
    print(f"✓ Loaded {len(messages_list)} problems from AIME24")
    
    return messages_list, ground_truths, problem_ids


def setup_model_and_tokenizer(model_path: str):
    """Initialize model and tokenizer"""
    print(f"Loading model: {model_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    actor_model_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    
    return tokenizer, actor_model_config


def prepare_prompts(messages_list, tokenizer, max_prompt_length: int):
    """Prepare input prompts"""
    # Apply chat template without generation prompt for rollout
    preencode_prompts = [
        tokenizer.apply_chat_template(messages, tokenize=False) 
        for messages in messages_list
    ]
    
    # Set tokenizer parameters
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'
    
    # Tokenize
    prompts = tokenizer(preencode_prompts, return_tensors='pt', padding=True)
    input_ids = prompts['input_ids']
    attention_mask = prompts['attention_mask']
    
    # Pad to specified length
    input_ids = pad_sequence_to_length(
        input_ids, max_prompt_length, tokenizer.pad_token_id, left_pad=True
    ).cuda()
    attention_mask = pad_sequence_to_length(
        attention_mask, max_prompt_length, 0, left_pad=True
    ).cuda()
    
    # Calculate position IDs
    position_ids = torch.zeros_like(attention_mask, dtype=torch.long).cuda()
    for i in range(len(input_ids)):
        non_pad_positions = attention_mask[i].nonzero().squeeze(-1)
        position_ids[i, non_pad_positions] = torch.arange(
            len(non_pad_positions), device=position_ids.device
        )
    
    # Build batch data
    batch = TensorDict({
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'position_ids': position_ids
    }, batch_size=len(input_ids))
    
    prompts_data = DataProto(batch=batch, meta_info={
        'eos_token_id': [151645, 151643],  # Qwen-specific end tokens
        'do_sample': True
    })
    
    return prompts_data, preencode_prompts


def create_rollout_config(tensor_model_parallel_size: int, max_prompt_length: int, response_length: int, n_samples: int = 8):
    """Create inference configuration"""
    return {
        'tensor_model_parallel_size': tensor_model_parallel_size,
        'prompt_length': max_prompt_length,
        'response_length': response_length,
        'dtype': 'bfloat16',
        'enforce_eager': True,
        'gpu_memory_utilization': 0.8,
        'load_format': 'dummy_dtensor',
        'disable_log_stats': True,
        'enable_chunked_prefill': False,
        'free_cache_engine': False,
        'n': n_samples,  # Number of sequences per prompt (8 for avg@8)
        'temperature': 0.7,
        'top_p': 0.9,
        'max_num_batched_tokens': 8192 * 4,
        'max_model_len': 1024 * 4,
        'num_turns': 2,
        "val_kwargs": {
            "num_turns": 2,
        }
    }


def evaluate_outputs(outputs, tokenizer, preencode_prompts, ground_truths, problem_ids, rollout_config):
    """Evaluate generation results and compute avg@8"""
    n = rollout_config['n']  # Should be 8 for avg@8
    num_problems = len(preencode_prompts)
    
    # Extract full responses (all turns combined)
    input_ids = outputs.batch['input_ids']  # Shape: [num_problems * n, seq_len]
    
    # Group by problem (each problem has n samples)
    problem_accuracies = []
    all_sample_results = []
    
    for prob_idx in range(num_problems):
        problem_id = problem_ids[prob_idx] if prob_idx < len(problem_ids) else prob_idx
        ground_truth = ground_truths[prob_idx]
        
        # Get all n samples for this problem
        sample_accs = []
        sample_results = []
        
        for sample_idx in range(n):
            sample_global_idx = prob_idx * n + sample_idx
            if sample_global_idx >= len(input_ids):
                continue
                
            # Decode the full response
            full_response = tokenizer.decode(input_ids[sample_global_idx], skip_special_tokens=True)
            
            # Verify answer
            verify_result = verify_answer(full_response, ground_truth)
            acc = verify_result['acc']
            sample_accs.append(acc)
            
            sample_results.append({
                'sample_idx': sample_idx,
                'response': full_response,
                'predicted_answer': verify_result['pred'],
                'accuracy': acc
            })
        
        # Calculate avg@8 for this problem
        if len(sample_accs) > 0:
            avg_acc = np.mean(sample_accs)
            best_acc = np.max(sample_accs)
            problem_accuracies.append(avg_acc)
            all_sample_results.append({
                'problem_id': problem_id,
                'problem_idx': prob_idx,
                'ground_truth': ground_truth,
                'avg@8': avg_acc,
                'best@8': best_acc,
                'samples': sample_results
            })
            
            if torch.distributed.get_rank() == 0:
                print(f"Problem {prob_idx+1}/{num_problems} (ID: {problem_id}): "
                      f"Avg@8={avg_acc:.4f}, Best@8={best_acc:.4f}")
    
    # Calculate overall avg@8
    if len(problem_accuracies) > 0:
        overall_avg8 = np.mean(problem_accuracies)
        overall_best8 = np.mean([r['best@8'] for r in all_sample_results])
        overall_std = np.std(problem_accuracies)
        
        # Calculate pass@1 (at least one correct per problem)
        pass_at_1 = np.mean([1 if r['best@8'] > 0 else 0 for r in all_sample_results])
    else:
        overall_avg8 = 0.0
        overall_best8 = 0.0
        overall_std = 0.0
        pass_at_1 = 0.0
    
    # Print final results
    if torch.distributed.get_rank() == 0:
        print("\n" + "="*80)
        print("📊 AIME24 Evaluation Results (Multi-turn)")
        print("="*80)
        print(f"\nOverall Metrics:")
        print(f"  Avg@8: {overall_avg8:.4f} ({overall_avg8*100:.2f}%)")
        print(f"  Best@8 (average): {overall_best8:.4f} ({overall_best8*100:.2f}%)")
        print(f"  Std@8: {overall_std:.4f}")
        print(f"  Pass@1: {pass_at_1:.4f} ({pass_at_1*100:.2f}%)")
        print(f"\nDataset:")
        print(f"  Total problems: {num_problems}")
        print(f"  Samples per problem: {n}")
        print(f"  Total generations: {num_problems * n}")
        print("="*80)
    
    metrics = {
        'avg@8': overall_avg8,
        'best@8': overall_best8,
        'std@8': overall_std,
        'pass@1': pass_at_1,
        'num_problems': num_problems,
        'n_samples': n
    }
    
    return all_sample_results, metrics


def main():
    """Main function"""
    # Environment check
    assert torch.cuda.is_available(), 'CUDA must be available to run this example'
    
    # Initialize distributed environment
    local_rank, rank, world_size = initialize_global_process_group()
    
    # Configuration parameters
    model_path = os.environ.get('MODEL_PATH', "Qwen/Qwen2.5-7B-Instruct")
    max_prompt_length = int(os.environ.get('MAX_PROMPT_LENGTH', '2048'))
    response_length = int(os.environ.get('RESPONSE_LENGTH', '2048'))
    tensor_model_parallel_size = int(os.environ.get('TENSOR_MODEL_PARALLEL_SIZE', '1'))
    n_samples = int(os.environ.get('N_SAMPLES', '8'))  # 8 for avg@8
    max_problems = int(os.environ.get('MAX_PROBLEMS', '0'))  # 0 means all problems
    
    if torch.distributed.get_rank() == 0:
        print("="*80)
        print("AIME24 Multi-turn Evaluation Configuration")
        print("="*80)
        print(f"Model: {model_path}")
        print(f"Max prompt length: {max_prompt_length}")
        print(f"Response length: {response_length}")
        print(f"Tensor parallel size: {tensor_model_parallel_size}")
        print(f"Samples per problem (n): {n_samples}")
        print(f"Max problems (0=all): {max_problems if max_problems > 0 else 'all'}")
        print("="*80)
    
    # Prepare AIME24 data
    messages_list, ground_truths, problem_ids = prepare_aime24_data(
        max_problems=max_problems if max_problems > 0 else None
    )
    
    # Initialize model and tokenizer
    if torch.distributed.get_rank() == 0:
        print("Setting up model and tokenizer...")
    tokenizer, actor_model_config = setup_model_and_tokenizer(model_path)
    
    # Process problems in batches to avoid memory issues
    batch_size = int(os.environ.get('BATCH_SIZE', '1'))  # Process one problem at a time by default
    
    all_results = []
    all_metrics = []
    
    for batch_start in range(0, len(messages_list), batch_size):
        batch_end = min(batch_start + batch_size, len(messages_list))
        batch_messages = messages_list[batch_start:batch_end]
        batch_ground_truths = ground_truths[batch_start:batch_end]
        batch_problem_ids = problem_ids[batch_start:batch_end]
        
        if torch.distributed.get_rank() == 0:
            print(f"\nProcessing batch {batch_start//batch_size + 1}/{(len(messages_list)-1)//batch_size + 1} "
                  f"(problems {batch_start+1}-{batch_end})")
        
        # Prepare input prompts for this batch
        prompts_data, preencode_prompts = prepare_prompts(
            batch_messages, tokenizer, max_prompt_length
        )
        
        # Create inference configuration
        rollout_config = create_rollout_config(
            tensor_model_parallel_size, max_prompt_length, response_length, n_samples=n_samples
        )
        
        # Initialize vLLM direct multi-turn rollout (reuse if possible)
        if batch_start == 0:
            if torch.distributed.get_rank() == 0:
                print("Initializing vLLM direct multi-turn rollout...")
            rollout = vLLMMutliTurnRollout(
                model_path=model_path,
                config=DictConfig(rollout_config),
                tokenizer=tokenizer,
                model_hf_config=actor_model_config,
                trust_remote_code=True
            )
        
        # Start generation
        if torch.distributed.get_rank() == 0:
            print(f"Generating {n_samples} samples for {len(batch_messages)} problem(s)...")
        outputs = rollout.generate_sequences(prompts_data)
        
        # Evaluate results
        if torch.distributed.get_rank() == 0:
            print("Evaluating results...")
        batch_results, batch_metrics = evaluate_outputs(
            outputs, tokenizer, preencode_prompts, batch_ground_truths, 
            batch_problem_ids, rollout_config
        )
        
        all_results.extend(batch_results)
        all_metrics.append(batch_metrics)
    
    # Aggregate final metrics
    if len(all_metrics) > 0:
        final_avg8 = np.mean([m['avg@8'] for m in all_metrics])
        final_best8 = np.mean([m['best@8'] for m in all_metrics])
        final_pass1 = np.mean([m['pass@1'] for m in all_metrics])
        total_problems = sum([m['num_problems'] for m in all_metrics])
        
        if torch.distributed.get_rank() == 0:
            print("\n" + "="*80)
            print("🎯 FINAL AIME24 EVALUATION RESULTS")
            print("="*80)
            print(f"Avg@8: {final_avg8:.4f} ({final_avg8*100:.2f}%)")
            print(f"Best@8 (average): {final_best8:.4f} ({final_best8*100:.2f}%)")
            print(f"Pass@1: {final_pass1:.4f} ({final_pass1*100:.2f}%)")
            print(f"Total problems evaluated: {total_problems}")
            print("="*80)
    
    if torch.distributed.get_rank() == 0:
        print("\n✓ Evaluation completed successfully!")
    
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
