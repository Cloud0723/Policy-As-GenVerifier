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
vLLM Direct Multi-turn Rollout Test Script

This script tests vLLM direct multi-turn rollout functionality for dialogue generation and reward evaluation.
Key features:
- Mathematical reasoning problems in multi-turn
- Integrated reward model for result evaluation
- Distributed inference and tensor parallelism support
- Comprehensive result display and analysis
"""

import os
import numpy as np
from types import SimpleNamespace
from transformers import AutoTokenizer, AutoConfig
import torch
from omegaconf import DictConfig
from tensordict import TensorDict

from verl.utils.distributed import initialize_global_process_group
from verl import DataProto
from verl.utils.torch_functional import pad_sequence_to_length
from verl.workers.rollout.vllm_rollout.vllm_multiturn_rollout_spmd import vLLMMutliTurnRollout
from verl.workers.reward_manager.multiturn import MultiTurnRewardManager


def prepare_test_data():
    """Prepare test dataset"""
    messages_list = [
        [
            {"role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{}."},
            {"role": "user", "content": "Calculate what 1+1 equals?"}
        ],
        [
            {"role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{}."},
            {"role": "user", "content": "Simplify $\\sqrt{242}$."}
        ],
        [
            {"role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{}."},
            {"role": "user", "content": "A geometric sequence has first three terms 2, 4, 8. Find the 5th term."}
        ]
    ]
    
    # Standard answers
    answers = ["2", "11\\sqrt{2}", "32"]
    
    return messages_list, answers


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


def create_rollout_config(tensor_model_parallel_size: int, max_prompt_length: int, response_length: int):
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
        'n': 1,  # Number of sequences per prompt
        'temperature': 0.7,
        'top_p': 0.9,
        'max_num_batched_tokens': 8192 * 4,
        'max_model_len': 1024 * 4,
        'num_turns': 2,
        "val_kwargs": {
            "num_turns": 2,
        }
    }


def evaluate_outputs(outputs, tokenizer, preencode_prompts, answers, rollout_config):
    """Evaluate generation results"""
    vllm_output = outputs.batch['responses']
    n = rollout_config['n']
    
    # Print generation results
    if torch.distributed.get_rank() == 0:
        print("\n" + "="*60)
        print("Generation Results")
        print("="*60)
        
        for i in range(len(preencode_prompts) * n):
            print(f"\n===== Test Case {i+1} =====")
            
            multiturn_mask = outputs.batch['multiturn_mask'][i]
            response = vllm_output[i]
            
            # Get valid response (excluding prompt)
            valid_response = outputs.batch['input_ids'][i]
            
            print(f"Cases {i+1} : {tokenizer.decode(valid_response, skip_special_tokens=True)}")
    
    # Reward evaluation
    config = SimpleNamespace()
    config.reward_model = {}
    
    reward_manager = MultiTurnRewardManager(
        tokenizer=tokenizer, 
        num_examine=2,  # Print detailed info for first two samples
        config=config.reward_model
    )
    
    # Prepare reward evaluation data
    non_tensor_batch = {
        'reward_model': np.array([
            {'ground_truth': answer} for answer in answers
        ], dtype=object),
        'final_generation_turn': np.array([rollout_config['num_turns']-1] * len(answers), dtype=np.int32)
    }
    outputs.non_tensor_batch.update(non_tensor_batch)
    reward_data = DataProto(batch=outputs.batch, non_tensor_batch=outputs.non_tensor_batch)
    
    # Calculate rewards and metrics
    rewards, metrics = reward_manager(reward_data)
    
    print("\n" + "="*60)
    print("Evaluation Metrics")
    print("="*60)
    for metric_name, metric_value in metrics.items():
        if isinstance(metric_value, (int, float)):
            print(f"{metric_name}: {metric_value:.4f}")
        else:
            print(f"{metric_name}: {metric_value}")
    
    return rewards, metrics


def main():
    """Main function"""
    # Environment check
    assert torch.cuda.is_available(), 'CUDA must be available to run this example'
    
    # Initialize distributed environment
    local_rank, rank, world_size = initialize_global_process_group()
    
    # Configuration parameters
    model_path = os.environ.get('MODEL_PATH', "Qwen/Qwen2.5-7B-Instruct") # define your model path here.
    max_prompt_length = 200
    response_length = 200
    tensor_model_parallel_size = 1
    
    # Prepare test data
    messages_list, answers = prepare_test_data()
    
    # Initialize model and tokenizer
    print("Setting up model and tokenizer...")
    tokenizer, actor_model_config = setup_model_and_tokenizer(model_path)
    
    # Prepare input prompts
    prompts_data, preencode_prompts = prepare_prompts(
        messages_list, tokenizer, max_prompt_length
    )
    
    # Create inference configuration
    rollout_config = create_rollout_config(
        tensor_model_parallel_size, max_prompt_length, response_length
    )
    
    # Initialize vLLM direct multi-turn rollout
    print("Initializing vLLM direct multi-turn rollout...")
    rollout = vLLMMutliTurnRollout(
        model_path=model_path,
        config=DictConfig(rollout_config),
        tokenizer=tokenizer,
        model_hf_config=actor_model_config,
        trust_remote_code=True
    )
    # Start generation
    print("Starting sequence generation...")
    outputs = rollout.generate_sequences(prompts_data)
    
    # Evaluate results
    print("Evaluating results...")
    rewards, metrics = evaluate_outputs(
        outputs, tokenizer, preencode_prompts, answers, rollout_config
    )
    
    print("\n" + "="*80)
    print("TEST COMPLETED SUCCESSFULLY")
    print("="*80)
    
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
