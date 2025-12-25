#!/usr/bin/env python3
"""
数据预处理脚本：处理 ZJU-REAL/InftyThink 数据集
将数据集转换为适合 SFT 训练的格式
"""

import pandas as pd
import argparse
from pathlib import Path
from datasets import load_dataset
import sys
sys.path.append(str(Path(__file__).parent / "prompt"))
from prompt_multi_turn_v2_lcl import FIRST_TURN_REASONING_PROMPT, MIDDLE_TURN_REASONING_PROMPT


def process_infty_think_dataset(
    dataset_name: str = "ZJU-REAL/InftyThink",
    subset: str = "default",
    split: str = "InftyThink__OpenR1__default__2k",
    output_dir: str = None,
    max_samples: int = None
):
    """
    处理 InftyThink 数据集
    
    Args:
        dataset_name: HuggingFace 数据集名称
        subset: 数据集子集
        split: 数据集分割
        output_dir: 输出目录
        max_samples: 最大样本数（用于测试）
    """
    print(f"加载数据集: {dataset_name}/{subset}/{split}")
    
    # 加载数据集
    dataset = load_dataset(dataset_name, subset, split=split)
    
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
        print(f"使用前 {len(dataset)} 个样本进行测试")
    
    print(f"数据集大小: {len(dataset)}")
    
    # 处理数据
    processed_data = []
    
    for idx, item in enumerate(dataset):
        question = item['question']
        history = item['history']
        reasoning_process = item['reasoning_process']
        summary = item['summary']
        conclusion = item['conclusion']
        
        # 判断是否为第一个 turn（history 为空）
        is_first_turn = not history or history.strip() == "" or history == "null"
        
        if is_first_turn:
            # 第一个 turn：使用 FIRST_TURN_REASONING_PROMPT
            prompt = FIRST_TURN_REASONING_PROMPT.format(question=question)
            
            # response = reasoning_process + summary（直接拼接，不加额外标签）
            if summary and summary.strip() and summary != "null":
                response = f"{reasoning_process}\n{summary}"
            else:
                # 如果没有 summary，只有 reasoning_process
                response = reasoning_process
        else:
            # 后续 turn：使用 MIDDLE_TURN_REASONING_PROMPT
            prompt = MIDDLE_TURN_REASONING_PROMPT.format(
                question=question,
                summary_reasoning=history
            )
            
            # response = reasoning_process + conclusion
            if conclusion and conclusion.strip() and conclusion != "null":
                response = f"{reasoning_process}\n{conclusion}"
            else:
                # 如果没有 conclusion，只有 reasoning_process
                response = reasoning_process
        
        processed_data.append({
            'uuid': item.get('uuid', f'sample_{idx}'),
            'question': question,
            'prompt': prompt,
            'response': response,
            'is_first_turn': is_first_turn,
            'turn_number': item.get('reasoning_index', 1 if is_first_turn else 2),
            'has_summary': bool(summary and summary.strip() and summary != "null"),
            'has_conclusion': bool(conclusion and conclusion.strip() and conclusion != "null")
        })
        
        # 打印进度
        if (idx + 1) % 10000 == 0:
            print(f"已处理 {idx + 1} 条数据...")
    
    # 转换为 DataFrame
    df = pd.DataFrame(processed_data)
    
    print(f"\n处理完成！总共 {len(df)} 条数据")
    print(f"  - 第一轮 turn: {df['is_first_turn'].sum()} 条")
    print(f"    - 有 summary: {df[df['is_first_turn']]['has_summary'].sum()} 条")
    print(f"  - 后续 turn: {(~df['is_first_turn']).sum()} 条")
    print(f"    - 有 conclusion: {df[~df['is_first_turn']]['has_conclusion'].sum()} 条")
    
    # 保存数据
    if output_dir is None:
        output_dir = Path.home() / "data" / "InftyThink"
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存为 parquet 格式（推荐用于大数据集）
    output_file = output_dir / f"{split}_processed.parquet"
    df.to_parquet(output_file, index=False)
    print(f"\n数据已保存到: {output_file}")
    
    # 同时保存一份带有所有字段的版本
    output_file_full = output_dir / f"{split}_processed_full.parquet"
    df.to_parquet(output_file_full, index=False)
    print(f"完整数据已保存到: {output_file_full}")
    
    # 显示示例
    print("\n" + "="*80)
    print("示例数据（第一个 first turn）:")
    print("="*80)
    first_turn_sample = df[df['is_first_turn']].iloc[0]
    print(f"\n[PROMPT]")
    print(first_turn_sample['prompt'])
    print(f"\n[RESPONSE (前 800 字符)]")
    response_preview = first_turn_sample['response'][:800]
    print(response_preview + ("..." if len(first_turn_sample['response']) > 800 else ""))
    
    if (~df['is_first_turn']).any():
        print("\n" + "="*80)
        print("示例数据（第一个后续 turn）:")
        print("="*80)
        later_turn_sample = df[~df['is_first_turn']].iloc[0]
        print(f"\n[PROMPT]")
        print(later_turn_sample['prompt'])
        print(f"\n[RESPONSE (前 800 字符)]")
        response_preview = later_turn_sample['response'][:800]
        print(response_preview + ("..." if len(later_turn_sample['response']) > 800 else ""))
    
    return df


def create_train_val_split(
    df: pd.DataFrame,
    output_dir: str,
    val_ratio: float = 0.05,
    random_seed: int = 42
):
    """
    创建训练集和验证集分割
    
    Args:
        df: 处理后的 DataFrame
        output_dir: 输出目录
        val_ratio: 验证集比例
        random_seed: 随机种子
    """
    from sklearn.model_selection import train_test_split
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 只保留训练需要的字段
    df_sft = df[['prompt', 'response']].copy()
    
    # 分割训练集和验证集
    train_df, val_df = train_test_split(
        df_sft,
        test_size=val_ratio,
        random_state=random_seed
    )
    
    print(f"\n创建训练/验证集分割:")
    print(f"  - 训练集: {len(train_df)} 条")
    print(f"  - 验证集: {len(val_df)} 条")
    
    # 保存
    train_file = output_dir / "train.parquet"
    val_file = output_dir / "test.parquet"  # verl 使用 test.parquet 作为验证集
    
    train_df.to_parquet(train_file, index=False)
    val_df.to_parquet(val_file, index=False)
    
    print(f"\n训练集已保存到: {train_file}")
    print(f"验证集已保存到: {val_file}")
    
    return train_df, val_df


def main():
    parser = argparse.ArgumentParser(description="预处理 InftyThink 数据集")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="ZJU-REAL/InftyThink",
        help="HuggingFace 数据集名称"
    )
    parser.add_argument(
        "--subset",
        type=str,
        default="default",
        help="数据集子集"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="InftyThink__OpenR1__default__2k",
        help="数据集分割"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="输出目录（默认：~/data/InftyThink）"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="最大样本数（用于测试）"
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.05,
        help="验证集比例（默认：0.05）"
    )
    parser.add_argument(
        "--no_split",
        action="store_true",
        help="不创建训练/验证集分割"
    )
    
    args = parser.parse_args()
    
    # 处理数据集
    df = process_infty_think_dataset(
        dataset_name=args.dataset_name,
        subset=args.subset,
        split=args.split,
        output_dir=args.output_dir,
        max_samples=args.max_samples
    )
    
    # 创建训练/验证集分割
    if not args.no_split:
        output_dir = args.output_dir or str(Path.home() / "data" / "InftyThink")
        create_train_val_split(df, output_dir, args.val_ratio)
    
    print("\n✅ 数据预处理完成！")


if __name__ == "__main__":
    main()

