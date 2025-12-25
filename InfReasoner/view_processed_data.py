#!/usr/bin/env python3
"""
查看处理后的数据集
"""

import pandas as pd
import argparse
from pathlib import Path


def view_data(file_path: str, num_samples: int = 3, show_full: bool = False):
    """
    查看处理后的数据
    
    Args:
        file_path: parquet 文件路径
        num_samples: 显示的样本数量
        show_full: 是否显示完整的 response
    """
    print(f"读取数据: {file_path}")
    df = pd.read_parquet(file_path)
    
    print(f"\n数据集大小: {len(df)} 条")
    print(f"\n列名: {list(df.columns)}")
    
    if 'is_first_turn' in df.columns:
        print(f"\n统计信息:")
        print(f"  - 第一轮 turn: {df['is_first_turn'].sum()} 条")
        print(f"  - 后续 turn: {(~df['is_first_turn']).sum()} 条")
        
        if 'has_summary' in df.columns:
            print(f"  - 有 summary: {df['has_summary'].sum()} 条")
        if 'has_conclusion' in df.columns:
            print(f"  - 有 conclusion: {df['has_conclusion'].sum()} 条")
    
    # 显示第一轮 turn 的示例
    if 'is_first_turn' in df.columns:
        first_turn_df = df[df['is_first_turn']]
        if len(first_turn_df) > 0:
            print("\n" + "="*100)
            print("第一轮 TURN 示例")
            print("="*100)
            
            for i in range(min(num_samples, len(first_turn_df))):
                sample = first_turn_df.iloc[i]
                print(f"\n【示例 {i+1}】")
                print(f"\n[PROMPT]")
                print(sample['prompt'])
                
                if show_full:
                    print(f"\n[RESPONSE (完整)]")
                    print(sample['response'])
                else:
                    print(f"\n[RESPONSE (前 500 字符)]")
                    response_preview = sample['response'][:500]
                    print(response_preview + ("..." if len(sample['response']) > 500 else ""))
                
                print(f"\n[长度] Prompt: {len(sample['prompt'])} 字符, Response: {len(sample['response'])} 字符")
                print("-"*100)
    
        # 显示后续 turn 的示例
        later_turn_df = df[~df['is_first_turn']]
        if len(later_turn_df) > 0:
            print("\n" + "="*100)
            print("后续 TURN 示例")
            print("="*100)
            
            for i in range(min(num_samples, len(later_turn_df))):
                sample = later_turn_df.iloc[i]
                print(f"\n【示例 {i+1}】")
                print(f"\n[PROMPT]")
                print(sample['prompt'])
                
                if show_full:
                    print(f"\n[RESPONSE (完整)]")
                    print(sample['response'])
                else:
                    print(f"\n[RESPONSE (前 500 字符)]")
                    response_preview = sample['response'][:500]
                    print(response_preview + ("..." if len(sample['response']) > 500 else ""))
                
                print(f"\n[长度] Prompt: {len(sample['prompt'])} 字符, Response: {len(sample['response'])} 字符")
                print("-"*100)
    else:
        # 没有 is_first_turn 字段，直接显示前几条
        print("\n" + "="*100)
        print("数据示例")
        print("="*100)
        
        for i in range(min(num_samples, len(df))):
            sample = df.iloc[i]
            print(f"\n【示例 {i+1}】")
            print(f"\n[PROMPT]")
            print(sample['prompt'])
            
            if show_full:
                print(f"\n[RESPONSE (完整)]")
                print(sample['response'])
            else:
                print(f"\n[RESPONSE (前 500 字符)]")
                response_preview = sample['response'][:500]
                print(response_preview + ("..." if len(sample['response']) > 500 else ""))
            
            print(f"\n[长度] Prompt: {len(sample['prompt'])} 字符, Response: {len(sample['response'])} 字符")
            print("-"*100)


def main():
    parser = argparse.ArgumentParser(description="查看处理后的数据集")
    parser.add_argument(
        "file_path",
        type=str,
        help="Parquet 文件路径"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=3,
        help="显示的样本数量（默认：3）"
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="显示完整的 response"
    )
    
    args = parser.parse_args()
    
    file_path = Path(args.file_path)
    if not file_path.exists():
        print(f"错误：文件不存在 - {file_path}")
        return
    
    view_data(str(file_path), args.num_samples, args.full)


if __name__ == "__main__":
    main()

