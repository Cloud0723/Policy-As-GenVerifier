#!/usr/bin/env python3
"""
比较 Single-turn 和 Multi-turn 的准确率
"""

import json
from pathlib import Path
from collections import defaultdict

def load_trajectories(trajectory_dir):
    """加载所有轨迹文件"""
    trajectory_dir = Path(trajectory_dir)
    trajectory_files = sorted(trajectory_dir.glob("traj_*.json"))
    
    results = {}
    for traj_file in trajectory_files:
        with open(traj_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 提取模式（single-turn 或 multi-turn）
        mode = "single-turn" if "single-turn" in traj_file.name else "multi-turn"
        question_idx = data.get('question_idx', -1)
        
        if question_idx not in results:
            results[question_idx] = {}
        
        results[question_idx][mode] = {
            'accuracy': data.get('accuracy', 0),
            'correct_count': data.get('correct_count', 0),
            'total_samples': data.get('total_samples', 0),
            'question': data.get('question', '')[:100] + '...' if len(data.get('question', '')) > 100 else data.get('question', ''),
            'ground_truth': data.get('ground_truth', ''),
        }
        
        # Multi-turn 特有信息
        if mode == "multi-turn":
            results[question_idx][mode]['avg_turns'] = data.get('avg_turns', 0)
            results[question_idx][mode]['completion_rate'] = data.get('completion_rate', 0)
    
    return results

def compare_results(single_turn_dir, multi_turn_dir):
    """比较两个目录的结果"""
    print("=" * 80)
    print("Single-turn vs Multi-turn 准确率对比")
    print("=" * 80)
    
    single_results = load_trajectories(single_turn_dir)
    multi_results = load_trajectories(multi_turn_dir)
    
    # 找到共同的问题
    common_questions = set(single_results.keys()) & set(multi_results.keys())
    
    if not common_questions:
        print("❌ 没有找到共同的问题进行对比")
        return
    
    print(f"\n找到 {len(common_questions)} 个共同问题\n")
    
    # 统计信息
    single_total_correct = 0
    single_total_samples = 0
    multi_total_correct = 0
    multi_total_samples = 0
    
    improvements = []
    degradations = []
    same_results = []
    
    print("问题级别对比:")
    print("-" * 80)
    print(f"{'Q#':<5} {'Single-turn':<15} {'Multi-turn':<15} {'变化':<10} {'提升':<10}")
    print("-" * 80)
    
    for q_idx in sorted(common_questions):
        single = single_results[q_idx].get('single-turn', {})
        multi = multi_results[q_idx].get('multi-turn', {})
        
        single_acc = single.get('accuracy', 0)
        multi_acc = multi.get('accuracy', 0)
        
        single_total_correct += single.get('correct_count', 0)
        single_total_samples += single.get('total_samples', 0)
        multi_total_correct += multi.get('correct_count', 0)
        multi_total_samples += multi.get('total_samples', 0)
        
        diff = multi_acc - single_acc
        diff_str = f"{diff:+.1f}%"
        
        if diff > 0:
            improvements.append((q_idx, diff))
            status = "✅ 提升"
        elif diff < 0:
            degradations.append((q_idx, diff))
            status = "❌ 下降"
        else:
            same_results.append((q_idx, diff))
            status = "➖ 相同"
        
        print(f"{q_idx:<5} {single_acc:>6.1f}% ({single.get('correct_count', 0)}/{single.get('total_samples', 0)})  "
              f"{multi_acc:>6.1f}% ({multi.get('correct_count', 0)}/{multi.get('total_samples', 0)})  "
              f"{diff_str:>8}  {status}")
        
        # Multi-turn 额外信息
        if 'avg_turns' in multi:
            print(f"      └─ Multi-turn: 平均 {multi['avg_turns']:.2f} 轮, 完成率 {multi['completion_rate']:.1f}%")
    
    print("-" * 80)
    
    # 总体统计
    single_overall = (single_total_correct / single_total_samples * 100) if single_total_samples > 0 else 0
    multi_overall = (multi_total_correct / multi_total_samples * 100) if multi_total_samples > 0 else 0
    overall_diff = multi_overall - single_overall
    
    print(f"\n总体准确率:")
    print(f"  Single-turn: {single_overall:.2f}% ({single_total_correct}/{single_total_samples})")
    print(f"  Multi-turn:  {multi_overall:.2f}% ({multi_total_correct}/{multi_total_samples})")
    print(f"  变化: {overall_diff:+.2f}%")
    
    if overall_diff > 0:
        print(f"  ✅ Multi-turn 准确率提升了 {overall_diff:.2f}%")
    elif overall_diff < 0:
        print(f"  ❌ Multi-turn 准确率下降了 {abs(overall_diff):.2f}%")
    else:
        print(f"  ➖ 准确率相同")
    
    # 详细统计
    print(f"\n问题级别统计:")
    print(f"  提升的问题: {len(improvements)} 个")
    if improvements:
        avg_improvement = sum(diff for _, diff in improvements) / len(improvements)
        print(f"    平均提升: {avg_improvement:.2f}%")
        print(f"    最大提升: {max(diff for _, diff in improvements):.2f}% (问题 {max(improvements, key=lambda x: x[1])[0]})")
    
    print(f"  下降的问题: {len(degradations)} 个")
    if degradations:
        avg_degradation = sum(diff for _, diff in degradations) / len(degradations)
        print(f"    平均下降: {avg_degradation:.2f}%")
        print(f"    最大下降: {min(diff for _, diff in degradations):.2f}% (问题 {min(degradations, key=lambda x: x[1])[0]})")
    
    print(f"  相同的问题: {len(same_results)} 个")
    
    print("=" * 80)

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("用法: python compare_results.py <single_turn_dir> <multi_turn_dir>")
        print("\n示例:")
        print("  python compare_results.py output/trajectories_gpt-4o_20251219_030048 output/trajectories_gpt-4o_20251219_025233")
        sys.exit(1)
    
    single_turn_dir = sys.argv[1]
    multi_turn_dir = sys.argv[2]
    
    compare_results(single_turn_dir, multi_turn_dir)
