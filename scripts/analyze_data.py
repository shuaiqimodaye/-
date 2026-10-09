"""
数据分析与统计脚本
功能：
1. 统计清洗后数据的分布
2. 验证配对数据的格式和质量
3. 抽样展示数据样本
"""
import os
import json
import argparse
from pathlib import Path
from collections import Counter


def analyze_cleaned_data(cleaned_dir: str):
    """分析清洗后的数据"""
    cleaned_path = Path(cleaned_dir)
    print("=" * 60)
    print("清洗数据统计")
    print("=" * 60)

    total = 0
    category_stats = []

    for jsonl_file in sorted(cleaned_path.glob("*.jsonl")):
        if jsonl_file.name == "all_cleaned.jsonl":
            continue

        count = 0
        lengths = []
        with open(jsonl_file, encoding="utf-8") as f:
            for line in f:
                record = json.loads(line.strip())
                count += 1
                lengths.append(len(record["text"]))

        if count > 0:
            avg_len = sum(lengths) / len(lengths)
            min_len = min(lengths)
            max_len = max(lengths)
            category_stats.append({
                "category": jsonl_file.stem,
                "count": count,
                "avg_len": avg_len,
                "min_len": min_len,
                "max_len": max_len
            })
            total += count

    print(f"\n分类数: {len(category_stats)}")
    print(f"总样本数: {total}")
    print(f"\n{'分类':<12} {'样本数':>8} {'平均长度':>10} {'最短':>6} {'最长':>6}")
    print("-" * 50)
    for stat in sorted(category_stats, key=lambda x: -x["count"]):
        print(f"{stat['category']:<12} {stat['count']:>8} "
              f"{stat['avg_len']:>10.1f} {stat['min_len']:>6} {stat['max_len']:>6}")


def analyze_paired_data(paired_file: str):
    """分析配对数据"""
    print("\n" + "=" * 60)
    print("配对数据统计")
    print("=" * 60)

    if not os.path.exists(paired_file):
        print(f"文件不存在: {paired_file}")
        return

    records = []
    with open(paired_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    print(f"总样本数: {len(records)}")

    # 检查格式
    required_keys = {"instruction", "input", "output"}
    format_errors = 0
    for r in records:
        if not required_keys.issubset(r.keys()):
            format_errors += 1

    print(f"格式错误数: {format_errors}")

    # 长度统计
    input_lengths = [len(r["input"]) for r in records]
    output_lengths = [len(r["output"]) for r in records]

    print(f"\n输入文本: 平均长度 {sum(input_lengths)/len(input_lengths):.1f}, "
          f"范围 [{min(input_lengths)}, {max(input_lengths)}]")
    print(f"输出文本: 平均长度 {sum(output_lengths)/len(output_lengths):.1f}, "
          f"范围 [{min(output_lengths)}, {max(output_lengths)}]")

    # 长度比统计
    ratios = [len(r["output"]) / max(len(r["input"]), 1) for r in records]
    avg_ratio = sum(ratios) / len(ratios)
    print(f"输出/输入 长度比: 平均 {avg_ratio:.2f}, "
          f"范围 [{min(ratios):.2f}, {max(ratios):.2f}]")

    # 抽样展示
    print(f"\n--- 随机抽样 5 条 ---")
    import random
    random.seed(42)
    samples = random.sample(records, min(5, len(records)))
    for i, s in enumerate(samples, 1):
        print(f"\n样本 {i}:")
        print(f"  指令: {s['instruction'][:60]}...")
        print(f"  输入: {s['input'][:80]}...")
        print(f"  输出: {s['output'][:80]}...")


def main():
    parser = argparse.ArgumentParser(description="数据分析与统计")
    parser.add_argument("--cleaned-dir", type=str,
                        default=r"E:\毕业论文\data\cleaned")
    parser.add_argument("--paired-file", type=str,
                        default=r"E:\毕业论文\data\paired\style_paired.jsonl")
    args = parser.parse_args()

    if os.path.exists(args.cleaned_dir):
        analyze_cleaned_data(args.cleaned_dir)

    analyze_paired_data(args.paired_file)


if __name__ == "__main__":
    main()
