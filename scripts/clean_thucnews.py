"""
THUCNews 数据清洗脚本
功能：
1. 遍历 data/raw/THUCNews/ 下各分类目录
2. 读取 .txt 文件，清洗文本（去特殊字符、合并空白、过滤长度）
3. 按分类输出 JSONL 到 data/cleaned/
"""
import os
import re
import json
import argparse
from pathlib import Path


def clean_text(text: str) -> str | None:
    """清洗单条文本"""
    if not text or not text.strip():
        return None

    # 去除 HTML 标签
    text = re.sub(r"<[^>]+>", "", text)
    # 去除 URL
    text = re.sub(r"https?://\S+", "", text)
    # 去除邮箱
    text = re.sub(r"\S+@\S+\.\S+", "", text)
    # 去除特殊字符（保留中英文、数字、常用标点）
    text = re.sub(r"[^\u4e00-\u9fff\u3000-\u303fa-zA-Z0-9，。！？、；：""''（）《》\s]", " ", text)
    # 合并连续空白
    text = re.sub(r"\s+", " ", text).strip()

    # 长度过滤：30-600 字
    if len(text) < 30 or len(text) > 600:
        return None

    return text


def read_file_with_encoding(file_path: str) -> str | None:
    """尝试多种编码读取文件"""
    for encoding in ("utf-8", "gbk", "gb2312", "gb18030", "latin-1"):
        try:
            with open(file_path, encoding=encoding) as f:
                return f.read()
        except (UnicodeDecodeError, UnicodeError):
            continue
    return None


def clean_category(category_dir: str, output_path: str, max_per_category: int = 500) -> int:
    """清洗单个分类的数据"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    txt_files = sorted([
        f for f in os.listdir(category_dir)
        if f.endswith(".txt")
    ])

    count = 0
    skipped = 0

    with open(output_path, "w", encoding="utf-8") as out_f:
        for file_name in txt_files[:max_per_category]:
            file_path = os.path.join(category_dir, file_name)
            text = read_file_with_encoding(file_path)

            if text is None:
                skipped += 1
                continue

            cleaned = clean_text(text)
            if cleaned is None:
                skipped += 1
                continue

            record = {
                "source": "thucnews",
                "category": os.path.basename(category_dir),
                "text": cleaned
            }
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1

    return count


def main():
    parser = argparse.ArgumentParser(description="清洗 THUCNews 数据")
    parser.add_argument("--raw-dir", type=str,
                        default=r"E:\毕业论文\data\raw\THUCNews",
                        help="原始数据目录")
    parser.add_argument("--output-dir", type=str,
                        default=r"E:\毕业论文\data\cleaned",
                        help="输出目录")
    parser.add_argument("--max-per-category", type=int, default=500,
                        help="每个分类最多保留的样本数")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    output_dir = Path(args.output_dir)

    if not raw_dir.exists():
        print(f"错误：原始数据目录不存在: {raw_dir}")
        return

    # 只处理 14 个正确中文分类目录，跳过残留的乱码目录
    VALID_CATEGORIES = {
        "体育", "娱乐", "家居", "彩票", "房产", "教育",
        "时尚", "时政", "星座", "游戏", "社会", "科技",
        "股票", "财经"
    }
    category_dirs = sorted([
        d for d in raw_dir.iterdir()
        if d.is_dir() and d.name in VALID_CATEGORIES
    ])
    skipped_dirs = [d.name for d in raw_dir.iterdir()
                    if d.is_dir() and d.name not in VALID_CATEGORIES]
    print(f"发现 {len(category_dirs)} 个有效分类目录")
    if skipped_dirs:
        print(f"跳过 {len(skipped_dirs)} 个无效目录: {skipped_dirs}")

    total_count = 0
    results = []

    for cat_dir in category_dirs:
        category_name = cat_dir.name
        output_path = str(output_dir / f"{category_name}.jsonl")

        print(f"  处理分类: {category_name} ...", end=" ", flush=True)
        count = clean_category(str(cat_dir), output_path, args.max_per_category)
        print(f"{count} 条")

        total_count += count
        results.append((category_name, count))

    # 同时输出一份汇总文件
    all_output = str(output_dir / "all_cleaned.jsonl")
    with open(all_output, "w", encoding="utf-8") as out_f:
        for cat_dir in category_dirs:
            cat_file = output_dir / f"{cat_dir.name}.jsonl"
            if cat_file.exists():
                with open(cat_file, encoding="utf-8") as f:
                    for line in f:
                        out_f.write(line)

    print(f"\n清洗完成！")
    print(f"共处理 {len(category_dirs)} 个分类，{total_count} 条数据")
    print(f"输出目录: {output_dir}")

    # 打印统计
    print("\n各分类统计:")
    for name, count in results:
        if count > 0:
            print(f"  {name}: {count} 条")


if __name__ == "__main__":
    main()
