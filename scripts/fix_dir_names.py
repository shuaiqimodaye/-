"""
修复 THUCNews 解压后的重复目录和乱码目录名问题。
1. 删除 10 个与正确中文名重复的乱码目录
2. 将剩余 3 个只有乱码名的目录重命名为正确中文名
"""
import os
import shutil

base = r"E:\毕业论文\data\raw\THUCNews"

# 乱码名 → 正确中文名的映射（已确认文件数完全一致）
garbled_to_correct = {
    "浣撹偛": "体育",
    "濞变箰": "娱乐",
    "瀹跺眳": "家居",
    "鏁欒偛": "教育",
    "鏃跺皻": "时尚",
    "鏃舵斂": "时政",
    "鏄熷骇": "星座",
    "娓告垙": "游戏",
    "绀句細": "社会",
    "绉戞妧": "科技",
}

# 只有乱码名的目录（需要重命名）
rename_only = {
    "褰╃エ": "彩票",
    "鎴夸骇": "房产",
    "鑲＄エ": "股票",
}

# Step 1: 删除重复的乱码目录
print("=== 删除重复的乱码目录 ===")
for garbled, correct in garbled_to_correct.items():
    garbled_path = os.path.join(base, garbled)
    correct_path = os.path.join(base, correct)
    if os.path.isdir(garbled_path) and os.path.isdir(correct_path):
        g_count = len(os.listdir(garbled_path))
        c_count = len(os.listdir(correct_path))
        print(f"  删除 {garbled} ({g_count} 文件) -> 保留 {correct} ({c_count} 文件)")
        shutil.rmtree(garbled_path)
    else:
        print(f"  跳过 {garbled} (目录不存在)")

# Step 2: 重命名剩余乱码目录
print("\n=== 重命名乱码目录 ===")
for garbled, correct in rename_only.items():
    garbled_path = os.path.join(base, garbled)
    correct_path = os.path.join(base, correct)
    if os.path.isdir(garbled_path):
        if os.path.exists(correct_path):
            print(f"  跳过 {garbled} -> {correct}: 目标已存在")
        else:
            count = len(os.listdir(garbled_path))
            print(f"  重命名 {garbled} -> {correct} ({count} 文件)")
            os.rename(garbled_path, correct_path)
    else:
        print(f"  跳过 {garbled} (目录不存在)")

# Step 3: 验证结果
print("\n=== 最终目录列表 ===")
dirs = sorted([d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d))])
print(f"共 {len(dirs)} 个分类:")
for d in dirs:
    count = len(os.listdir(os.path.join(base, d)))
    print(f"  {d}: {count} 个文件")
