"""
解压 THUCNews.zip 数据集
处理 GBK 编码的中文文件名，解压到 data/raw/ 目录
"""
import zipfile
import os
import sys

def extract_thucnews(zip_path, extract_dir):
    """解压 THUCNews 数据集，正确处理中文文件名编码"""
    
    os.makedirs(extract_dir, exist_ok=True)
    
    print(f"正在打开: {zip_path}")
    print(f"解压到: {extract_dir}")
    
    with zipfile.ZipFile(zip_path, 'r') as z:
        names = z.namelist()
        print(f"压缩包含 {len(names)} 个条目")
        
        # 统计分类
        categories = set()
        for name in names:
            if name.startswith('THUCNews/'):
                parts = name.split('/')
                if len(parts) >= 3 and parts[1]:
                    # 尝试解码文件名
                    try:
                        decoded = parts[1].encode('cp437').decode('gbk')
                        categories.add(decoded)
                    except (UnicodeDecodeError, UnicodeEncodeError):
                        categories.add(parts[1])
        
        print(f"发现 {len(categories)} 个分类: {sorted(categories)}")
        
        # 解压文件
        extracted = 0
        errors = 0
        
        for info in z.infolist():
            if info.filename.endswith('/'):
                # 是目录，跳过
                continue
            
            try:
                # 尝试解码文件名
                try:
                    # THUCNews zip 使用 GBK 编码
                    original_name = info.filename.encode('cp437').decode('gbk')
                except (UnicodeDecodeError, UnicodeEncodeError):
                    original_name = info.filename
                
                # 构建目标路径
                target_path = os.path.join(extract_dir, original_name)
                
                # 创建父目录
                os.makedirs(os.path.dirname(target_path), exist_ok=True)
                
                # 读取并写入文件
                data = z.read(info.filename)
                with open(target_path, 'wb') as f:
                    f.write(data)
                
                extracted += 1
                if extracted % 10000 == 0:
                    print(f"已解压 {extracted} 个文件...")
                    
            except Exception as e:
                errors += 1
                if errors <= 10:
                    print(f"错误: {info.filename} - {e}")
        
        print(f"\n解压完成!")
        print(f"成功解压: {extracted} 个文件")
        print(f"失败: {errors} 个文件")
        
        # 验证解压结果
        print("\n验证解压结果:")
        for cat_dir in sorted(os.listdir(os.path.join(extract_dir, 'THUCNews'))):
            cat_path = os.path.join(extract_dir, 'THUCNews', cat_dir)
            if os.path.isdir(cat_path):
                file_count = len([f for f in os.listdir(cat_path) if os.path.isfile(os.path.join(cat_path, f))])
                print(f"  {cat_dir}: {file_count} 个文件")

if __name__ == "__main__":
    zip_path = r"E:\毕业论文\THUCNews.zip"
    extract_dir = r"E:\毕业论文\data\raw"
    
    extract_thucnews(zip_path, extract_dir)
