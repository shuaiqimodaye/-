# 个性化语言风格转换系统

基于语义重构和 LoRA 微调的 Qwen2.5 模型，实现个性化语言风格转换。

## 功能特性

- **4 种风格转换**：正式、口语化、简洁、丰富
- **语义保持**：使用语义抽取技术保持原文核心信息
- **LoRA 微调**：基于 1.19M 条配对数据微调，参数量仅 2.18M（占基座模型的 0.14%）
- **交互式演示**：提供 Web 界面和命令行两种使用方式

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 下载模型

```bash
python scripts/download_models.py
```

### 3. 运行演示

**Web 界面**（推荐）：
```bash
python app.py
```

**命令行**：
```bash
python app.py --cli
```

## 项目结构

```
style_transfer_project/
├── src/                    # 核心代码
│   ├── __init__.py
│   ├── semantic_extractor.py  # 语义抽取模块
│   ├── style_generator.py     # 风格生成模块
│   └── pipeline.py            # 转换流程管线
├── scripts/                # 工具脚本
│   ├── download_models.py     # 模型下载
│   ├── train_lora.py          # LoRA 训练
│   ├── evaluate_model.py      # 模型评估
│   ├── clean_thucnews.py      # 数据清洗
│   ├── build_paired_data.py   # 构建配对数据
│   └── analyze_data.py        # 数据分析
├── data/                   # 数据文件
│   ├── cleaned/               # 清洗后的新闻数据
│   └── paired/                # 风格配对训练数据
├── app.py                  # 主程序入口
├── requirements.txt        # 依赖列表
└── README.md              # 项目说明
```

## 使用方法

### Python API

```python
from src.pipeline import StyleTransferPipeline

# 初始化
pipeline = StyleTransferPipeline(
    base_model_path="models/Qwen2.5-1.5B-Instruct",
    lora_weights_path="outputs/lora_checkpoint_v2/final_lora_weights"
)

# 风格转换
result = pipeline.transfer(
    text="今天天气不错，我们去公园散步吧。",
    style="正式"
)

print(result['generated_text'])
# 输出：今日天气甚佳，适宜前往公园散步。
```

### 命令行示例

```bash
$ python app.py
请输入要转换的文本: 今天天气不错
选择风格 (1-正式 2-口语化 3-简洁 4-丰富): 1
转换结果: 今日天气甚佳
```

## 训练自己的模型

1. 准备数据：
```bash
python scripts/build_paired_data.py --input data/cleaned --output data/paired
```

2. 训练 LoRA：
```bash
python scripts/train_lora.py \
  --model_path models/Qwen2.5-1.5B-Instruct \
  --data_path data/paired \
  --output_dir outputs/lora_checkpoint_v2
```

## 技术细节

- **基座模型**：Qwen2.5-1.5B-Instruct
- **微调方法**：LoRA (r=8, alpha=16, dropout=0.05)
- **训练数据**：1.19M 条风格配对数据
- **评估指标**：ROUGE-1=16.34, ROUGE-2=5.29, BERTScore=68.26

## 许可证

MIT License

## 作者

基于毕业论文《基于语义重构的个性化语言风格转换系统设计与实现》
