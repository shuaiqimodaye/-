"""
LoRA 微调脚本
使用风格配对数据微调 Qwen2.5-1.5B-Instruct，学习风格转换能力
"""
import os
import json
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset
from dataclasses import dataclass
from typing import Optional
import argparse


# LoRA 配置
@dataclass
class LoRAArguments:
    r: int = 8                      # LoRA 秩
    lora_alpha: int = 16            # LoRA alpha 缩放因子
    lora_dropout: float = 0.05      # dropout 率
    target_modules: list = None     # 目标模块
    
    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]


# 训练配置
@dataclass
class TrainingConfig:
    output_dir: str = "outputs/lora_checkpoint"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    max_seq_length: int = 512
    fp16: bool = True
    logging_steps: int = 10
    save_steps: int = 100
    save_total_limit: int = 2
    warmup_ratio: float = 0.03
    weight_decay: float = 0.01
    report_to: str = "none"


def build_prompt(instruction: str, input_text: str) -> str:
    """构建训练 prompt"""
    return f"{instruction}\n\n{input_text}"


def tokenize_function(examples, tokenizer, max_length=512):
    """将数据转换为模型输入格式"""
    prompts = [
        build_prompt(inst, inp) 
        for inst, inp in zip(examples['instruction'], examples['input'])
    ]
    outputs = examples['output']
    
    # 构造完整的训练样本：prompt + 输出
    full_texts = [p + " " + o for p, o in zip(prompts, outputs)]
    
    # tokenize
    tokenized = tokenizer(
        full_texts,
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors=None
    )
    
    # 构造 labels（用于计算 loss）
    tokenized["labels"] = tokenized["input_ids"].copy()
    
    # 将 prompt 部分的 labels 设为 -100（不计算 loss）
    for i, prompt in enumerate(prompts):
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        prompt_len = len(prompt_ids)
        tokenized["labels"][i][:prompt_len] = [-100] * prompt_len
    
    return tokenized


def main():
    parser = argparse.ArgumentParser(description="LoRA 微调风格转换模型")
    parser.add_argument("--model-path", type=str, 
                        default="models/Qwen2.5-1.5B-Instruct",
                        help="基座模型路径")
    parser.add_argument("--data-file", type=str,
                        default="data/paired/style_paired.jsonl",
                        help="训练数据文件")
    parser.add_argument("--output-dir", type=str,
                        default="outputs/lora_checkpoint",
                        help="输出目录")
    parser.add_argument("--epochs", type=int, default=3,
                        help="训练轮数")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="batch size")
    parser.add_argument("--lr", type=float, default=2e-4,
                        help="学习率")
    parser.add_argument("--lora-r", type=int, default=8,
                        help="LoRA 秩")
    parser.add_argument("--max-length", type=int, default=512,
                        help="最大序列长度")
    args = parser.parse_args()
    
    print("=" * 70)
    print("LoRA 微调 - 风格转换模型")
    print("=" * 70)
    
    # 1. 加载 tokenizer
    print("\n[1/6] 加载 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # 2. 加载基座模型
    print(f"[2/6] 加载基座模型: {args.model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    print(f"      模型参数量: {model.num_parameters() / 1e9:.2f}B")
    
    # 3. 应用 LoRA
    print(f"[3/6] 应用 LoRA 配置 (r={args.lora_r})...")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        bias="none"
    )
    model = get_peft_model(model, lora_config)
    
    # 打印可训练参数
    trainable_params, total_params = model.get_nb_trainable_parameters()
    print(f"      可训练参数: {trainable_params:,} / {total_params:,} "
          f"({trainable_params/total_params*100:.2f}%)")
    
    # 4. 加载数据集
    print(f"[4/6] 加载数据集: {args.data_file}")
    dataset = load_dataset("json", data_files=args.data_file)
    print(f"      训练样本数: {len(dataset['train'])}")
    
    # 5. 数据预处理
    print(f"[5/6] 数据预处理 (max_length={args.max_length})...")
    tokenized_datasets = dataset.map(
        lambda examples: tokenize_function(examples, tokenizer, args.max_length),
        batched=True,
        remove_columns=dataset["train"].column_names
    )
    
    # 6. 训练配置
    print("[6/6] 配置训练参数...")
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        learning_rate=args.lr,
        fp16=torch.cuda.is_available(),
        logging_steps=10,
        save_steps=100,
        save_total_limit=2,
        warmup_ratio=0.03,
        weight_decay=0.01,
        report_to="none",
        remove_unused_columns=False,
        optim="adamw_torch",
        dataloader_pin_memory=True,
        dataloader_num_workers=2,
    )
    
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        padding=True,
        pad_to_multiple_of=8
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        data_collator=data_collator
    )
    
    # 开始训练
    print("\n" + "=" * 70)
    print("开始训练...")
    print("=" * 70 + "\n")
    
    trainer.train()
    
    # 保存 LoRA 权重
    lora_output = os.path.join(args.output_dir, "final_lora_weights")
    print(f"\n保存 LoRA 权重到: {lora_output}")
    model.save_pretrained(lora_output)
    tokenizer.save_pretrained(lora_output)
    
    # 保存训练配置
    config_output = os.path.join(args.output_dir, "training_config.json")
    with open(config_output, "w") as f:
        json.dump({
            "model_path": args.model_path,
            "data_file": args.data_file,
            "lora_r": args.lora_r,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "train_samples": len(dataset["train"])
        }, f, indent=2)
    
    print("\n" + "=" * 70)
    print("训练完成！")
    print(f"LoRA 权重: {lora_output}")
    print(f"训练配置: {config_output}")
    print("=" * 70)


if __name__ == "__main__":
    main()
