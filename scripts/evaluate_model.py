"""
模型评估脚本
评估指标：BLEU、ROUGE、BERTScore（语义相似度）
"""
import torch
import json
import argparse
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from datasets import load_dataset

# 评估指标
from sacrebleu import corpus_bleu
from rouge_score import rouge_scorer
from bert_score import score as bert_score_fn


def load_model(model_path, lora_path=None):
    """加载模型（基座或 LoRA）"""
    print(f"加载 tokenizer: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    print(f"加载模型: {model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        torch_dtype=torch.float16, 
        device_map="auto"
    )
    
    if lora_path:
        print(f"加载 LoRA 权重: {lora_path}")
        model.resize_token_embeddings(len(tokenizer))
        model = PeftModel.from_pretrained(model, lora_path)
    
    model.eval()
    return tokenizer, model


def generate_text(tokenizer, model, prompt, max_new_tokens=150):
    """生成文本"""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.8,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    generated = tokenizer.decode(
        outputs[0][inputs['input_ids'].shape[1]:], 
        skip_special_tokens=True
    )
    return generated.strip()


def evaluate_model(model_path, lora_path, data_file, output_file, num_samples=100):
    """评估模型"""
    # 加载模型
    tokenizer, model = load_model(model_path, lora_path)
    
    # 加载测试数据
    dataset = load_dataset("json", data_files=data_file)
    test_data = dataset["train"].shuffle(seed=42).select(range(min(num_samples, len(dataset["train"]))))
    
    print(f"\n评估样本数: {len(test_data)}")
    print("开始生成...")
    
    # 生成结果
    results = []
    references = []
    hypotheses = []
    
    for item in tqdm(test_data, desc="生成"):
        prompt = f"{item['instruction']}\n\n{item['input']}"
        generated = generate_text(tokenizer, model, prompt)
        
        results.append({
            "input": item['input'],
            "instruction": item['instruction'],
            "reference": item['output'],
            "generated": generated,
            "style": item.get('style', 'unknown')
        })
        
        references.append(item['output'])
        hypotheses.append(generated)
    
    # 计算指标
    print("\n计算评估指标...")
    
    # BLEU
    bleu_score = corpus_bleu(hypotheses, [references]).score
    print(f"BLEU: {bleu_score:.2f}")
    
    # ROUGE
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=False)
    rouge_scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}
    
    for ref, hyp in zip(references, hypotheses):
        scores = scorer.score(ref, hyp)
        for key in rouge_scores:
            rouge_scores[key].append(scores[key].fmeasure)
    
    rouge_avg = {k: sum(v) / len(v) * 100 for k, v in rouge_scores.items()}
    print(f"ROUGE-1: {rouge_avg['rouge1']:.2f}")
    print(f"ROUGE-2: {rouge_avg['rouge2']:.2f}")
    print(f"ROUGE-L: {rouge_avg['rougeL']:.2f}")
    
    # BERTScore
    print("计算 BERTScore（需要几分钟）...")
    P, R, F1 = bert_score_fn(hypotheses, references, lang='zh', verbose=False)
    bertscore_avg = F1.mean().item() * 100
    print(f"BERTScore: {bertscore_avg:.2f}")
    
    # 保存结果
    output_data = {
        "metrics": {
            "bleu": bleu_score,
            "rouge1": rouge_avg['rouge1'],
            "rouge2": rouge_avg['rouge2'],
            "rougeL": rouge_avg['rougeL'],
            "bertscore": bertscore_avg
        },
        "results": results
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n结果已保存到: {output_file}")
    
    # 打印示例
    print("\n" + "="*60)
    print("示例输出（前 5 个）")
    print("="*60)
    for i, r in enumerate(results[:5], 1):
        print(f"\n【示例 {i}】")
        print(f"输入: {r['input'][:100]}...")
        print(f"参考: {r['reference'][:100]}...")
        print(f"生成: {r['generated'][:100]}...")
    
    return output_data['metrics']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="models/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--lora-path", type=str, default="outputs/lora_checkpoint/final_lora_weights")
    parser.add_argument("--data-file", type=str, default="data/paired/style_paired.jsonl")
    parser.add_argument("--output-file", type=str, default="outputs/evaluation_results.json")
    parser.add_argument("--num-samples", type=int, default=100)
    args = parser.parse_args()
    
    print("="*60)
    print("模型评估")
    print("="*60)
    
    metrics = evaluate_model(
        args.model_path,
        args.lora_path,
        args.data_file,
        args.output_file,
        args.num_samples
    )
    
    print("\n" + "="*60)
    print("评估完成")
    print("="*60)
    print(f"BLEU: {metrics['bleu']:.2f}")
    print(f"ROUGE-1: {metrics['rouge1']:.2f}")
    print(f"ROUGE-2: {metrics['rouge2']:.2f}")
    print(f"ROUGE-L: {metrics['rougeL']:.2f}")
    print(f"BERTScore: {metrics['bertscore']:.2f}")


if __name__ == "__main__":
    main()
