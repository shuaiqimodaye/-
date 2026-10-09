"""
风格配对数据生成脚本
功能：
1. 从 cleaned 数据中采样原始文本
2. 使用本地 Qwen2.5 模型批量生成 4 种风格的改写结果
3. 质量过滤后输出为 style_paired.jsonl
"""
import os
import json
import random
import argparse
from pathlib import Path

# ---- 风格定义 ----
STYLES = {
    "formal": {
        "name": "正式",
        "description": "改写得更正式，适合学术或商务场景",
        "prompt_template": (
            "把下面这句改写得更正式，适合学术或商务场景。"
            "要求：严格保持原意，禁止添加原文没有的事实、数据、评价或结论。"
            "只改变表达方式和用词风格，不改变信息量。"
            "输出只包含改写后的句子，不要解释。\n\n"
            "{text}"
        )
    },
    "casual": {
        "name": "口语化",
        "description": "改写得更口语化，用日常说话的方式表达",
        "prompt_template": (
            "把下面这句改写得更口语化，用日常说话的方式表达。"
            "要求：严格保持原意，禁止添加原文没有的事实、数据、评价或结论。"
            "只改变表达方式和用词风格，不改变信息量。"
            "输出只包含改写后的句子，不要解释。\n\n"
            "{text}"
        )
    },
    "concise": {
        "name": "简洁",
        "description": "改写得简洁精炼，去掉冗余修饰",
        "prompt_template": (
            "把下面这句改写得简洁精炼，去掉冗余修饰。"
            "要求：严格保持原意，禁止添加原文没有的事实、数据、评价或结论。"
            "只改变表达方式和用词风格，不改变信息量。"
            "输出只包含改写后的句子，不要解释。\n\n"
            "{text}"
        )
    },
    "elaborate": {
        "name": "丰富",
        "description": "用更丰富的措辞重述",
        "prompt_template": (
            "把下面这句用更丰富的措辞重述，但只改写表达方式。"
            "要求：严格保持原意，禁止添加原文没有的事实、数据、评价或结论。"
            "只改变表达方式和用词风格，不改变信息量。"
            "输出只包含改写后的句子，不要解释。\n\n"
            "{text}"
        )
    }
}


def load_cleaned_data(cleaned_dir: str) -> list[dict]:
    """加载所有 cleaned 数据"""
    all_records = []
    cleaned_path = Path(cleaned_dir)

    for jsonl_file in sorted(cleaned_path.glob("*.jsonl")):
        if jsonl_file.name == "all_cleaned.jsonl":
            continue  # 跳过汇总文件
        with open(jsonl_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    record = json.loads(line)
                    all_records.append(record)

    return all_records


def quality_filter(original: str, generated: str) -> bool:
    """质量过滤：检查生成结果是否合理"""
    if not generated or not generated.strip():
        return False

    generated = generated.strip()

    # 长度比检查（生成文本不应太短或太长）
    len_ratio = len(generated) / max(len(original), 1)
    if len_ratio < 0.3 or len_ratio > 3.0:
        return False

    # 不能和原文完全相同（至少要有风格变化）
    if generated == original:
        return False

    # 不能包含拒绝生成的标志词
    reject_patterns = ["无法", "抱歉", "作为AI", "我不能", "sorry", "I cannot"]
    for pattern in reject_patterns:
        if pattern in generated:
            return False

    return True


def generate_style_rewrites(model, tokenizer, texts: list[str],
                           batch_size: int = 8) -> list[dict]:
    """使用模型批量生成 4 种风格的改写"""
    import torch

    results = []
    total = len(texts)

    done = 0
    total_inferences = total * len(STYLES)

    for i in range(0, total, batch_size):
        batch_texts = texts[i:i + batch_size]

        for style_key, style_info in STYLES.items():
            # 单条处理：显存有限时更安全，也避免 batch 内 padding 问题
            for text in batch_texts:
                prompt = style_info["prompt_template"].format(text=text)
                chat_input = [{"role": "user", "content": prompt}]

                encoded = tokenizer.apply_chat_template(
                    chat_input,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_tensors="pt",
                    truncation=True,
                    max_length=1024,
                )
                input_ids = encoded["input_ids"].to(model.device)
                attention_mask = encoded.get("attention_mask")
                if attention_mask is not None:
                    attention_mask = attention_mask.to(model.device)

                with torch.no_grad():
                    output = model.generate(
                        input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=256,
                        temperature=0.8,
                        top_p=0.9,
                        do_sample=True,
                        pad_token_id=tokenizer.eos_token_id
                    )[0]

                input_len = input_ids.shape[1]
                generated_ids = output[input_len:]
                generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

                done += 1
                if done % 10 == 0 or done == total_inferences:
                    print(f"  进度: {done}/{total_inferences}", flush=True)

                results.append({
                    "original": text,
                    "style": style_key,
                    "style_name": style_info["name"],
                    "generated": generated_text,
                    "pass_filter": quality_filter(text, generated_text)
                })

    return results


def generate_with_api(texts: list[str], style_key: str,
                      api_url: str = "http://localhost:11434/api/generate",
                      model_name: str = "qwen2.5") -> list[str]:
    """备选调用 Ollama API 生成（如果本地 GPU 不够）"""
    import requests

    style_info = STYLES[style_key]
    results = []

    for i, text in enumerate(texts):
        if (i + 1) % 50 == 0:
            print(f"  [{style_info['name']}] 生成进度: {i + 1}/{len(texts)}")

        prompt = style_info["prompt_template"].format(text=text)

        try:
            resp = requests.post(api_url, json={
                "model": model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.8,
                    "top_p": 0.9,
                    "num_predict": 256
                }
            }, timeout=60)
            resp.raise_for_status()
            generated = resp.json().get("response", "").strip()
        except Exception as e:
            print(f"  API 错误: {e}")
            generated = ""

        results.append(generated)

    return results


def main():
    parser = argparse.ArgumentParser(description="生成风格配对训练数据")
    parser.add_argument("--cleaned-dir", type=str,
                        default=r"E:\毕业论文\data\cleaned",
                        help="清洗后的数据目录")
    parser.add_argument("--output", type=str,
                        default=r"E:\毕业论文\data\paired\style_paired.jsonl",
                        help="输出文件路径")
    parser.add_argument("--model-path", type=str,
                        default=r"E:\毕业论文\models\Qwen2.5-1.5B-Instruct",
                        help="模型路径")
    parser.add_argument("--samples-per-style", type=int, default=1000,
                        help="每种风格生成的样本数")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="批处理大小")
    parser.add_argument("--use-ollama", action="store_true",
                        help="使用 Ollama API 代替本地模型")
    parser.add_argument("--ollama-url", type=str,
                        default="http://localhost:11434/api/generate",
                        help="Ollama API 地址")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子")
    args = parser.parse_args()

    random.seed(args.seed)

    # 1. 加载清洗后的数据
    print(f"加载清洗数据: {args.cleaned_dir}")
    all_records = load_cleaned_data(args.cleaned_dir)
    print(f"共加载 {len(all_records)} 条数据")

    if not all_records:
        print("错误：没有找到清洗后的数据，请先运行 clean_thucnews.py")
        return

    # 2. 采样
    sample_size = min(args.samples_per_style, len(all_records))
    sampled = random.sample(all_records, sample_size)
    texts = [r["text"] for r in sampled]
    print(f"采样 {sample_size} 条文本用于风格转换")

    # 3. 生成风格改写
    all_results = []

    if args.use_ollama:
        print("使用 Ollama API 生成...")
        for style_key in STYLES:
            print(f"\n生成风格: {STYLES[style_key]['name']}")
            generated = generate_with_api(
                texts, style_key,
                api_url=args.ollama_url
            )
            for text, gen in zip(texts, generated):
                if quality_filter(text, gen):
                    style_info = STYLES[style_key]
                    all_results.append({
                        "instruction": style_info["prompt_template"].format(text="").split("\n\n")[0],
                        "input": text,
                        "output": gen,
                        "style": style_key,
                        "style_name": style_info["name"]
                    })
    else:
        print(f"加载模型: {args.model_path}")
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch

            tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
            # 强制 CPU 模式，避免显存不足导致参数卸载到 CPU 而卡住
            model = AutoModelForCausalLM.from_pretrained(
                args.model_path,
                torch_dtype=torch.float32,
                device_map="cpu",
                trust_remote_code=True
            )

            print("开始批量生成...")
            raw_results = generate_style_rewrites(
                model, tokenizer, texts,
                batch_size=args.batch_size
            )

            for r in raw_results:
                if r["pass_filter"]:
                    style_info = STYLES[r["style"]]
                    all_results.append({
                        "instruction": style_info["prompt_template"].format(text="").split("\n\n")[0],
                        "input": r["original"],
                        "output": r["generated"],
                        "style": r["style"],
                        "style_name": r["style_name"]
                    })

        except Exception as e:
            print(f"模型加载失败: {e}")
            print("请检查模型路径和 GPU 显存，或使用 --use-ollama 参数")
            return

    # 4. 输出
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # 打乱顺序
    random.shuffle(all_results)

    with open(args.output, "w", encoding="utf-8") as f:
        for record in all_results:
            # 输出训练格式（instruction/input/output）
            train_record = {
                "instruction": record["instruction"],
                "input": record["input"],
                "output": record["output"]
            }
            f.write(json.dumps(train_record, ensure_ascii=False) + "\n")

    # 统计
    style_counts = {}
    for r in all_results:
        style = r.get("style_name", r.get("style", "unknown"))
        style_counts[style] = style_counts.get(style, 0) + 1

    print(f"\n生成完成！")
    print(f"共生成 {len(all_results)} 条配对数据")
    print(f"输出文件: {args.output}")
    print("\n各风格统计:")
    for style, count in sorted(style_counts.items()):
        print(f"  {style}: {count} 条")


if __name__ == "__main__":
    main()
