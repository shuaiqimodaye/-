"""
风格约束生成模块
在语义骨架约束下，注入目标风格生成最终文本
"""
import torch
from typing import Optional
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


# 支持的 4 种风格及其提示词模板
STYLE_TEMPLATES = {
    "正式": {
        "name": "正式",
        "description": "书面化、正式的表达方式",
        "prompt": "将以下文本转换为正式、书面化的风格：\n\n{text}"
    },
    "口语化": {
        "name": "口语化",
        "description": "日常对话、轻松自然的风格",
        "prompt": "将以下文本转换为口语化、日常对话的风格：\n\n{text}"
    },
    "简洁": {
        "name": "简洁",
        "description": "精炼、去掉冗余修饰",
        "prompt": "将以下文本转换为简洁精炼的风格：\n\n{text}"
    },
    "丰富": {
        "name": "丰富",
        "description": "扩展细节、增加修饰",
        "prompt": "将以下文本转换为丰富、详细的风格：\n\n{text}"
    },
}


class StyleGenerator:
    """风格约束生成器 - 基于 LoRA 微调的 Qwen2.5 模型"""
    
    def __init__(self, base_model_path: str, lora_weights_path: str, device: str = "auto"):
        self.device = device
        self.tokenizer = None
        self.model = None
        self._load_model(base_model_path, lora_weights_path)
    
    def _load_model(self, base_model_path: str, lora_weights_path: str):
        """加载基座模型 + LoRA 权重"""
        print("加载 tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(lora_weights_path)
        
        print("加载基座模型...")
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.float16,
            device_map=self.device
        )
        self.model.resize_token_embeddings(len(self.tokenizer))
        
        print("加载 LoRA 权重...")
        self.model = PeftModel.from_pretrained(self.model, lora_weights_path)
        self.model.eval()
        print("模型加载完成！")
    
    def generate(
        self,
        text: str,
        style: str,
        max_new_tokens: int = 150,
        temperature: float = 0.8,
        top_p: float = 0.9
    ) -> str:
        """
        风格转换生成
        
        Args:
            text: 输入文本
            style: 目标风格（正式/口语化/简洁/丰富）
            max_new_tokens: 最大生成长度
            temperature: 采样温度
            top_p: nucleus sampling 参数
            
        Returns:
            风格转换后的文本
        """
        if style not in STYLE_TEMPLATES:
            raise ValueError(f"不支持的风格: {style}，可选: {list(STYLE_TEMPLATES.keys())}")
        
        prompt = STYLE_TEMPLATES[style]["prompt"].format(text=text)
        
        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=512
        ).to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        return self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        ).strip()
    
    @staticmethod
    def list_styles() -> dict:
        """返回支持的风格列表"""
        return {k: v["description"] for k, v in STYLE_TEMPLATES.items()}
