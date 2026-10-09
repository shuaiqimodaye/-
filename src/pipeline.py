"""
风格转换管线
整合语义抽取和风格生成，提供完整的风格转换流程
"""
from typing import Optional
from .semantic_extractor import SemanticExtractor
from .style_generator import StyleGenerator


class StyleTransferPipeline:
    """
    风格转换管线
    
    流程：输入文本 → 语义抽取 → 风格约束生成 → 输出文本
    """
    
    def __init__(
        self,
        base_model_path: str,
        lora_weights_path: str,
        device: str = "auto"
    ):
        self.semantic_extractor = SemanticExtractor()
        self.style_generator = StyleGenerator(
            base_model_path=base_model_path,
            lora_weights_path=lora_weights_path,
            device=device
        )
    
    def transfer(
        self,
        text: str,
        style: str,
        use_semantic_extraction: bool = True,
        max_new_tokens: int = 150,
        temperature: float = 0.8
    ) -> dict:
        """
        执行风格转换
        
        Args:
            text: 输入文本
            style: 目标风格（正式/口语化/简洁/丰富）
            use_semantic_extraction: 是否使用语义抽取
            max_new_tokens: 最大生成长度
            temperature: 采样温度
            
        Returns:
            包含原文、语义信息、生成结果的字典
        """
        result = {
            'original_text': text,
            'target_style': style,
        }
        
        # 语义抽取
        if use_semantic_extraction:
            semantic_info = self.semantic_extractor.extract(text)
            result['semantic_info'] = semantic_info
        
        # 风格生成
        generated_text = self.style_generator.generate(
            text=text,
            style=style,
            max_new_tokens=max_new_tokens,
            temperature=temperature
        )
        result['generated_text'] = generated_text
        
        return result
    
    def batch_transfer(
        self,
        texts: list,
        style: str,
        **kwargs
    ) -> list:
        """批量风格转换"""
        return [self.transfer(text, style, **kwargs) for text in texts]
