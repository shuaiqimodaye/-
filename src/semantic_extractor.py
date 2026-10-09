"""
语义抽取模块
从输入文本中提取核心语义信息（实体、事件、关系）
"""
import re
from typing import List, Dict, Any


class SemanticExtractor:
    """
    语义抽取器 - 从文本中提取关键语义信息
    
    基于规则和统计方法提取：
    - 关键实体（人名、地名、时间）
    - 核心事件（动词短语）
    - 逻辑关系（因果、转折）
    """
    
    def __init__(self):
        self.entity_patterns = {
            'person': r'[\u4e00-\u9fa5]{2,4}(?:先生|女士|老师|教授|博士)',
            'place': r'[\u4e00-\u9fa5]{2,8}(?:市|区|县|镇|村|路|街)',
            'time': r'\d{4}[年/-]\d{1,2}[月/-]\d{1,2}[日号]?|\d{1,2}[月/-]\d{1,2}[日号]?'
        }
    
    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """提取文本中的实体"""
        entities = {}
        for entity_type, pattern in self.entity_patterns.items():
            matches = re.findall(pattern, text)
            if matches:
                entities[entity_type] = matches
        return entities
    
    def extract_core_event(self, text: str) -> str:
        """提取核心事件（简化版）"""
        # 这里可以实现更复杂的依存句法分析
        # 目前使用简单的句子分割
        sentences = re.split(r'[。！？!?]', text)
        return sentences[0] if sentences else text
    
    def extract(self, text: str) -> Dict[str, Any]:
        """
        完整的语义抽取
        
        Args:
            text: 输入文本
            
        Returns:
            包含实体、核心事件、原文的字典
        """
        return {
            'original_text': text,
            'entities': self.extract_entities(text),
            'core_event': self.extract_core_event(text),
            'text_length': len(text)
        }


if __name__ == "__main__":
    # 测试
    extractor = SemanticExtractor()
    test_text = "张三教授在北京大学发表了一篇重要论文。"
    result = extractor.extract(test_text)
    print(result)
