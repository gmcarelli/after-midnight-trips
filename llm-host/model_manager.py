from typing import List
from llm_tools import LLMTools

class ModelManager:
    def __init__(self, tools: LLMTools):
        self.tools: LLMTools = tools
        
    def create_custom_model(self, base_model: str, model_name: str, system_role: str) -> bool:
        return self.tools.create_model(base_model, model_name, system_role)

    def list_models(self) -> List:
        return self.tools.list_models()
