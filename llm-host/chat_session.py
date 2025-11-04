from llm_tools import LLMTools

class ChatSession:
    def __init__(self, tools: LLMTools, model_name:str):
        self.tools: LLMTools = LLMTools
        self.model_name: str = model_name
    
    def chat_with_model(self, question:str) -> str:
        return self.tools.chat(self.model_name, question)
