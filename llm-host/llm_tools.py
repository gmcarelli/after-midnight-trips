from typing import Protocol, Generic, TypeVar

TClient = TypeVar("TClient", covariant=True)

class LLMTools(Protocol, Generic[TClient]):
    
    def create_model(self, base_model: str, model_name: str) -> bool: ...
    
    def chat(self, model_name: str, message: str) -> str: ...
