import ollama
from typing import List, Dict, Any
from llm_host.llm_tools import LLMTools
from llm_host.ollama_connector import OllamaConnector
from logger_config import log

class OllamaOperations(LLMTools):
    """
    Implementa as operações de negócio (chat, criar modelo, etc.)
    usando um cliente Ollama já conectado.
    """
    def __init__(self, client: ollama.Client):
        self._client = client

    def create_model(self, base_model: str, model_name: str, system_role: str) -> bool:
        """
        Cria um modelo personalizado no Ollama.
        Retorna True se o modelo foi criado com sucesso, False caso contrário.
        """
        try:
            modelfile = f"FROM {base_model}\nSYSTEM {system_role}"
            self._client.create(model=model_name, modelfile=modelfile)
            log.info(
                f"Modelo personalizado '{model_name}' criado com sucesso a partir do modelo base '{base_model}'."
            )
            return True
        except Exception as e:
            log.error(f"Erro ao criar o modelo personalizado '{model_name}': {e}")
            return False

    def list_models(self) -> List[Dict[str, Any]]:
        """
        Lista os modelos disponíveis no host Ollama.
        """
        try:
            models = self._client.list().get("models", [])
            log.info(f"Encontrados {len(models)} modelos no host.")
            return models
        except Exception as e:
            log.error(f"Erro ao listar os modelos: {e}")
            return []

    def chat(self, model_name: str, message: str) -> str:
        """
        Envia uma mensagem para o modelo de chat e retorna a resposta.
        """
        try:
            log.info(f"Enviando prompt para o modelo '{model_name}'...")
            response = self._client.chat(
                model=model_name, messages=[{"role": "user", "content": message}]
            )
            content = response.get("message", {}).get("content", "")
            log.info("Resposta recebida do modelo.")
            return content
        except Exception as e:
            log.error(f"Erro durante o chat com o modelo '{model_name}': {e}")
            return ""


class OllamaClient:
    """
    Fachada para interagir com o host Ollama.
    Gerencia a conexão e fornece um objeto de operações.
    """
    def __init__(self, host_url: str):
        self._host_url = host_url
        self._connector = OllamaConnector()
        self._client = self._connector.connect_to_host(self._host_url)
        self.operations = OllamaOperations(self._client)
