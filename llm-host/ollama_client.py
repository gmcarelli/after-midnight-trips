import ollama
from typing import List, Dict, Any

from llm_host.host_connector import HostConnector
from llm_host.llm_tools import LLMTools
from logger_config import log

class OllamaConnectionError(Exception):
    """Exceção personalizada para erros de conexão com o host Ollama."""
    pass

class OllamaClient(HostConnector, LLMTools):
    """
    Cliente unificado para interagir com o host Ollama.

    Esta classe gerencia a conexão e fornece métodos para todas as operações
    relacionadas ao Ollama, seguindo uma abordagem de fachada simples.
    """
    def __init__(self, host_url: str):
        self._host_url = host_url
        self._client: ollama.Client = self._connect_to_host()

    def _connect_to_host(self) -> ollama.Client:
        """
        Método privado para estabelecer a conexão durante a inicialização.
        """
        try:
            client = ollama.Client(host=self._host_url)
            # Força a verificação da conexão com uma chamada leve.
            client.list()
            log.info(f"Conexão com o host Ollama em {self._host_url} bem-sucedida.")
            return client
        except Exception as e:
            error_message = f"Não foi possível conectar ao servidor Ollama em {self._host_url}. Detalhes: {e}"
            log.error(error_message)
            raise OllamaConnectionError(error_message) from e

    def create_model(self, base_model: str, model_name: str, system_role: str) -> bool:
        """
        Cria um modelo personalizado no Ollama.
        """
        try:
            modelfile = f"FROM {base_model}\nSYSTEM {system_role}"
            self._client.create(model=model_name, modelfile=modelfile)
            log.info(f"Modelo '{model_name}' criado com sucesso a partir de '{base_model}'.")
            return True
        except Exception as e:
            log.error(f"Erro ao criar o modelo '{model_name}': {e}")
            return False

    def list_models(self) -> List[Dict[str, Any]]:
        """
        Lista os modelos disponíveis no host Ollama.
        """
        try:
            return self._client.list().get("models", [])
        except Exception as e:
            log.error(f"Erro ao listar os modelos: {e}")
            return []

    def chat(self, model_name: str, message: str) -> str:
        """
        Envia uma mensagem para o modelo de chat e retorna a resposta.
        """
        try:
            response = self._client.chat(model=model_name, messages=[{"role": "user", "content": message}])
            return response.get("message", {}).get("content", "")
        except Exception as e:
            log.error(f"Erro durante o chat com o modelo '{model_name}': {e}")
            return ""
