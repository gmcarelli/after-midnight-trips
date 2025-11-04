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
    Cliente concreto para interagir com o host Ollama.
    Implementa as interfaces HostConnector e LLMTools.
    """

    _client: ollama.Client

    def connect_to_host(self, host_url: str) -> "OllamaClient":
        """
        Conecta-se ao host Ollama e inicializa o cliente.
        Retorna a própria instância do cliente para encadeamento de chamadas.
        """
        try:
            self._client = ollama.Client(host=host_url)
            # A biblioteca Ollama não valida a conexão no __init__,
            # então fazemos uma chamada leve para forçar a verificação.
            self._client.list()
            log.info(f"Cliente Ollama conectado com sucesso ao host: {host_url}")
            return self
        except Exception as e:
            error_message = f"Não foi possível conectar ao servidor Ollama em {host_url}. Detalhes: {e}"
            log.error(error_message)
            raise OllamaConnectionError(error_message) from e

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
