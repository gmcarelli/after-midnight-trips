import ollama
from llm_host.host_connector import HostConnector
from logger_config import log

class OllamaConnectionError(Exception):
    """Exceção personalizada para erros de conexão com o host Ollama."""
    pass

class OllamaConnector(HostConnector):
    """
    Implementação concreta do HostConnector para se conectar a um host Ollama.
    """
    def connect_to_host(self, host_url: str) -> ollama.Client:
        """
        Conecta-se ao host Ollama e retorna um cliente bruto da biblioteca.
        """
        try:
            client = ollama.Client(host=host_url)
            # A biblioteca Ollama não valida a conexão no __init__,
            # então fazemos uma chamada leve para forçar a verificação.
            client.list()
            log.info(f"Conexão com o host Ollama em {host_url} bem-sucedida.")
            return client
        except Exception as e:
            error_message = f"Não foi possível conectar ao servidor Ollama em {host_url}. Detalhes: {e}"
            log.error(error_message)
            raise OllamaConnectionError(error_message) from e
