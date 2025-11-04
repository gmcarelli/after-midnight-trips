from typing import Protocol, TypeVar, Generic
import ollama
from logger_config import log

TClient = TypeVar("TClient", covariant=True)


class HostConnector(Protocol, Generic[TClient]):
    def connect_to_host(self, host_url) -> TClient: ...


class OllamaClient(HostConnector):
            
    def connect_to_host(self, host_url: str) -> ollama.Client:
        try:
            client: ollama.Client = ollama.Client(host=host_url)
            log.info("Cliente Ollama inicializado e conexão com o host bem-sucedida.")
            return client

        except Exception as e:
            log.error(f"Não foi possível conectar ao servidor Ollama.")
            log.error(f"Detalhes do erro: {e}")
            raise
            

