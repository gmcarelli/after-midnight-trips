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
            




class OllamaClient:

    client: ollama.Client | None

    def __init__(self) -> None:
        self.client: ollama.Client 

    def create_custom_model(
        self, model_name: str, base_model: str, system_role: str
    ) -> None:
        try:
            self.client.create(model=model_name, from_=base_model, system=system_role)
            log.info(
                f"Modelo personalizado '{model_name}' criado com sucessso. Modelo base '{base_model}'."
            )
        except Exception as e:
            log.error(f"Erro ao criar o modelo personalizado: {e}")

    def list_models(self) -> list:
        return self.client.list()


def prompt_model(client: ollama.Client, model_name: str, prompt) -> str:
    """
    Asks a question and returns its answer ...duh
    """
    response: str = client.chat(
        model=model_name, messages=[{"role": "user", "content": prompt}]
    )

    return response.message.content


def main() -> None:
    pass


if __name__ == "__main__":
    main()
