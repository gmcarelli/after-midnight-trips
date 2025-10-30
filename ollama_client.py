import os
import ollama
from dotenv import load_dotenv
from typing import Optional, Mapping, Any, Dict
from logger_config import log

class OllamaClient:
    """
    Uma classe para gerenciar a conexão com o cliente Ollama,
    com funcionalidades para criar modelos personalizados.
    """
    client: Optional[ollama.Client]

    def __init__(self) -> None:
        """
        Inicializa o cliente, carregando as configurações do ambiente e
        tratando possíveis erros de conexão.
        """
        self.client = None
        host: Optional[str] = None
        try:
            load_dotenv()
            host = os.getenv("OLLAMA_HOST")
            if not host:
                raise ValueError("A variável de ambiente OLLAMA_HOST não foi definida.")

            self.client = ollama.Client(host=host)
            self.client.list()
            log.info("Cliente Ollama inicializado e conexão com o host bem-sucedida.")

        except Exception as e:
            self.client = None
            log.error(f"Não foi possível conectar ao host Ollama em '{host}'.")
            log.error(f"Detalhes do erro: {e}")

    def create_custom_model(self, profile_name: str, base_model: str, system_role: str, temperature: float = 0.7) -> None:
        """
        Cria um novo modelo personalizado no host Ollama com base em um perfil.
        """
        if not self.client:
            log.warning("Operação cancelada: O cliente Ollama não está conectado.")
            return

        try:
            parameters: Mapping[str, Any] = {"temperature": temperature}

            self.client.create(
                model=profile_name,
                from_=base_model,
                system=system_role,
                parameters=parameters
            )
            log.info(f"Modelo personalizado '{profile_name}' criado com sucesso a partir de '{base_model}'.")

        except Exception as e:
            log.error(f"Erro ao criar o modelo '{profile_name}': {e}")

def query_model(connector: OllamaClient, model_name: str, prompt: str) -> Optional[str]:
    """
    Função genérica para conversar com um modelo e retornar a resposta.
    """
    if not connector.client:
        log.warning("Chat cancelado: O cliente Ollama não está conectado.")
        return None

    try:
        log.info(f"Conversando com o modelo: {model_name}...")
        log.info(f"Prompt: {prompt[:100]}...")

        messages: list[Dict[str, str]] = [{'role': 'user', 'content': prompt}]
        response: Dict[str, Any] = connector.client.chat(model=model_name, messages=messages)

        response_content: Optional[str] = response.get('message', {}).get('content')

        if response_content:
            return response_content
        else:
            log.warning("Nenhuma resposta recebida do modelo.")
            return None

    except Exception as e:
        log.error(f"Erro ao conversar com o modelo '{model_name}'. Ele existe no host? Erro: {e}")
        return None
