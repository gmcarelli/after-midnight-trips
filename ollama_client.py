import os
import ollama
from dotenv import load_dotenv
from typing import Optional, Mapping, Any, Dict

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
        host = None
        try:
            load_dotenv()
            host = os.getenv("OLLAMA_HOST")
            if not host:
                raise ValueError("A variável de ambiente OLLAMA_HOST não foi definida.")

            self.client = ollama.Client(host=host)
            self.client.list()
            print("Cliente Ollama inicializado e conexão com o host bem-sucedida.")

        except Exception as e:
            self.client = None
            print(f"ERRO CRÍTICO: Não foi possível conectar ao host Ollama em '{host}'.")
            print(f"Detalhes do erro: {e}")

    def create_custom_model(self, profile_name: str, base_model: str, system_role: str, temperature: float = 0.7) -> None:
        """
        Cria um novo modelo personalizado no host Ollama com base em um perfil.
        """
        if not self.client:
            print("Operação cancelada: O cliente Ollama não está conectado.")
            return

        try:
            parameters: Mapping[str, Any] = {"temperature": temperature}

            self.client.create(
                model=profile_name,
                from_=base_model,
                system=system_role,
                parameters=parameters
            )
            print(f"Modelo personalizado '{profile_name}' criado com sucesso a partir de '{base_model}'.")

        except Exception as e:
            print(f"Erro ao criar o modelo '{profile_name}': {e}")

def query_model(connector: OllamaClient, model_name: str, prompt: str) -> Optional[str]:
    """
    Função genérica para conversar com um modelo e retornar a resposta.
    """
    if not connector.client:
        print("Chat cancelado: O cliente Ollama não está conectado.")
        return None

    try:
        print(f"\\n--- Conversando com o modelo: {model_name} ---")
        print(f"Prompt: {prompt[:100]}...")

        messages = [{'role': 'user', 'content': prompt}]
        response: Dict[str, Any] = connector.client.chat(model=model_name, messages=messages)

        response_content = response.get('message', {}).get('content')

        if response_content:
            print(f"\\n[Resposta de {model_name}]:")
            print(response_content)
            return response_content
        else:
            print("Nenhuma resposta recebida do modelo.")
            return None

    except Exception as e:
        print(f"Erro ao conversar com o modelo '{model_name}'. Ele existe no host? Erro: {e}")
        return None
