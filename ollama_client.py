import os
import ollama
from dotenv import load_dotenv
from typing import Optional, Mapping, Any

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

            # Tenta conectar e inicializar o cliente
            self.client = ollama.Client(host=host)
            # Verifica a conexão listando os modelos (uma operação leve)
            self.client.list()
            print("Cliente Ollama inicializado e conexão com o host bem-sucedida.")

        except Exception as e:
            # Garante que o cliente seja None em caso de qualquer falha na conexão
            self.client = None
            print(f"ERRO CRÍTICO: Não foi possível conectar ao host Ollama em '{host}'.")
            print(f"Detalhes do erro: {e}")

    def create_custom_model(self, profile_name: str, base_model: str, system_role: str, temperature: float = 0.7) -> None:
        """
        Cria um novo modelo personalizado no host Ollama com base em um perfil.

        Args:
            profile_name (str): O nome que o novo modelo terá (ex: 'analista').
            base_model (str): O nome do modelo base a ser usado (ex: 'llama3').
            system_role (str): A instrução de sistema para o modelo.
            temperature (float): A temperatura padrão para o modelo.
        """
        if not self.client:
            print("Operação cancelada: O cliente Ollama não está conectado.")
            return

        try:
            # Passa os parâmetros diretamente para o método create
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
