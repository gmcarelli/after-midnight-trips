import os
from dotenv import load_dotenv
from typing import List, Optional

# Importa o cliente unificado e o erro de conexão pela fachada do pacote
from llm_host import OllamaClient, OllamaConnectionError
from llm_host.model_manager import ModelManager
from llm_host.chat_session import ChatSession
import tools
from logger_config import log

# Carrega as variáveis de ambiente do arquivo .env
load_dotenv()

def main() -> None:
    """Função principal que orquestra a execução do script."""

    # 1. Configuração inicial
    OLLAMA_HOST = os.getenv("OLLAMA_HOST")
    if not OLLAMA_HOST:
        log.error("A variável de ambiente OLLAMA_HOST não está definida.")
        return

    MODEL_NAME = "resumidor_tecnico"
    BASE_MODEL = "gemma2"
    SYSTEM_ROLE = "Você é um especialista em resumir textos técnicos de forma clara e concisa."

    # 2. Inicialização do cliente
    # A conexão é feita no construtor. Lança OllamaConnectionError em caso de falha.
    try:
        ollama_client = OllamaClient(host_url=OLLAMA_HOST)
    except OllamaConnectionError:
        # O erro já foi logado, então apenas encerramos.
        return

    # 3. Injetar o cliente unificado nos outros módulos (que esperam LLMTools)
    model_manager = ModelManager(tools=ollama_client)
    chat_session = ChatSession(tools=ollama_client, model_name=MODEL_NAME)

    # 4. Listar modelos para verificação
    log.info("Modelos disponíveis no host:")
    available_models = model_manager.list_models()
    for model in available_models:
        log.info(f"- {model.get('name')}")

    # 5. Criar o modelo personalizado, se necessário
    model_exists = any(model.get('name') == f"{MODEL_NAME}:latest" for model in available_models)
    if not model_exists:
        log.info(f"Criando o modelo personalizado '{MODEL_NAME}'...")
        success = model_manager.create_custom_model(
            base_model=BASE_MODEL, model_name=MODEL_NAME, system_role=SYSTEM_ROLE
        )
        if not success:
            log.error("Não foi possível criar o modelo personalizado. Encerrando.")
            return
    else:
        log.info(f"Modelo '{MODEL_NAME}' já existe. Pulando a criação.")


    # 6. Processar arquivos PDF em lote
    log.info("Iniciando o processamento de arquivos PDF...")
    pdf_directory: str = 'data'
    pdf_files: List[str] = tools.get_pdf_files_from_directory(pdf_directory)

    if not pdf_files:
        log.warning(f"Nenhum arquivo PDF encontrado em '{pdf_directory}'. Encerrando.")
        return

    # 7. Iterar e processar cada PDF
    for pdf_path in pdf_files:
        log.info(f"--- Processando arquivo: {pdf_path} ---")
        pdf_prompt: Optional[str] = tools.read_pdf_first_page(pdf_path)

        if pdf_prompt:
            resposta: str = chat_session.chat_with_model(pdf_prompt)
            if resposta:
                base_filename: str = os.path.splitext(os.path.basename(pdf_path))[0]
                output_filename: str = f"{base_filename}.txt"
                payload = tools.SavePayload(
                    output_filename=output_filename,
                    response_content=resposta
                )
                tools.save_result(payload)
        else:
            log.warning(f"Não foi possível ler o prompt do PDF '{pdf_path}'. Arquivo ignorado.")

if __name__ == "__main__":
    main()
