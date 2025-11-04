import os
from dotenv import load_dotenv
from typing import List, Optional

# Importações da nova arquitetura
from llm_host.ollama_client import OllamaClient
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

    # 2. Inicialização do cliente e dos gerenciadores
    try:
        # Instancia e conecta o cliente Ollama
        ollama_client = OllamaClient().connect_to_host(OLLAMA_HOST)
    except Exception:
        # A função connect_to_host já loga o erro, então apenas encerramos.
        return

    # Injeta o cliente nos outros módulos
    model_manager = ModelManager(tools=ollama_client)
    chat_session = ChatSession(tools=ollama_client, model_name=MODEL_NAME)

    # 3. Listar modelos para verificação (opcional)
    log.info("Modelos disponíveis no host:")
    available_models = model_manager.list_models()
    for model in available_models:
        log.info(f"- {model.get('name')}")

    # 4. Criar o modelo personalizado, se necessário
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


    # 5. Processar arquivos PDF em lote
    log.info("Iniciando o processamento de arquivos PDF...")
    pdf_directory: str = 'data'
    pdf_files: List[str] = tools.get_pdf_files_from_directory(pdf_directory)

    if not pdf_files:
        log.warning(f"Nenhum arquivo PDF encontrado em '{pdf_directory}'. Encerrando.")
        return

    # 6. Iterar e processar cada PDF
    for pdf_path in pdf_files:
        log.info(f"--- Processando arquivo: {pdf_path} ---")
        pdf_prompt: Optional[str] = tools.read_pdf_first_page(pdf_path)

        if pdf_prompt:
            # a. Chamar o modelo através da sessão de chat
            resposta: str = chat_session.chat_with_model(pdf_prompt)

            if resposta:
                # b. Determinar o nome do arquivo de saída
                base_filename: str = os.path.splitext(os.path.basename(pdf_path))[0]
                output_filename: str = f"{base_filename}.txt"

                # c. Criar o payload e salvar
                payload = tools.SavePayload(
                    output_filename=output_filename,
                    response_content=resposta
                )
                tools.save_result(payload)
        else:
            log.warning(f"Não foi possível ler o prompt do PDF '{pdf_path}'. Arquivo ignorado.")

if __name__ == "__main__":
    main()
