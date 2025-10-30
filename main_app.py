import os
from ollama_client import OllamaClient, query_model
import tools
from typing import List, Optional

# --- FLUXO PRINCIPAL DO PROGRAMA ---
def main() -> None:
    """Função principal que orquestra a execução do script."""

    meu_conector: OllamaClient = OllamaClient()

    if meu_conector.client:
        # 1. Criar modelo personalizado
        log.info("Etapa 1: Criando modelo personalizado...")
        meu_conector.create_custom_model(
            profile_name='resumidor_tecnico',
            base_model='gemma2',
            system_role='Você é um especialista em resumir textos técnicos de forma clara e concisa.',
            temperature=0.3
        )

        # 2. Encontrar todos os PDFs no diretório 'data'
        log.info("Etapa 2: Processando arquivos PDF em lote...")
        pdf_directory: str = 'data'
        pdf_files: List[str] = tools.get_pdf_files_from_directory(pdf_directory)

        if not pdf_files:
            log.warning(f"Nenhum arquivo PDF encontrado em '{pdf_directory}'. Encerrando.")
            return

        # 3. Iterar e processar cada PDF
        for pdf_path in pdf_files:
            log.info(f"--- Processando arquivo: {pdf_path} ---")
            pdf_prompt: Optional[str] = tools.read_pdf_first_page(pdf_path)

            if pdf_prompt:
                # a. Chamar o modelo
                resposta: Optional[str] = query_model(meu_conector, 'resumidor_tecnico', pdf_prompt)

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
