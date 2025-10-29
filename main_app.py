from ollama_client import OllamaClient
import tools
from typing import Dict, Any

def query_model(connector: OllamaClient, model_name: str, prompt: str) -> None:
    """
    Função genérica para conversar com um modelo e salvar o resultado.

    Args:
        connector (OllamaClient): A instância do conector Ollama.
        model_name (str): O nome do modelo a ser usado.
        prompt (str): A pergunta ou comando para o modelo.
    """
    if not connector.client:
        print("Chat cancelado: O cliente Ollama não está conectado.")
        return

    try:
        print(f"\n--- Conversando com o modelo: {model_name} ---")
        print(f"Prompt: {prompt[:100]}...") # Mostra os primeiros 100 caracteres do prompt

        messages = [{'role': 'user', 'content': prompt}]
        response: Dict[str, Any] = connector.client.chat(model=model_name, messages=messages)

        response_content = response.get('message', {}).get('content', 'Nenhuma resposta recebida.')

        print(f"\n[Resposta de {model_name}]:")
        print(response_content)

        # Salva o resultado usando a função do `tools`
        tools.save_result(model_name, response_content)

    except Exception as e:
        print(f"Erro ao conversar com o modelo '{model_name}'. Ele existe no host? Erro: {e}")


# --- FLUXO PRINCIPAL DO PROGRAMA ---
def main() -> None:
    """Função principal que orquestra a execução do script."""

    meu_conector = OllamaClient()

    # O script só continua se a conexão com o Ollama for bem-sucedida
    if meu_conector.client:
        # 1. Criar múltiplos perfis de modelo
        print("\n--- Etapa 1: Criando modelos personalizados ---")
        meu_conector.create_custom_model(
            profile_name='resumidor_tecnico',
            base_model='gemma2',
            system_role='Você é um especialista em resumir textos técnicos de forma clara e concisa.',
            temperature=0.3
        )

        # 2. Ler um prompt de um arquivo PDF
        print("\n--- Etapa 2: Lendo prompt do arquivo PDF ---")
        pdf_path = 'data/sample.pdf'
        pdf_prompt = tools.read_pdf_first_page(pdf_path)

        if pdf_prompt:
            # Usa o texto extraído do PDF como prompt para o modelo personalizado
            query_model(meu_conector, 'resumidor_tecnico', pdf_prompt)
        else:
            print("Não foi possível ler o prompt do PDF. A demonstração com PDF será ignorada.")

        # 3. Exemplo de chat com um modelo base
        print("\n--- Etapa 3: Chat com um modelo base ---")
        query_model(meu_conector, 'gemma2', "Explique o que é a singularidade tecnológica em três frases.")

if __name__ == "__main__":
    main()
