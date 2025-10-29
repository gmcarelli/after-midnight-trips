import os
import fitz  # PyMuPDF
from datetime import datetime
from typing import Optional

def save_result(model_name: str, response_content: str) -> None:
    """
    Salva o conteúdo da resposta de um modelo em um arquivo de texto.

    O arquivo é salvo no diretório 'results' com um nome que inclui o nome do
    modelo e um timestamp.

    Args:
        model_name (str): O nome do modelo que gerou a resposta.
        response_content (str): O conteúdo da resposta a ser salvo.
    """
    try:
        # Garante que o diretório de resultados exista
        results_dir = "results"
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)

        # Cria um nome de arquivo único
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_model_name = model_name.replace(":", "_") # Para modelos como 'llama3:8b'
        file_name = f"{safe_model_name}_{timestamp}.txt"
        file_path = os.path.join(results_dir, file_name)

        # Salva o conteúdo no arquivo
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(response_content)

        print(f"Resultado salvo com sucesso em: {file_path}")

    except Exception as e:
        print(f"Erro ao salvar o resultado: {e}")


def read_pdf_first_page(file_path: str) -> Optional[str]:
    """
    Lê e retorna o texto da primeira página de um arquivo PDF.

    Args:
        file_path (str): O caminho para o arquivo PDF.

    Returns:
        Optional[str]: O texto extraído da primeira página, ou None se ocorrer um erro.
    """
    try:
        doc = fitz.open(file_path)

        if len(doc) > 0:
            first_page = doc[0]
            text = first_page.get_text()
            doc.close()
            return text
        else:
            print(f"Aviso: O PDF '{file_path}' está vazio e não contém páginas.")
            doc.close()
            return None

    except FileNotFoundError:
        print(f"Erro: O arquivo PDF '{file_path}' não foi encontrado.")
        return None
    except Exception as e:
        print(f"Erro ao ler o arquivo PDF '{file_path}': {e}")
        return None
