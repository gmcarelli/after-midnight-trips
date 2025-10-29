import os
import fitz  # PyMuPDF
from typing import Optional
from dataclasses import dataclass

@dataclass
class SavePayload:
    """
    Estrutura de dados para passar as informações necessárias
    para salvar o resultado de um modelo.
    """
    output_filename: str
    response_content: str

def save_result(payload: SavePayload) -> None:
    """
    Salva o conteúdo da resposta de um modelo em um arquivo de texto.

    O arquivo é salvo no diretório 'results' com o nome de arquivo
    fornecido no payload.

    Args:
        payload (SavePayload): O objeto contendo o nome do arquivo e o conteúdo.
    """
    try:
        # Garante que o diretório de resultados exista
        results_dir = "results"
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)

        # Usa o nome de arquivo fornecido no payload
        file_path = os.path.join(results_dir, payload.output_filename)

        # Salva o conteúdo no arquivo
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(payload.response_content)

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


def get_pdf_files_from_directory(directory_path: str) -> list[str]:
    """
    Escaneia um diretório e retorna uma lista de caminhos completos para todos os arquivos .pdf.

    Args:
        directory_path (str): O caminho para o diretório a ser escaneado.

    Returns:
        list[str]: Uma lista de caminhos de arquivo completos para os arquivos PDF encontrados.
    """
    pdf_files = []
    try:
        if not os.path.isdir(directory_path):
            print(f"Erro: O diretório '{directory_path}' não foi encontrado.")
            return []

        for filename in os.listdir(directory_path):
            if filename.lower().endswith(".pdf"):
                full_path = os.path.join(directory_path, filename)
                pdf_files.append(full_path)

        print(f"Encontrados {len(pdf_files)} arquivos PDF em '{directory_path}'.")
        return pdf_files
    except Exception as e:
        print(f"Erro ao escanear o diretório '{directory_path}': {e}")
        return []
