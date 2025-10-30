import os
import fitz  # PyMuPDF
from typing import Optional, List
from dataclasses import dataclass
from logger_config import log

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
    """
    try:
        results_dir: str = "results"
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
            log.info(f"Diretório '{results_dir}' criado.")

        file_path: str = os.path.join(results_dir, payload.output_filename)

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(payload.response_content)

        log.info(f"Resultado salvo com sucesso em: {file_path}")

    except Exception as e:
        log.error(f"Erro ao salvar o resultado em '{payload.output_filename}': {e}")


def read_pdf_first_page(file_path: str) -> Optional[str]:
    """
    Lê e retorna o texto da primeira página de um arquivo PDF.
    """
    try:
        doc: fitz.Document = fitz.open(file_path)

        if len(doc) > 0:
            first_page: fitz.Page = doc[0]
            text: str = first_page.get_text()
            doc.close()
            return text
        else:
            log.warning(f"O PDF '{file_path}' está vazio e não contém páginas.")
            doc.close()
            return None

    except FileNotFoundError:
        log.error(f"O arquivo PDF '{file_path}' não foi encontrado.")
        return None
    except Exception as e:
        log.error(f"Erro ao ler o arquivo PDF '{file_path}': {e}")
        return None


def get_pdf_files_from_directory(directory_path: str) -> List[str]:
    """
    Escaneia um diretório e retorna uma lista de caminhos para todos os arquivos .pdf.
    """
    pdf_files: List[str] = []
    try:
        if not os.path.isdir(directory_path):
            log.error(f"O diretório '{directory_path}' não foi encontrado.")
            return []

        for filename in os.listdir(directory_path):
            if filename.lower().endswith(".pdf"):
                full_path: str = os.path.join(directory_path, filename)
                pdf_files.append(full_path)

        log.info(f"Encontrados {len(pdf_files)} arquivos PDF em '{directory_path}'.")
        return pdf_files
    except Exception as e:
        log.error(f"Erro ao escanear o diretório '{directory_path}': {e}")
        return []
