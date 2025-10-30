import logging
import sys

def setup_logger() -> logging.Logger:
    """
    Configura e retorna um logger padrão para o projeto.
    """
    # Cria um logger
    logger: logging.Logger = logging.getLogger('OllamaBatchProcessor')

    # Previne a duplicação de handlers se a função for chamada múltiplas vezes
    if logger.hasHandlers():
        return logger

    logger.setLevel(logging.INFO)

    # Cria um handler para o console (stdout)
    handler: logging.StreamHandler = logging.StreamHandler(sys.stdout)

    # Define o formato da mensagem de log
    formatter: logging.Formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    handler.setFormatter(formatter)

    # Adiciona o handler ao logger
    logger.addHandler(handler)

    return logger

# Instancia o logger para ser importado por outros módulos
log: logging.Logger = setup_logger()
