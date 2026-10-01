import sys
from pathlib import Path

class Logger:
    """
    Gestisce scrittura simultanea dei log su terminale e
    su file di testo all'interno di una cartella 'outputs'.
    """
    def __init__(self, output_dir: Path, filename: str = "training_log.txt"):
        self.terminal = sys.stdout
        self.log_path = output_dir / filename
        self.log_file = open(self.log_path, "w", encoding="utf-8")

    def log(self, message: str = ""):
        # Stampa a schermo e scrive sul file contemporaneamente
        print(message)
        self.log_file.write(str(message) + "\n")
        self.log_file.flush() # Forza la scrittura immediata su disco

    def close(self):
        if self.log_file:
            self.log_file.close()