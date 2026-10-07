from pathlib import Path
import torch

class EarlyStopping:
    '''
    Interrompe l'addestramento se la validation loss non migliora dopo una determinata "pazienza".
    '''
    def __init__(self, patience: int = 7, min_delta: float = 0.0001, verbose: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False

    def __call__(self, val_loss: float) -> bool:
        # Se la nuova val_loss migliora significativamente rispetto al record
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0  # Resetta il contatore
        else:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} di {self.patience}")
            
            if self.counter >= self.patience:
                self.early_stop = True
                
        return self.early_stop