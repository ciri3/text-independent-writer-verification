# Text-Independent Writer Verification via Convolutional Siamese Neural Network

Text-independent writer verification using a Siamese CNN to determine whether two handwritten word images were produced by the same writer.


## Struttura del Progetto

Struttura delle cartelle e dei file:
```text
text-independent-writer-verification/
│
├── data/                  # Cartella del dataset (esclusa da Git)
├── src/                   # Moduli software del progetto
│   ├── __init__.py        # Vuoto, necessario per l'importazione dei moduli senza errori
│   ├── dataset.py         # Gestione e generazione delle coppie (Dataset PyTorch)
│   ├── losses.py          # Contrastive Loss
│   └── models/
│       ├── siamese.py     # Architettura della CNN e della Siamese Network
|       └── __init__.py
│
├── train.py               # Addestramento
├── test.py                # Ttest e inferenza con pesi salvati
├── pyproject.toml         # Configurazione dipendenze, con uv
├── README.md
└── uv.lock
```


## Requisiti e Installazione

Il progetto utilizza `uv` per una gestione rapida e moderna delle dipendenze e dell'ambiente virtuale. Essendo un framework esterno è necessario installarlo separatamente.

Clona il repository e posizionati nella cartella del progetto:

```Bash
git clone https://github.com/ciri3/text-independent-writer-verification.git
cd text-independent-writer-verification
```
Sincronizza l'ambiente e installa le dipendenze:

```Bash
uv sync
```

### Nota: UV

È possibile evitare l'utilizzo di `uv` creando manualmente l'ambiente virtuale per il progetto e installando la versione di Python e delle librerie che si possono trovare nel file _pyproject.toml_.


## Training

Inserisci il dataset [IAM Handwritten Forms Dataset](https://www.kaggle.com/datasets/naderabdelghany/iam-handwritten-forms-dataset, "https://www.kaggle.com/datasets/naderabdelghany/iam-handwritten-forms-dataset") all'interno della cartella _data/_ seguendo la struttura con le cartelle dei singoli autori (000, 001, ecc.).

Avvia lo script di addestramento:

```Bash
uv run python train.py
```
Il sistema rileverà automaticamente la presenza di una GPU dedicata (CUDA) o farà il fallback su CPU/MPS.

Al termine dell'addestramento, i pesi del modello verranno salvati nella root come _siamese_writer_model.pth_.


## Testing / Inferenza

Per testare il modello addestrato e verificare se due campioni di scrittura appartengono allo stesso autore o a due autori differenti, puoi usare lo script di test dedicato:

```Bash
uv run python test.py
```