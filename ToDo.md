# TODO - Writer Verification Siamese Network

## 1. Architettura e Preprocessing
- [x] **Padding dinamico**: Gestito direttamente nei batch tramite `pad_collate`.
- [x] **Split per Autori**: Divisione del dataset (`split_by_writer`) per evitare data leakage tra train/val/test.
- [x] **Dataset di coppie**: Generazione bilanciata in `SiameseDataset` con supporto per coppie fisse (`fixed=True`).
- [x] **Pre-processing deterministico**: Ridimensionamento proporzionale con centramento e padding (`ResizeAndPad` 64x320) per tutto il dataset.
- [x] **Data Augmentation**: Integrazione di `RandomAffine` sulle immagini di training.
- [x] **Gestione isolata delle Run**: Creazione automatica delle cartelle `outputs/run_XXX/` per isolare pesi, log testuali e grafici.
- [x] **Visualizzazione Geometrica**: Estrazione embedding ed elaborazione t-SNE a intervalli regolari (`PLOT_INTERVAL`).


## 2. Qualità of Life (miglioramento codice)
- [ ] **Type Hints e Docstring Globale**:
  - [ ] Tipizzare e documentare `src/dataset.py` e `src/siamese_dataset.py` (definire `Dict[str, Any]` nei campioni).
  - [ ] Tipizzare e documentare `src/evaluation.py` (specificare input/output di `collect_distances`, `calculate_metrics`, ecc.).
  - [ ] Tipizzare e documentare `src/losses.py` e `src/models/siamese.py` (uso esplicito di `torch.Tensor` nei `forward`).
  - [ ] Tipizzare i moduli di utilità (`src/split.py`, `src/collate.py`, `src/run_manager.py`, `src/logger.py`).
- [ ] **Modularizzazione e Principio DRY (src/trainer.py)**: (da capire)
  - [ ] Estrarre `train_one_epoch` e `validate_one_epoch` in `src/trainer.py` per riutilizzare la logica sia in `train.py` che in futuri script di fine-tuning.
  - [ ] Creare la funzione `evaluate_test_set` in `src/evaluation.py` per ripulire lo script di test da calcoli ridondanti.


## 3. Ottimizzazione di Modello e Stabilizzazione
- [x] **Normalizzazione L2 degli Embedding**:
  - [x] Inserire `F.normalize(x, p=2, dim=1)` al termine di `HandwritingEncoder` (`src/models/siamese.py`) per vincolare gli embedding sull'ipersfera unitaria e stabilizzare la `Val loss`.
- [x] **Learning Rate Scheduler**:
  - [x] Integrare `ReduceLROnPlateau` nel ciclo di addestramento (`train.py`) per abbassare il LR quando la `Val loss` raggiunge un plateau, eliminando gli spike.
- [ ] **De-duplicazione SiameseDataset (fixed)**:
  - [ ] Assicurare che la generazione di coppie con `fixed=True` garantisca la riproducibilità senza estrarre duplicati.


## 4. Analisi Avanzata (OPZIONALE)
- [ ] **Metriche Disaggregate (Per Categoria di Coppia)**:
  - Estendere il calcolo delle metriche su 4 sottogruppi distinti per analizzare dove il modello fatica:
    - [ ] Stesso autore / testo diverso
    - [ ] Stesso autore / stesso testo
    - [ ] Autore diverso / testo diverso
    - [ ] Autore diverso / stesso testo
- [ ] **Biometric Evaluation Metrics**:
  - [ ] Calcolo della curva ROC e identificazione dell'**EER (Equal Error Rate)** sul test set per valutare il sistema secondo gli standard della biometrica.
- [ ] **Benchmark e Confronto Run**:
  - [ ] Salvare automaticamente le metriche finali di ciascuna run in un file riassuntivo (es. `outputs/benchmark.csv`) per mettere a confronto diretto `run_001` (senza L2) e `run_002` (con L2 e Scheduler).