~~padding dinamico~~
~~divisioni per autori~~
~~crezione dataset di coppie~~
    ~~decidere come creare le coppie~~ 
data augmentation
questioni sul seed (test "troppo favorevole)
  train con coppie diverse ma riproducibili
  validation e test con coppie fisse
cartella con risultati
  pesi (file binari)
  testuali (stampe)
modifica pre-processing per renderlo deterministico (sia train che inferenza)
  altezza fissa (es: 64)
  larghezza W
  -> resize fino cotenimento in 64xW + centramento parola nella cornice e padding per colmare
  GLOBALE: per tutto il dataset (train, validation, test)
calcolo metriche per ogni categoria
  stesso autore e parole diverse
  stesso autore e stesse parole
  autore diverso e parole diverse
  autore diverso e stesse parole

### STRUTTURA 
~~IAMDataset~~
  ~~gestisce le singole immagini IAM~~

SiameseDataset
  ~~prende IAMDataset e costruisce coppie~~
  ~~(image1, image2, label)~~
  se impostato su fixed non genera doppioni

~~pad_collate~~
  ~~prende più coppie e le rende batchabili con il padding~~

