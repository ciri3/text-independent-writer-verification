~~padding dinamico~~
~~divisioni per autori~~
~~crezione dataset di coppie~~
    ~~decidere come creare le coppie~~ 
data augmentation
questioni sul seed (test "troppo favorevole")
cartella con risultati
  pesi (file binari)
  testuali (stampe)

### STRUTTURA 
~~IAMDataset~~
  ~~gestisce le singole immagini IAM~~

SiameseDataset
  ~~prende IAMDataset e costruisce coppie~~
  ~~(image1, image2, label)~~
  se impostato su fixed non genera doppioni

~~pad_collate~~
  ~~prende più coppie e le rende batchabili con il padding~~

