padding dinamico
divisioni per autori
crezione dataset di coppie
    decidere come creare le coppie 
data augmentation

### STRUTTURA 
IAMDataset
  gestisce le singole immagini IAM

SiameseDataset
  prende IAMDataset e costruisce coppie
  (image1, image2, label)

pad_collate
  prende più coppie e le rende batchabili con il padding

