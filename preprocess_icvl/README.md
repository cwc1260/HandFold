# Preprocess ICVL dataset
## Prepare training set
```run matlab preprocess_icvl_single```
## prepare testing set
As described in the manuscript we use the result of global folding to segment the hand area, we provided the coordinates "saved_fold1.txt" from a pretrained global folding model for a convinient generation.

```run matlab preprocess_icvl_pre ```