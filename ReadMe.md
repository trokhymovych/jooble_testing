# Testing task (Data Scientist) Jooble

This repository contains the script DataProcessing.py as a solution for the task provided in the attached .pdf file.

As one of the requirements was to be able to work with large files, I decided to use batch processing 
which enables both using matrix operations and not loading all the data to memory.

In order to run the script run the following in command line:
1) ```pip install -r requirements.txt ```
2) ```python DataProcessing.py data/train.tsv data/train_proc.tsv 100```
, where data/train.tsv - file to process, data/train_proc.tsv - resulting file, 100 - size of the batch 