# G2P4LowResources
Code for Thesis Work Multilingual Grapheme-to-Phoneme Translation for Low-Resource Languages

PREREQUISITES

The scripts should work on every flavour of Linux, Windows and MacOS, they have been tested in Windows 10 and Ubuntu 18.04

pip3 install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio===0.8.1 -f https://download.pytorch.org/whl/torch_stable.html 
pip3 install sentencepiece fairseq sacrebleu tensorboardX pandas

RUNNING

!python datasets.py "dataset_name" "datasetfiles_folder" 
for creating the dataset for training from the Wikipron parsed data. Optional arguments are the conditioning and normalization flags available in the script's help

!python "dataset_name" "datasetprocessed_folder" --train --preprocess --eval
for serializing, training and creating evaluation file for a single processed dataset

!python evaluate.py "dataset_name" "datasetprocessed_folder" --BLEU --WER
for scoring an evaluation file of a dataset
