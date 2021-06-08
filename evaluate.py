#!/usr/bin/env python
"""Evaluates sequence model.

This script assumes the gold and hypothesis data is stored in a two-column TSV
file, one example per line."""

__author__ = "Kyle Gorman"

import argparse
import logging
import multiprocessing
import logging
import numpy  # type: ignore
from typing import Any, Iterator, List, Tuple
import os
import pandas as pd
import re
from fairseq.data.encoders.byte_bpe import ByteBPE
from collections import namedtuple
import sacrebleu
import pathlib
import evaluate

Labels = List[Any]

def edit_distance(x: Labels, y: Labels) -> int:
    # For a more expressive version of the same, see:
    #
    #     https://gist.github.com/kylebgorman/8034009
    idim = len(x) + 1
    jdim = len(y) + 1
    table = numpy.zeros((idim, jdim), dtype=numpy.uint8)
    table[1:, 0] = 1
    table[0, 1:] = 1
    for i in range(1, idim):
        for j in range(1, jdim):
            if x[i - 1] == y[j - 1]:
                table[i][j] = table[i - 1][j - 1]
            else:
                c1 = table[i - 1][j]
                c2 = table[i][j - 1]
                c3 = table[i - 1][j - 1]
                table[i][j] = min(c1, c2, c3) + 1
    return int(table[-1][-1])


def score(gold: Labels, hypo: Labels) -> Tuple[int, int]:
    """Computes sufficient statistics for LER calculation."""
    edits = edit_distance(gold, hypo)
    #if edits:
    #    logging.warning(
    #        "Incorrect prediction:\t%r (predicted: %r)",
    #        " ".join(gold),
    #        " ".join(hypo),
    #    )
    return (edits, len(gold))


def tsv_reader(gold, hypo):
    for i,v in enumerate(gold):
        yield (gold[i], hypo[i])


def WER(cores, gold, hypo) -> None:
    # Word-level measures.
    correct = 0
    incorrect = 0
    # Label-level measures.
    total_edits = 0
    total_length = 0
    # Since the edit distance algorithm is quadratic, let's do this with
    # multiprocessing.
    gold = [re.sub(r" ", r"", i) for i in gold]
    hypo = [re.sub(r" ", r"", i) for i in hypo]
    with multiprocessing.Pool(cores) as pool:
        gen = pool.starmap(score, tsv_reader(gold, hypo))
        for (edits, length) in gen:
            if edits == 0:
                correct += 1
            else:
                incorrect += 1
            total_edits += edits
            total_length += length
    print(f"WER:\t{100 * incorrect / (correct + incorrect):.2f}")
    print(f"LER:\t{100 * total_edits / total_length:.2f}")

def BLEU(hypo,gold):
    bleu = sacrebleu.corpus_bleu("\n".join(hypo), "\n".join(gold))
    print(bleu)

def get_dictionary(datafolder):
    g2pdata = pd.DataFrame(columns=["graphemes", "phonemes", "transcription"])
    dfs = []
    for file in os.listdir(datafolder):
        if os.path.isdir(datafolder.__str__() + os.path.sep + file):
            continue
        frame = pd.read_csv(datafolder.__str__() + os.path.sep + file, sep="\t", names=["graphemes", "phonemes"], usecols=[0,1], encoding="utf-8")
        tran = re.sub(r".+?(phone[m|t]ic)\.tsv", r"\1", file)
        frame["transcription"] = tran
        dfs.append(frame)
    g2pdata = g2pdata.append(dfs, ignore_index = True)
    phon = g2pdata[g2pdata['transcription'] == "phonemic"]
    phot = g2pdata[g2pdata['transcription'] == "phonetic"]
    listcharphon = list(phon["phonemes"])
    listcharphon = "".join(listcharphon)
    listcharphon = set(listcharphon)
    listcharphot = list(phot["phonemes"])
    listcharphot = "".join(listcharphot)
    listcharphot = set(listcharphot)
    return listcharphon - listcharphot

def hyphenated(string):
    return '-'.join([word for word in string.casefold().split()])

def main():
    parser = argparse.ArgumentParser(description='Train and evaluate datasets for g2p training')
    parser.add_argument('dataname', metavar='name', type=hyphenated,
                        help='define the name of the dataset')
    parser.add_argument('datapath', type=pathlib.Path, help='path where the models are located',
                        default="/mnt/d/Lorenzo/ProgramData/Thesis/wikiprondata/")
    parser.add_argument('--WER', action='store_true',
                        help='WER score')
    parser.add_argument('--BLEU', action='store_true',
                        help='BLEU score')
    parser.add_argument('--BBPE', action='store_true',
                        help='model has BBPE')
    
    
    args = parser.parse_args()
    argbbpe = args.BBPE
    argbleu = args.BLEU 
    argwer = args.WER
    datadir = args.datapath.__str__()
    resdir = args.datapath.__str__() + os.path.sep + args.dataname
    resfile = resdir + os.path.sep + "results.txt"
    dumptest = resdir + os.path.sep + "resultsdump.txt"
    if argbbpe:
        Args = namedtuple("Args", ["sentencepiece_model_path"])
        args = Args(sentencepiece_model_path=resdir + os.path.sep + "sentencepiece" + os.path.sep + "bbpe_gra4098.model")
        tokenizer = ByteBPE(args)


    with open(resfile, "r") as f:
        results = f.read()
    grapheme = re.findall(r"S.+\t(.+)", results)
    phonemes_gold = re.findall(r"T.+\t(.+)", results)
    phonemes_pred = re.findall(r"H.+\t.+\t(.+)", results)
    grapheme = [re.sub(r"^.{3,4}\s(.+)$", r"\1", i) for i in grapheme]
    grapheme = [re.sub(r"<?<unk>>?", r"", i) for i in grapheme]
    phonemes_gold  = [re.sub(r"^.{3,4}\s(.+)$", r"\1", i) for i in phonemes_gold]
    phonemes_gold = [re.sub(r"<?<unk>>?", r"", i) for i in phonemes_gold]
    phonemes_pred = [re.sub(r"^.{3,4}\s(.+)$", r"\1", i) for i in phonemes_pred]
    phonemes_pred = [re.sub(r"<?<unk>>?", r"", i) for i in phonemes_pred]
    phoremove = get_dictionary(datadir + os.path.sep + "datasetfiles")
    patterns_regex = re.compile('|'.join([re.escape(i) for i in phoremove]))
    phonemes_gold = [patterns_regex.sub(r"", i) for i in phonemes_gold]
    phonemes_pred = [patterns_regex.sub(r"", i) for i in phonemes_pred]
    if argbbpe:
        grapheme = [ByteBPE.decode(i) for i in grapheme]
        grapheme = [ByteBPE.decode(i) for i in grapheme]
    #resdf = pd.DataFrame(list(zip(grapheme,phonemes_gold,phonemes_pred)))

    if argbleu:
        BLEU(phonemes_pred,phonemes_gold)
    if argwer:
        WER(8, phonemes_gold, phonemes_pred)

if __name__ == "__main__":
    main()