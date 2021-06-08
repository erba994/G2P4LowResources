import pandas as pd
import os
import re
from sklearn.model_selection import train_test_split

from multiprocessing import cpu_count
from collections import namedtuple
import sentencepiece as sp
from fairseq import file_utils
import argparse
import pathlib

import os.path as op
from typing import List, Optional
from fairseq.data.encoders.byte_bpe import ByteBPE
from fairseq.data.encoders.byte_utils import byte_encode
from fairseq.data.encoders.bytes import Bytes
from fairseq.data.encoders.characters import Characters
from fairseq.data.encoders.moses_tokenizer import MosesTokenizer
from fairseq.data.encoders.sentencepiece_bpe import SentencepieceBPE


## Here are defined functions and variables necessary for BBPE encoding
## BYTE_ENCODE

def _convert_to_bchar(dataset_column):
    return dataset_column.apply(lambda x: byte_encode(x.strip()))
    
def _get_bpe(dataset_column, filename, model_prefix: str, vocab_size: int):
    with open(filename, "w+", encoding="utf-8") as r:
        r.write(dataset_column.to_string(header=False, index=False))


    arguments = [
        f'--input={filename}', f'--model_prefix={filename}',
        f'--model_type=bpe', f'--vocab_size={vocab_size}',
        '--character_coverage=1.0', '--normalization_rule_name=identity',
        f'--num_threads={cpu_count()}', '--hard_vocab_limit=false'
    ]
    sp.SentencePieceTrainer.Train(" ".join(arguments))


def train_bbpe(dataset_column, prefix, bbpe_size: int):
    dataset_column = _convert_to_bchar(dataset_column)
    os.makedirs(datanamefolder + os.path.sep + "sentencepiece", 0o777, exist_ok=True)
    _get_bpe(dataset_column, datanamefolder + os.path.sep + "sentencepiece" + os.path.sep + prefix + str(bbpe_size), prefix + str(bbpe_size),
             bbpe_size)
    
def _apply_bbpe(model_path: str, dataset_column):
    Args = namedtuple("Args", ["sentencepiece_model_path"])
    args = Args(sentencepiece_model_path=model_path)
    tokenizer = ByteBPE(args)
    return dataset_column.apply(lambda x: tokenizer.encode(x.strip()))

def apply_bbpe(dataset_column, prefix, bbpe_size: int):
    dataset_column = _convert_to_bchar(dataset_column)
    dataset_column = _apply_bbpe(datanamefolder + os.path.sep + "sentencepiece" + os.path.sep + prefix + str(bbpe_size) + '.model', dataset_column)
    return dataset_column


## Here the argument parser is defined

def hyphenated(string):
     return '-'.join([word for word in string.casefold().split()])

parser = argparse.ArgumentParser(description='Generate datasets for g2p training')
parser.add_argument('dataname', metavar='name', type=hyphenated,
                    help='define the name of the dataset')
parser.add_argument('datapath', type=pathlib.Path,  help='path where the language dictionaries are located', default="/mnt/d/Lorenzo/ProgramData/Thesis/wikiprondata/")
parser.add_argument('--bbpe', action='store_true',
                    help='apply Byte-level Byte-Pair-Encoding')
parser.add_argument('--normlang', action='store_true',
                    help='stratify database by languages')
parser.add_argument('--normgroup', action='store_true',
                    help='stratify database by language groups')
parser.add_argument('--normfam', action='store_true',
                    help='stratify database by language families')
parser.add_argument('--condlang', action='store_true',
                    help='condition database by languages')
parser.add_argument('--condgroup', action='store_true',
                    help='condition database by language groups')
parser.add_argument('--condfam', action='store_true',
                    help='condition database by language families')
args = parser.parse_args()
dataname = args.dataname
datafolder = args.datapath
datanamefolder = "." + os.path.sep + dataname
os.mkdir(datanamefolder)

## load language dicts

languagedata = pd.read_csv('languagefamilydata.csv', sep=";", header=0, index_col="Lcode")

## dataset normalization

g2pdata = pd.DataFrame(columns=["graphemes", "phonemes", "language", "group", "family", "transcription", "string"])
g2pfull = pd.DataFrame(columns=["graphemes", "phonemes", "language", "group", "family", "transcription", "string"])
datalen = []
dfs = []
for file in os.listdir(datafolder):
    if os.path.isdir(datafolder.__str__() + os.path.sep + file):
        continue
    frame = pd.read_csv(datafolder.__str__() + os.path.sep + file, sep="\t", names=["graphemes", "phonemes"], usecols=[0,1], encoding="utf-8")
    lang = re.sub(r"([a-z]+_).+\.tsv", r"\1", file)
    tran = re.sub(r".+?(phone[m|t]ic)\.tsv", r"\1", file)
    frame["language"] = lang
    frame["group"] = languagedata.loc[lang,"Gcode"]
    frame["family"] = languagedata.loc[lang,"Fcode"]
    frame["transcription"] = tran
    dfs.append(frame)
g2pdata = g2pdata.append(dfs, ignore_index = True)
g2pfull = g2pdata
g2punk = g2pdata[g2pdata["language"].isin(["kur_","ban_","scn_","acw_","bel_","kaz_","mic_","tpw_"])]
subset = g2pdata[g2pdata["language"].isin(["dan_"])].iloc[0:1500,:]
g2punk = pd.concat([g2punk,subset]).reset_index()
g2pdata = g2pdata[~g2pdata["language"].isin(["dan_","kur_","ban_","scn_","acw_","bel_","kaz_","mic_","tpw_"])].reset_index()
g2pdata = g2pdata.fillna('')
g2punk = g2punk.fillna('')

if args.normlang:
    g2pdata = g2pdata.drop_duplicates(subset=['graphemes'])
    g2punk = g2punk.drop_duplicates(subset=['graphemes'])
    counts = g2pdata.groupby('language').count().graphemes.sort_values(ascending=False)
    samplen = int(g2pdata.groupby('language').count().graphemes.sort_values(ascending=False).mean())
    filterlanguage = list([item for item in counts.keys() if counts[item] > samplen])
    g2pmore = g2pdata[g2pdata['language'].isin(filterlanguage)].groupby('language').apply(lambda x: x.sample(n=samplen)).reset_index(drop=True)
    g2pless = g2pdata[~g2pdata['language'].isin(filterlanguage)]
    g2pdata = pd.concat([g2pmore,g2pless], ignore_index=True).reset_index(drop=True)
if args.normgroup:
    g2pdata = g2pdata.drop_duplicates(subset=['graphemes'])    
    g2punk = g2punk.drop_duplicates(subset=['graphemes'])
    counts = g2pdata.groupby('group').count().graphemes.sort_values(ascending=False)
    samplen = int(g2pdata.groupby('group').count().graphemes.sort_values(ascending=False).mean())
    filterlanguage = list([item for item in counts.keys() if counts[item] > samplen])
    g2pmore = g2pdata[g2pdata['group'].isin(filterlanguage)].groupby('group').apply(lambda x: x.sample(n=samplen)).reset_index(drop=True)
    g2pless = g2pdata[~g2pdata['group'].isin(filterlanguage)]
    g2pdata = pd.concat([g2pmore,g2pless], ignore_index=True).reset_index(drop=True)
if args.normfam:
    g2pdata = g2pdata.drop_duplicates(subset=['graphemes'])
    g2punk = g2punk.drop_duplicates(subset=['graphemes'])
    counts = g2pdata.groupby('family').count().graphemes.sort_values(ascending=False)
    samplen = int(g2pdata.groupby('family').count().graphemes.sort_values(ascending=False).mean())
    filterlanguage = list([item for item in counts.keys() if counts[item] > samplen])
    g2pmore = g2pdata[g2pdata['family'].isin(filterlanguage)].groupby('family').apply(lambda x: x.sample(n=samplen)).reset_index(drop=True)
    g2pless = g2pdata[~g2pdata['family'].isin(filterlanguage)]
    g2pdata = pd.concat([g2pmore,g2pless], ignore_index=True).reset_index(drop=True)

g2pdata["graphemes"] = g2pdata["graphemes"].apply(lambda x: " ".join(list(str(x))))
g2punk["graphemes"] = g2punk["graphemes"].apply(lambda x: " ".join(list(str(x))))

if args.bbpe:
    train_bbpe(pd.concat([g2pdata["graphemes"], g2punk["graphemes"]]), "bbpe_gra", 4098)
    g2pdata["graphemes"] = apply_bbpe(g2pdata["graphemes"], "bbpe_gra", 4098)
    g2punk["graphemes"] = apply_bbpe(g2punk["graphemes"], "bbpe_gra", 4098)

if args.condlang:
    g2pdata["string"] = g2pdata["string"] + " " + g2pdata["language"]
    g2punk["string"] = g2punk["string"] + " " + g2punk["language"]
if args.condgroup:
    g2punk["string"] = g2punk["string"] + " " + g2punk["group"]
    g2pdata["string"] = g2pdata["string"] + " " + g2pdata["group"]
if args.condfam:
    g2pdata["string"] = g2pdata["string"] + " " + g2pdata["family"]
    g2punk["string"] = g2punk["string"] + " " + g2punk["family"]
if args.condfam or args.condgroup or args.condlang:
    g2pdata["string"] = g2pdata["string"] + " "
    g2punk["string"] = g2punk["string"] + " "

if args.normfam:
    train, dev = train_test_split(g2pdata, test_size=0.05, random_state=42, stratify=g2pdata.family.to_numpy())
elif args.normgroup:
    train, dev = train_test_split(g2pdata, test_size=0.05, random_state=42, stratify=g2pdata.group.to_numpy())
else:
    train, dev = train_test_split(g2pdata, test_size=0.05, random_state=42, stratify=g2pdata.language.to_numpy())

test = g2punk


datasets = [train, test, dev]

for split in ["train", "test", "dev"]:
    for lang in ["gra", "pho"]:
        with open(datanamefolder + os.path.sep + split + "_" + dataname + "." + lang, "w+", encoding="utf-8") as f:
            f.write("")
for index, split in enumerate(["train", "test", "dev"]):
    with open(datanamefolder + os.path.sep + split + "_" + dataname + "." + "gra", "a", encoding="utf-8") as f_g:
        with open(datanamefolder + os.path.sep + split + "_" + dataname + "." + "pho", "a", encoding="utf-8") as f_p:
            for index, data in datasets[index].iterrows():
                f_g.write(data["string"] + data["graphemes"] + "\n")
                f_p.write(data["string"] + data["phonemes"] + "\n")
