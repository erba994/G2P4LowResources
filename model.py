import argparse
import pathlib
import os
import sys
import logging
import torch
from fairseq import options, distributed_utils
from fairseq.logging import meters, metrics, progress_bar
from fairseq_cli import preprocess, train, generate

def main():
#python model.py testbbpe D:\MEGAsync\Program\Thesis\GITHUB

    def hyphenated(string):
        return '-'.join([word for word in string.casefold().split()])

    parser = argparse.ArgumentParser(description='Train and evaluate datasets for g2p training')
    parser.add_argument('dataname', metavar='name', type=hyphenated,
                        help='define the name of the dataset')
    parser.add_argument('datapath', type=pathlib.Path, help='path where the language dictionaries are located',
                        default="/mnt/d/Lorenzo/ProgramData/Thesis/wikiprondata/")
    parser.add_argument('--preprocess', action='store_true',
                        help='preprocess data')
    parser.add_argument('--train', action='store_true',
                        help='train data')
    parser.add_argument('--eval', action='store_true',
                        help='evaluate data')

    args = parser.parse_args()
    dataname = args.dataname
    basepath = args.datapath.__str__()
    datapath = args.datapath.__str__() + os.path.sep + dataname
    destpath = args.datapath.__str__() + os.path.sep + dataname + os.path.sep + "datasetbin"
    argtrain = args.train
    argprocess = args.preprocess
    argeval = args.eval
    #destnamefolder = "." + os.path.sep + destpath
    #os.mkdir(destnamefolder)

    if argprocess:

        arguments = ["", "--source-lang=gra", "--target-lang=pho", f"--destdir={destpath}",
        "--workers=4", f"--testpref={datapath}{os.path.sep}test_{dataname}", "--fp16",
        f"--trainpref={datapath}{os.path.sep}train_{dataname}", f"--validpref={datapath}{os.path.sep}dev_{dataname}", "--thresholdtgt=-1", "--thresholdsrc=-1"]

        sys.argv = arguments
        preprocess.cli_main()

    if argtrain:

        logging.basicConfig(
            format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            level=os.environ.get("LOGLEVEL", "INFO").upper(),
            stream=sys.stdout,
        )
        logger = logging.getLogger("fairseq_cli.train")

        arguments = [f"{destpath}", "--task=translation", f"--user-dir={basepath}{os.path.sep}model{os.path.sep}gru_transformer",
            "--arch=gru_transformer", "--encoder-layers=2", "--decoder-layers=2", "--dropout=0.3",
            "--optimizer=adam", "--adam-betas=(0.9, 0.98)", "--lr=5e-4", "--lr-scheduler=inverse_sqrt", "--warmup-updates=4000",
            "--criterion=label_smoothed_cross_entropy", "--label-smoothing=0.1", "--log-format=tqdm", "--log-interval=100",
            f"--save-dir={datapath}{os.path.sep}checkpoints", "--batch-size=256", "--patience=5",
            "--no-epoch-checkpoints", "--num-workers=4", "--no-last-checkpoints", f"--tensorboard-logdir={basepath}{os.path.sep}tensorboard", "--fp16"]

        args = options.parse_args_and_arch(options.get_training_parser(), input_args=arguments, modify_parser=None)
        train.main(args)

    if argeval:

        arguments = [f"{destpath}", "--task=translation",
        f"--user-dir={basepath}{os.path.sep}model{os.path.sep}gru_transformer",
        "--source-lang=gra", "--target-lang=pho", "--gen-subset=test",
        f"--path={datapath}{os.path.sep}checkpoints{os.path.sep}checkpoint_best.pt", "--num-workers=4", f"--tensorboard-logdir={basepath}{os.path.sep}tensorboard"]

        args = options.parse_args_and_arch(options.get_generation_parser(), input_args=arguments, modify_parser=None)
        sys.stdout = open(datapath + os.path.sep + "results.txt", "w+", encoding='utf-8')
        generate.main(args)
        sys.stdout.close()

if __name__ == "__main__":
    main()



"""
!fairseq-train "/content/drive/My Drive/COLABDRIVE/bin_bbpe4096" --task translation \
--user-dir "/content/drive/My Drive/Colab Notebooks/thesis/bbpe/gru_transformer" \
    --arch gru_transformer --encoder-layers 2 --decoder-layers 2 --dropout 0.3 --share-all-embeddings \
    --optimizer adam --adam-betas '(0.9, 0.98)' \
    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --log-format 'tqdm' --log-interval 100 --save-dir "/content/drive/My Drive/COLABDRIVE/checkpoints" \
    --max-sentences 512 --max-update 500000 --update-freq 2 --no-epoch-checkpoints

!fairseq-generate "/mnt/d/Lorenzo/ProgramData/Thesis/g2p_testadata/bin2" --task translation \
--user-dir "/mnt/d/MEGAsync/Program/Thesis/bbpe/gru_transformer" \
    --source-lang gra --gen-subset test --sacrebleu --path "/mnt/d/Lorenzo/ProgramData/Thesis/g2p_testadata/checkpoints_old/checkpoint_best.pt" \
    --target-lang pho > "/mnt/d/Lorenzo/ProgramData/Thesis/g2p_testadata/checkpoints_old_results.txt"
"""