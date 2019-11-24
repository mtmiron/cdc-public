"""
Handles the training and evaluation of single (non-ensemble) Torch models.
"""
import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
from inspect import getsourcefile
import plac
from transformers import *
import pandas as pd
import numpy as np
import preproc as cdcpreproc
import re
from simpletransformersmodel import TransformerModel
import logging
import torch
import vars as conf
import random
from gibberish import *
from distilbertmod import MyDistilBert


MODELMAP = { 'distilbert': ('distilbert', 'distilbert-base-uncased'),
             'bert': ('bert', 'bert-large-uncased-whole-word-masking'),
             'roberta': ('roberta', 'roberta-large'),
             'xlnet': ('xlnet', 'xlnet-large-cased')
           }

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

MODEL_ARGS = { 'eval_all_checkpoints': True,
               'use_tensorboard': True,
               'process_count': 6,
               'save_steps': 500,
               'max_position_embeddings': 64,
#               'use_fp16': False,
#               'fp16': False,
             }

TEXTCOL = conf.TEXTCOL
LABELCOL = conf.LABELCOL

cdcpreproc.TEXTCOL = TEXTCOL
cdcpreproc.LABELCOL = LABELCOL

EVENTCOUNTS_DF = pd.read_csv("eventcounts.csv")

def build_model(arch='distilbert', nocuda=False):
    """
    Return a built model.
    """
    model = TransformerModel(MODELMAP[arch][0], MODELMAP[arch][1], num_labels=100, use_cuda=bool(not nocuda), args=MODEL_ARGS)
#    model.model = MyDistilBert.from_pretrained("distilbert-base-uncased", num_labels=100)
#    model.model = model.model.from_pretrained('distilbert-base-uncased', num_labels=100)
    return model


def fastText_solution(df):
    """
    Return a fastText solution (don't bother with this).
    """
    import fasttext
    model = fasttext.load_model("fastText.bin")
    ret = []
    solution = model.predict(list(df['text']))
    for i in range(len(solution[0])):
        prob = solution[1][i][0]
        label = solution[0][i][0]
        ret.append({'prob': prob, 'label': int(label.replace("__label__", ""))})
    return ret


def ensemble_solution(prelines, ft_lines, threshold=0.99):
    """
    Return a fastText-ensembled solution (don't bother with this).
    """
    ret = []
    for preline, ft_line in zip(prelines, ft_lines):
        bert_label = int(preline.split(',')[-1])
        if bert_label != ft_line['label']:
            logger.debug("BERT != fastText: {},{} vs {}".format(preline.split(",")[0],
                                                                bert_label,
                                                                ft_line['label']))
            if ft_line['prob'] >= threshold:
                logger.info("fastText probability {} overriding BERT label, {}->{}".format(ft_line['prob'],
                                                                                            bert_label,
                                                                                            ft_line['label']))
                text, sex, age, event = preline.split(',')
                preline = "{},{},{},{}".format(text,
                                               sex,
                                               age,
                                               ft_line['label'])
        ret.append(preline)
    return ret


def append_gibberish(df, n):
    """
    Append mostly-grammatically-correct gibberish to list of input samples.
    """
    gibs = []
    sexes = []
    ages = []
    events = []
    while n > 0:
        text = A[random.randint(0, len(A) - 1)]
        for s in [B, C, D, E, F]:
            text += " " + s[random.randint(0, len(s) - 1)]
        sex = random.randint(1,2)
        age = random.randint(1, 100)
        event = 99
        gibs.append(text)
        ages.append(age)
        sexes.append(sex)
        events.append(event)
        n -= 1
    gib_df = pd.DataFrame({TEXTCOL: gibs, 'sex': sexes, 'ages': ages, LABELCOL: events})
    return pd.concat([df, gib_df], ignore_index=True, sort=False)

def prep_data(df=pd.read_csv("train.csv"), **kwargs):
    """
    Expand abbreviations and whatnot
    """
    df = cdcpreproc.preproc(df, **kwargs)#stack=stack, lower=lower, random=random, nopreprocess=nopreprocess)
    if 'gibberish' in kwargs and kwargs['gibberish']:
        df = append_gibberish(df, kwargs['gibberish'])
    if 'augment' in kwargs and kwargs['augment']:
        df = pd.concat([df, conf.AUGMENT_DF], ignore_index=True, sort=False)
    if 'random' in kwargs and kwargs['random']:
        df = df.sample(frac=1).reset_index(drop=True)
    try:
        return pd.DataFrame({'text': df[TEXTCOL], 'labels': df[LABELCOL].astype(int)})
    except:
        return pd.DataFrame({'text': df['text']})


def predict(model, df=pd.read_csv("test.csv"), fastText=False):
    """
    Return a list of predictions for the given dataframe['text'] values.
    """
    return model.predict([x for x in df['text']])
    counts = {}
    preds = []
    for i in len(tokenpreds):
        for j in tokenpreds[i]:
            if counts.get(j, None) is not None:
                counts[j] += 1
            else:
                counts[j] = 1
        highest = sorted(counts, key=lambda x: counts[x])
        highest = highest[-1]
        preds.append(highest)
    return preds


def main(modelpath: ("The location of the model checkpoint", "option", "c"),
         datapath: ("The location of the data .csv", "option", "d"),
         solutionpath: ("The path to output the solution to", "option", "s"),
         loadpath: ("Load this model", "option", "l"),
         train: ("Do training", "flag", "t"),
         evaluate: ("Evaluate", "flag", "e"),
         lower: ("Don't lowercase text samples", "flag", "L"),
         stack: ("Stack examples (before preproc + after preproc)", "flag", "S"),
         random: ("Shuffle training samples", "flag", "r"),
         nopreprocess: ("Do not preprocess data", "flag", "np"),
         fastText: ("Use fastText in a simple ensemble configuration", "flag", "f"),
         verbose: ("Set logLevel to INFO", "flag", "v"),
         spellcheck: ("Spellcheck and correct input samples", "flag", "C"),
         nocuda: ("Do not use the GPU", "flag", "nc"),
         nonumbers: ("Strip out numbers", "flag", "nn"),
         gibberish: ("Number of gibberish texts to generate and assign UNCLASSIFIABLE to", "option", "g", int),
         permute: ("Append permuted samples to training data by randomly swapping order of words", "flag", "P"),
         augment: ("Augment training data", "flag", "A"),
#         validpath: ("Path to the validation set", "option", "V"),
         desc: ("Add event code descriptions to the data", "flag", "D"),
         max_seq_len: ("Max sequence length", "option", "m", int)=64,
         learning_rate: ("Learning rate", "option", "lr", float)=3e-5,
         adam_epsilon: ("Adam epsilon", "option", "E", float)=1e-6,
         weight_decay: ("Weight decay", "option", "w", float)=0.1,
         warmup_ratio: ("Warmup ratio", "option", "wr", float)=0.06,
         batch_size: ("Batch size to use", "option", "B", int)=32,
         num_train_epochs: ("Number of training epochs", "option", "ep", int)=5,
         architecture: ("The model to use, e.g. distilbert, roberta, etc.", "option", "a")='distilbert'):
    """
    The -d DATAPATH parameter is required.
    """

    MODEL_ARGS['max_seq_len'] = max_seq_len
    MODEL_ARGS['adam_epsilon'] = adam_epsilon
    MODEL_ARGS['learning_rate'] = learning_rate
    MODEL_ARGS['weight_decay'] = weight_decay
    MODEL_ARGS['warmup_ratio'] = warmup_ratio 
    MODEL_ARGS['train_batch_size'] = batch_size
    MODEL_ARGS['num_train_epochs'] = num_train_epochs
    lower = bool(not lower)

    if not modelpath and not datapath:
        plac.call(main, ["-h"])
        sys.exit()
    if verbose:
        logging.getLogger().setLevel(logging.INFO)
    if train:
        model = build_model(architecture, nocuda=nocuda)
        try:
            if loadpath:
                loaded = model.model.from_pretrained(loadpath, num_labels=100)
                model.model = loaded
                tokenizer = model.tokenizer.from_pretrained(loadpath)
                model.tokenizer = tokenizer
                logger.warning("Successfully loaded checkpoint and tokenizer from {}".format(loadpath))
        except Exception as exc:
            logger.warning("Unable to load checkpoint: {}".format(exc))
        model.model.train()
        df = pd.read_csv(datapath)
        eval_df = pd.read_csv("valid5000.csv")
        eval_mini_df = pd.read_csv("validmini.csv")
        data = prep_data(df, stack=stack, lower=lower, random=random, nopreprocess=nopreprocess, descriptions=desc,
                         gibberish=gibberish, permutations=permute, augment=augment, spelling=spellcheck, nonumbers=nonumbers)
        eval_data = prep_data(eval_df, lower=lower, spelling=spellcheck, nonumbers=nonumbers)
        eval_mini = prep_data(eval_mini_df, lower=lower, spelling=spellcheck, nonumbers=nonumbers)

        print("Stacking: {}\nLowercasing: {}\nShuffling: {}\nPreprocessing: {}\nNumber of samples: {}\n".format(
              stack, lower, random, not nopreprocess, len(df)) +
              "Descriptions added: {}\nfastText ensembling: {}\nArchitecture: {}\nPermuting: {}\nGibberish: {}\n".format(
              desc, fastText, architecture, permute, gibberish) +
              "Augmenting: {}\n\n".format(augment))

        TransformerModel.train_model(model, data, eval_df=eval_data, output_dir=modelpath, eval_mini_df=eval_mini)

        with open(os.path.join(modelpath, "cmdline.txt"), mode="w") as f:
            f.write(str(sys.argv))
    else:
        model = build_model(architecture, nocuda=nocuda)
        model.model = model.model.from_pretrained(loadpath, num_labels=100)
        model.model.eval()
        df = prep_data(pd.read_csv(datapath), stack=stack,
                       lower=lower, random=False, nopreprocess=nopreprocess,
                       descriptions=desc, gibberish=gibberish, permutations=permute,
                       augment=augment, spelling=spellcheck, nonumbers=nonumbers)
        with torch.no_grad():
            results = predict(model, df)
        lines = []
        df = pd.read_csv(datapath)
        for i, result in enumerate(results[0]):
            line = "{},{},{},{}".format(df[TEXTCOL][i],
                                        df['sex'][i],
                                        df['age'][i],
                                        results[0][i])
            lines.append(line)
        if fastText:
            logger.warning("Generating an ensemble solution.")
            lines = ensemble_solution(lines, fastText_solution(df))
        if evaluate:
#            evaluate_model(model)
            with open(datapath) as f:
                gt = f.readlines()
            print("Score: {}".format(score(gt[1:], lines)))
            return
        with open(solutionpath, mode="w") as f:
            f.write("{},sex,age,event\n".format(TEXTCOL))
            for line in lines:
                f.write(line.upper() + "\n")


def evaluate_model(model, output_dir="/tmp", df=pd.read_csv("validset.csv")):
    """
    Evaluate a model using the TransformerModel method.
    """
    return TransformerModel.eval_model(model, df, output_dir=output_dir, verbose=True)


class FScore:
    """
    Object to hold an event class' TP, FP, and FN values.
    """
    def __init__(self):
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.tn = 0


def score(true, pred):
    """
    Score a model by the contest formula.  Expects true & pred to be a list of solution lines.
    """
    df = EVENTCOUNTS_DF
    labels = {}
    weights = {}
    total = 0
    for i, label in enumerate(df['label']):
        weights[label] = df['count'][i]
        labels[label] = FScore()
        total += weights[label]
    fp = tp = 0
    for gt, pr in zip(true, pred):
        if isinstance(gt, np.int64):
            true_label = gt
        elif isinstance(gt, int):
            true_label = gt
        else:
            true_label = int(gt.split(',')[-1][:-1])
        if isinstance(pr, int):
            pred_label = pr
        elif isinstance(pr, np.int64):
            pred_label = pr
        else:
            pred_label = int(pr.split(",")[-1][:-1])
        if true_label == pred_label:
            labels[true_label].tp += 1
        else:
            try:
                labels[true_label].fn += 1
                labels[pred_label].fp += 1
            except KeyError:
                #import pdb; pdb.set_trace()
                logger.error("Unrecognized event class {}".format(pred_label))
    sumF = 0
    sumW = 0
    for label in labels:
        if labels[label].tp == 0:
            f1 = 0
        else:
            precision = labels[label].tp / (labels[label].tp + labels[label].fp)
            recall = labels[label].tp / (labels[label].tp + labels[label].fn)
            f1 = precision * recall * 2 / (precision + recall)

        weight = labels[label].tp + labels[label].fn
        sumF += weight * f1
        sumW += weight
    return sumF / sumW
    return tp / (tp + fp)


if __name__ == "__main__":
    os.chdir(os.path.abspath(os.path.dirname(getsourcefile(lambda:0))))
    os.system("rm -rf cache_dir")
    plac.call(main)
    sys.exit()
#    train_data = prep_data()
#    model = build_model()
#    TransformerModel.train_model(model, train_data, output_dir="/data/cdc/checkpoints")
