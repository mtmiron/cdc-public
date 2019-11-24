"""
Code dealing with ensembles of models; much of it was written around the 24 hour mark, so it's not particularly elegant.
"""
import itertools
import time
import re
import glob
import random
import plac
import pandas as pd
import numpy as np
import os
import sys
import simpletransformersmodel as st
import logging
import preproc
import tensorflow as tf
import pickle
import math
import tqdm
from main import score

logging.basicConfig(level = logging.WARNING)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

prediction_cache = {}
prediction_cache_file = None


def nCr(n,k):
    f = math.factorial
    return f(n) / (f(k) * f(n-k))


def build_ensemble(models="ensemble.txt"):
    """
    Build a list of strings describing an ensemble.

    e.g. returns [('roberta', 'PATH/TO/CHECKPOINT', 'OPTIONS'), (...), (...), ...]
    """
    ensemble = []
    with open(models) as f:
        for line in f.readlines():
            # skip comments and blank lines
            if line.lstrip() == '' or line.lstrip()[0] == '#':
                continue
            arch = line.split(":")[0].strip()
            path = line.split(":")[1].strip()
            if len(line.split(":")) > 2:
                args = line.split(":")[2]
            else:
                args = ""
#            model = st.TransformerModel(arch, path, num_labels=100)
            ensemble.append((arch, path, args))
    return ensemble


def reverse_predict(ensemble, df):
    """
    Version of predict() that reverses the order of words in samples.
    """
    perm_weight = 1.0
    texts = list(df['text'])
    permdf = preproc.reverse(df)
    permtexts = list(permdf['text'])

    length = len(ensemble)
    total_logits = []
    while len(ensemble) > 0:
        modelargs = ensemble.pop()
        logger.warning("Model {}/{}: {}".format(length - len(ensemble), length, modelargs[0]))
        model = st.TransformerModel(modelargs[0], modelargs[1], num_labels=100)
        logits = model.predict(texts)[1]
        permlogits = model.predict(permtexts)[1]
        total_logits.append(logits)
        total_logits.append(permlogits)
        del model

    return logits_to_preds(np.array(total_logits))

def permute_predict(ensemble, df):
    """
    Version of predict() that randomly permutes the order of words in samples.
    """
    perm_weight = 1.0
    texts = list(df['text'])
    permdf = preproc.permute(df, concat=False)
    permtexts = list(permdf['text'])
#    logitary = []

    total_logits = []
    length = len(ensemble)
    while len(ensemble) > 0:
        modelargs = ensemble.pop()
        logger.warning("Model {}/{}: {}".format(length - len(ensemble), length, modelargs[0]))
        model = st.TransformerModel(modelargs[0], modelargs[1], num_labels=100)
        logits = model.predict(texts)[1]
        permlogits = model.predict(permtexts)[1]
        total_logits.append(logits)
        total_logits.append(permlogits)
        del model

    return logits_to_preds(np.array(total_logits))


def predict(ensemble, texts, nonumber_texts=None):
    """
    Predict using the list of models, passed as a list of descriptive strings in `ensemble`.

    e.g. ensemble == [('roberta', 'PATH/TO/CHECKPOINT', 'OPTIONS'), ...]
    """
    logits = []
    length = len(ensemble)
    ensemble = ensemble
#    start_time = time.time()
    starting_prediction_cache_length = len(prediction_cache)
    for i in range(length):
        modelargs = ensemble[i]
        if prediction_cache.get(modelargs, None) is not None:
            logits.append(prediction_cache[modelargs])
        else:
            logger.info("Model {}/{}: {}".format(length - len(ensemble), length, modelargs[0]))
            model = st.TransformerModel(modelargs[0].strip(), modelargs[1].strip(), num_labels=100)
            if 'nonumbers' in modelargs[2]:
                logger.warning("Passing samples with integers stripped to {}".format(modelargs[1]))
                predicted = model.predict(nonumber_texts)[1]
            else:
                predicted = model.predict(texts)[1]
            prediction_cache[modelargs] = predicted
            logits.append(predicted)
            del model

    if len(prediction_cache) > starting_prediction_cache_length:
        with open(prediction_cache_file, "wb") as f:
            try:
                pickle.dump(prediction_cache, f)
            except KeyboardInterrupt:
                f.seek(0)
                pickle.dump(prediction_cache, f)
                sys.exit()
#    logger.info("Prediction time: {}".format(time.time() - start_time))
    return logits_to_preds(np.array(logits))


def logits_to_preds(logits):
    """
    Convert lists of logits to event class predictions.
    """
    total_logits = np.array(logits[0])
    for i in range(1, len(logits)):
        total_logits += np.array(logits[i])
    preds = [total_logits[i].argmax() for i in range(len(total_logits))]
    return preds


def write_solution(lines):
    """
    Write out a solution.
    """
    with open("solution/solution.csv", "w") as f:
        f.write("text,sex,age,event\n")
        for line in lines:
            f.write(line + "\n")


def load_tried_set(datapath="ensemble_list.txt"):
    """
    Load the set of previously tried model combinations, to pick up
    where we left off.  Will load any file of the same format.
    """

    tried = set()
    if not os.path.exists(datapath):
        logger.warning("Could not find {} to load the tried set.".format(datapath))
        return set()
    with open(datapath) as f:
        for line in f.readlines():
            modelargs = line.split(":")[1]
            modelargs = re.findall('\([^)]+\)', modelargs)
            ensemble = []
            for arg in modelargs:
                triplet = arg.split(",")
                arch = triplet[0].replace("(", "").replace(")", "").replace("'", "").strip()
                path = triplet[1].replace("(", "").replace(")", "").replace("'", "").strip()
                opts = ""
                ensemble.append((arch,path,opts))
            tried.add(tuple(ensemble))
    return tried


def try_list(datapath, modelargs, outfile="ensemble_list.txt"):
    """
    Score a list of models, to find the best of a fixed number of combinations for ensembling.

    Expects `modelargs` to be a list of model descriptions, see predict().
    """
    df = pd.read_csv(datapath)
    df = preproc.preproc(df, lower=True)
    lines = []
    already_tried = load_tried_set(outfile)

    with open(datapath) as f:
        gt = f.readlines()[1:]
    while len(modelargs) > 0:
        lines.clear()
        ensemble = modelargs.pop()
        if len(set(ensemble)) != len(ensemble):
            logger.warning("Truncating ensemble: {}".format(str(ensemble)))
            ensemble = list(set(ensemble))

        results = predict(list(ensemble), list(df['text']), None)
        
        lines = []
        for i, result in enumerate(results):
            line = "{},{},{},{}".format(df['text'][i],
                                        df['sex'][i],
                                        df['age'][i],
                                        results[i])
            lines.append(line)
        
        en_score = score(gt, lines)
        print("{}: {}".format(en_score, str(ensemble)))
        with open(outfile, "a") as f:
            f.write("{}: {}\n".format(en_score, str(ensemble)))


def try_permutations(datapath, num_ensembles=8, outfile="tmp.txt"):
    """
    Brute force random walk over the list of available checkpoints.  Tries combinations at random
    to find the highest scoring ensemble.  Never returns.
    """
    DIRS = [ "cp/bert/*/pytorch_model.bin",
             "cp/roberta/*/pytorch_model.bin",
             "cp/xlnet/*/pytorch_model.bin",
             "cp/xlnet/09/checkpoint-21000",
             "cp/xlnet/07/checkpoint-42500",
             "cp/distilbert/*/pytorch_model.bin",
           ]
    score_hash = {}
    tried = load_tried_set(outfile)
    for models in tried:
        score_hash[tuple(set(models))] = 0.0
    setlist = [set(x) for x in tried]
    buf = []
    full_list = []
    df = pd.read_csv(datapath)
    df = preproc.preproc(df, lower=True)
    with open(datapath) as f:
        gt = f.readlines()[1:]
        gt = [int(x.split(',')[-1][:-1]) for x in gt]
    for d in DIRS:
        buf = glob.glob(d)
        for i, tmp in enumerate(buf):
            tmp = tmp.replace("/pytorch_model.bin", "")
            full_list.append(tmp)

    logger.warning("Models to permute: {}".format(str(full_list)))
    combo = itertools.combinations(full_list, num_ensembles)
    combo_list = list(combo)
    random.shuffle(combo_list)
    df_text_list = list(df['text'])
    progbar = tqdm.tqdm(combo_list, total=len(combo_list) - len(list(filter(lambda x: len(x) == num_ensembles, tried))), desc="HiScore: 0.0")
    hiscore = 0.0

    for paths in progbar:
        buf.clear()
        for path in paths:
            path = path.strip()
            if 'roberta' in path:
                buf.append(("roberta", path, ""))
            elif "distilbert" in path:
                buf.append(("distilbert", path, ""))
            elif 'bert' in path:
                buf.append(("bert", path, ""))
            elif 'xlnet' in path:
                buf.append(('xlnet', path, ''))
            else:
                logger.error("Epic fail, programmer.")
                sys.exit()
        buf_set = set(buf)
        buf_tuple = tuple(buf_set)
        if score_hash.get(buf_tuple, None) is not None:
            continue
#        if buf_set in setlist:
#            continue
#        setlist.append(buf_set)
#        if buf_tuple in tried:
#            continue
#        tried.add(buf_tuple)
        results = predict(buf_tuple, df_text_list, None)

#        lines = []
 #       for i, result in enumerate(results):
  #          line = ("%s,%d,%d,%d" % (df['text'][i],
   #                                 df['sex'][i],
    #                                df['age'][i],
     #                               results[i]))
      #      lines.append(line)
        
        en_score = score(gt, results)
        score_hash[buf_tuple] = en_score
        if en_score > hiscore:
            hiscore = en_score
            progbar.set_description("HiScore: {}".format(hiscore))
        #print("{}: {}".format(en_score, str(buf)))
#        if en_score > 0.83:
        with open(outfile, "a") as f:
            f.write("{}: {}\n".format(en_score, str(buf)))


def truncate_ensembles(tried):
    retlist = list()
    for ensemble in tried:
        retlist.append(tuple(set(ensemble)))
    return set(retlist)


def main(datapath: ("The CSV file to use as input", "option", "d", str)="valid5000.csv",
         permute: ("Permute the order of words in sentences randomly", "flag", "p")=False,
         reverse: ("Reverse the order of words in sentences", "flag", "r")=False,
         search: ("Do an exhaustive search to find the best ensemble of SEARCH models", "option", "s", int)=0,
         models: ("Path to a file with a list of models", "option", "m")=""):

    global prediction_cache_file
    global prediction_cache
    prediction_cache_file = datapath + ".prediction_cache.pickle"
    if os.path.exists(prediction_cache_file):
        with open(prediction_cache_file, "rb") as f:
            prediction_cache = pickle.load(f)
            logger.warning("Loaded prediction cache from {}".format(prediction_cache_file))
    
    if search > 0:
        try_permutations(datapath, num_ensembles=search, outfile="tmp.txt")
        sys.exit()
    elif models != "":
        already_tried = load_tried_set("tmp.txt")
        already_tried = truncate_ensembles(already_tried)
        to_try = load_tried_set(models)
        to_try = truncate_ensembles(to_try)
        ensembles = to_try - already_tried
        logger.info("List of {} ensembles to try: {}".format(len(ensembles), ensembles))
        try_list(datapath, ensembles, outfile="tmp.txt")
        sys.exit()

    ensemble = build_ensemble()
    df = pd.read_csv(datapath)
    df_nonums = pd.read_csv(datapath)
    df = preproc.preproc(df, lower=True)#, spelling=True)
    df_nonums = preproc.preproc(df_nonums, lower=True, nonumbers=True)

    if permute:
        results = permute_predict(ensemble, df)
    elif reverse:
        results = reverse_predict(ensemble, df)
    else:
        results = predict(ensemble, list(df['text']), list(df_nonums['text']))

    lines = []
    for i, result in enumerate(results):
        line = "{},{},{},{}".format(df['text'][i],
                                    df['sex'][i],
                                    df['age'][i],
                                    results[i])
        lines.append(line)
    write_solution(lines)

    with open(datapath) as f:
        gt = f.readlines()
    print("Score: {}".format(score(gt[1:], lines)))

if __name__ == "__main__":
    plac.call(main)
