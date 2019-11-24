import csv, re, sys
import traceback
 
def load_annotations(filename):
    ret = []
    with open(filename, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            event = row['event']
            ret.append(event)
    return ret
 
 
def evaluate(gt, pred, provisional_set, fname):
 
    assert len(gt) == len(pred), 'The lines of groundtruth and prediction mismatch!'
 
    label_set = set(gt)
    for event in pred:
        assert event in label_set, 'Unknown event code in your prediction!'
 
    truths, preds = [], []
    for index, event in enumerate(gt):
        if (fname == 'Provisional') == (index in provisional_set):
            truths.append(event)
    for index, event in enumerate(pred):
        if (fname == 'Provisional') == (index in provisional_set):
            preds.append(event)
 
    assert len(truths) == len(preds)
 
    sumF1, sumW = 0, 0
    for label in label_set:
        mat = [[0, 0], [0, 0]]
        for (truth, pred) in zip(truths, preds):
            mat[int(truth == label)][int(pred == label)] += 1
 
        if mat[1][1] == 0:
            f1 = 0
        else:
            precision = mat[1][1] / (mat[1][1] + mat[0][1])
            recall = mat[1][1] / (mat[1][1] + mat[1][0])
            f1 = precision * recall * 2 / (precision + recall)
 
        weight = mat[1][1] + mat[1][0]
 
        sumF1 += weight * f1
        sumW += weight
 
    weightedF1 = sumF1 / sumW
    return weightedF1
