#!/usr/bin/env python3
import pandas as pd
import re
import logging
import tqdm
import vars as conf
import random
import spellchecker
from aspell import Aspell

TEXTCOL = conf.TEXTCOL
LABELCOL = conf.LABELCOL
ABBREV_MAP = conf.ABBREV_MAP

logger = logging.getLogger(__name__)


def reverse(df):
    texts = []
    for sent in tqdm.tqdm(df[TEXTCOL], desc="Reversing"):
        words = sent.split(" ")
        i = len(words) - 1
        line = ""
        while i >= 0:
            line += words[i] + " "
            i -= 1
        texts.append(line)
    
    cols = {TEXTCOL: texts}
    for col in df.columns:
        if col == TEXTCOL:
            continue
        cols[col] = df[col]
    return pd.DataFrame(cols)


def correct_spelling(df, ignore_errors=False):
    aspell = Aspell()
    corrected = []
    for sent in tqdm.tqdm(df[TEXTCOL], desc="Spellchecking"):
        corrected.append(aspell.correct(sent))

    cols = {TEXTCOL: corrected}
    for col in df.columns:
        if col == TEXTCOL:
            continue
        cols[col] = df[col]
    return pd.DataFrame(cols)


def replace_abbreviations(df, column=TEXTCOL):
    df[column].replace(ABBREV_MAP, regex=True, inplace=True)#list(ABBREV_MAP.keys()), list(ABBREV_MAP.values()), regex=True, inplace=True)
    return df


def expand_numbers(df):
    df[TEXTCOL].replace(r'\b(\d\d)(\w+)\b', r'\g<1> \g<2>', regex=True, inplace=True)
    return df


def squeeze(df):
    df[TEXTCOL].replace(r'\s\s\s\s+(?!DX|D X|dx|d x|DIAGNOSIS|diagnosis|Diagnosis)', 'DIAGNOSIS', regex=True, inplace=False)
    df[TEXTCOL].replace(r'\s\s+', ' ', regex=True, inplace=True)
    return df


def lowercase(df):
    cols = {'text': [txt.lower() for txt in df['text']]}
    for col in df.columns:
        if col == 'text':
            continue
        cols[col] = df[col]
    return pd.DataFrame(cols)


def add_diagnoses(df):
    diagnoses = []
    for text in df[TEXTCOL]:
        match = re.search(r'(DX|D X|dx|d x)\s?(.+)', text)
        if match:
            diagnoses.append(match.group(2))
        else:
            match = re.search(r'\s\s\s\s+(.+)', text)
            if match:
                diagnoses.append(match.group(1))
        if not match:
            diagnoses.append(None)
    if LABELCOL in df.columns:
        df = pd.DataFrame({TEXTCOL: df[TEXTCOL],
                           'sex': df['sex'],
                           'age': df['age'],
                           'diag': diagnoses,
                           LABELCOL: df[LABELCOL]})
    else:
        df = pd.DataFrame({TEXTCOL: df[TEXTCOL],
                           'sex': df['sex'],
                           'age': df['age'],
                           'diag': diagnoses})
    return replace_abbreviations(df, 'diag')


def add_descriptions(df):
    desc = pd.read_csv("code_descriptions.csv")
    evmap = {99: "Unclassifiable"}
    for i,ev in enumerate(desc[LABELCOL]):
        evmap[ev] = desc['desc'][i]
    if 'diag' in df.columns:
        df = pd.DataFrame({TEXTCOL: df[TEXTCOL], 'sex': df['sex'],
                           'age': df['age'], LABELCOL: df[LABELCOL],
                           'desc': [evmap[x].upper() for x in df[LABELCOL]],
                           'diag': df['diag']})
    else:
        df = pd.DataFrame({TEXTCOL: df[TEXTCOL], 'sex': df['sex'],
                           'age': df['age'], LABELCOL: df[LABELCOL],
                           'desc': [evmap[x].upper() for x in df[LABELCOL]]})
    return df


def permute(df, times=1, concat=True):
    permutations = []
    ages = []
    sexes = []
    events = []
    for i, text in enumerate(df[TEXTCOL]):
        for _ in range(times):
            sent = ""
            words = text.split(" ")
            while len(words) > 0:
                idx = random.randint(0, len(words) - 1)
                sent += " " + words[idx]
                del words[idx]
            permutations.append(sent[1:])
            sexes.append(df['sex'][i])
            ages.append(df['age'][i])
            events.append(df[LABELCOL][i])
    perm = pd.DataFrame({TEXTCOL: permutations, 'sex': sexes, 'age': ages, LABELCOL: events})
    if concat:
        return pd.concat([df, perm])
    else:
        return perm


def preproc(df, stack=False, lower=False, random=False, nopreprocess=False,
            diagnoses=False, descriptions=False, gibberish=0, permutations=False,
            augment=False, spelling=False, reverse=False, nonumbers=False):
    if stack:
        copy = df.copy(deep=True)
    if nopreprocess:
        logger.warning("Not performing data preprocessing.")
    else:
        logger.info("Preprocessing data.")
        if diagnoses:
            df = add_diagnoses(df)
        df = expand_numbers(df)
        df = squeeze(df)
        df = replace_abbreviations(df)
    if descriptions:
        df = add_descriptions(df)
    if lower:
        logger.info("Lowercasing data.")
        df = lowercase(df)
    if reverse:
        df = reverse(df)
    if permutations:
        logger.info("Permuting data.")
        df = permute(df)
    if stack:
        logger.info("Stacking data.")
        if descriptions:
            df = pd.concat([df, copy, pd.DataFrame({TEXTCOL: df['desc'], LABELCOL: df[LABELCOL]})], ignore_index=True, sort=False)
        else:
            df = pd.concat([df, copy], ignore_index=True, sort=False)
    if spelling:
        df = correct_spelling(df)
    if random:
        logger.info("Shuffling data.")
        df = df.sample(frac=1).reset_index(drop=True)
    if nonumbers:
        df[TEXTCOL].replace({r'\d+': ''}, regex=True, inplace=True)
    return df


def maxcounts(df):
    maxlen = 0
    maxtokens = 0
    maxtext = ""
    for text in df[TEXTCOL]:
        if len(text) > maxlen:
            maxlen = len(text)
            maxtext = text
            maxtokens = len(re.findall(r'\S+', text))
    return maxtext, maxlen, maxtokens
