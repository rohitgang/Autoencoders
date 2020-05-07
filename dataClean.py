import pandas as pd
import numpy as np
import random
import math
from collections import OrderedDict, defaultdict
import matplotlib.pyplot as plt
import seaborn as sns


def checkNAN(df):
    missVals = dict()
    cols = df.columns
    for col in cols:
        percentMissing = df[col].isna().sum() / len(df[col]) * 100.00
        if percentMissing > 0.00:
            missVals[col] = '{:5.4f}'.format(percentMissing) + '%'
    if len(missVals.keys()) == 0:
        return 'You got a clean Dataframe'
    return missVals

def getCatCols(df):
    ret = list()
    for col in df.columns:
        if df[col].dtypes == object:
            ret.append(col)
    return ret

def majorMissing(df):
    missVals = dict()
    cols = df.columns
    for col in cols:
        percentMissing = df[col].isna().sum() / len(df[col]) * 100.00
        if percentMissing > 50.00:
            missVals[col] = '{:5.4f}'.format(percentMissing) + '%'
    return missVals


def missColType(df):
    missType = dict()
    cols = df.columns
    missVals = checkNAN(df)
    for key in missVals.keys():
        missType[key] = df[key].dtype
    return missType


def graphDiscrete(df, col):
    xs = list(pd.unique(df[col]))
    vals = OrderedDict([(x, 0) for x in xs])
    for x in df[col]:
        if math.isnan(x):
            continue
        vals[x] += 1
    sns.set_style('dark')
    graph = plt.scatter(vals.keys(), vals.values())
    graph = plt.title('Discrete Data')
    graph = plt.xlabel(col)
    graph = plt.ylabel('Count')
    return graph


def getNumeric(df):
    dtypes = list()
    for col in df.columns:
        if col == 'Id':
            continue
        if str(df[col].dtype) == 'float64' or str(df[col].dtype) == 'int64':
            dtypes.append(col)
    return dtypes


def getCateg(df, cats):
    ret = defaultdict(list)
    for cat in cats:
        ret[cat] = [item for item in df[cat].unique()]
    return dict(ret)


def fixDiscrete(df, miss):
    for col in miss:
        median = df[col].median()
        df[col] = df[col].fillna(median)
    return df


# def getClassProp(row):
#     if math.isnan(row):
#         wow = 1
#     uniq[row] += 1


def fixCateg(df, cols):
    for col in cols:
        uniq = list(pd.unique(df[col].dropna()))
        #     uniq= uniq.remove('nan')
        prop = {key: math.floor((1 - (df[col] == key).sum() / len(df[col])) * 100.00) for key in uniq}
        classes = list(key for key, value in prop.items() for i in range(value))
        indexes = df[df[col].isnull()].index.tolist()
        for i in indexes:
            rand = random.randint(0, len(classes) - 1)
            df[col][i] = classes[rand]
    return df
