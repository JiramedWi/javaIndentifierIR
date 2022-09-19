import pandas as pd
import string
from string import digits
import requests
import numpy as np
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from collections import defaultdict
from gensim import corpora

javaKeywords = ["abstract", "continue", "for", "new", "switch", "assert", "default", "goto", "package", "synchronized",
                "boolean", "do", "if", "private", "this", "break", "double", "implements", "protected", "throw",
                "byte", "else", "elseif", "import", "public", "throws", "case", "enum", "instanceof", "return",
                "transient",
                "catch", "extends", "int", "short", "try",
                "char", "final", "interface", "static", "void",
                "class", "finally", "long", "strictfp", "volatile",
                "const", "float", "native", "super", "while"]
removeFormat = r'\b(?:{})\b'.format('|'.join(javaKeywords))
path = 'resource/test7.txt'
classFile = 'resource/result_version_1/class.txt'
methodFile = 'resource/result_version_1/method.txt'
setUpFile = 'resource/result_version_1/setUp.txt'
methodWithoutSetupFile = 'resource/result_version_1/methodWithoutSetup.txt'
demo_path = 'resource/result_version_1/demo testclass.csv'
demo = pd.read_csv(demo_path, header=0)
df2 = pd.read_table(setUpFile, header=None, names=['context'])
df = pd.read_csv(path, sep="@@", header=None, names=['type', 'context'])


def cleandata():
    # data = df[df['context'].notnull()]
    # data = data['context']
    data = demo['Method_detail']
    # split word with underscore
    clean_data = data.apply(lambda s: ' '.join(s.split('_')))
    # delete white space
    clean_data = clean_data.apply(
        lambda s: s.translate(str.maketrans(string.whitespace, ' ' * len(string.whitespace), '')))
    # split word with Capital letter
    clean_data = clean_data.apply(lambda s: ' '.join(re.sub(r"([A-Z])", r" \1", s).split()))
    # split word with numerical by Tulyakai approach
    clean_data = clean_data.apply(lambda s: ' '.join(re.split('(-?\d+\.?\d*)', s)))
    clean_data = clean_data.apply(lambda s: ''.join(i for i in s if not i.isdigit()))
    for word in digits:
        clean_data = clean_data.apply(lambda s: ' '.join(s.split(digits)))
    # remove special characters
    clean_data = clean_data.apply(
        lambda s: s.translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation))).replace(' ' * 4,
                                                                                                        ' ').replace(
            ' ' * 3, ' ').replace(' ' * 2, ' ').strip())
    # cast to lower
    clean_data = clean_data.apply(lambda s: s.lower())
    # split word with java keyword
    clean_data = clean_data.apply(lambda s: ''.join(re.sub(r"\b(%s)\b" % "|".join(javaKeywords), "", s)))
    # delete whitespace
    clean_data = clean_data.apply(lambda s: ' '.join(w.strip() for w in s.split()))
    # save in the file
    # clean_data.to_csv('resource/clean_data_not_concat.csv', index=False)
    # clean_data = pd.concat([df["type"], clean_data], axis=1)
    clean_data = pd.concat([demo["Class"], clean_data], axis=1)
    clean_data.to_csv('resource/clean_data_demo_test_class', index=False)
    # clean_data.to_csv('resource/clean_data_concat.csv', index=False)
    return clean_data


def cleandata_lack_of_cohesion():
    df_method= pd.read_csv('resource/result_version_1/clean_data/methodNoSetup.csv', header=0)
    df_setup= pd.read_csv('resource/result_version_1/clean_data/setup_clean_not_concat.csv', header=0)
    df = pd.concat([df_setup, df_method],axis=1)
    df['new'] = df[['setup', 'methodNoSetup']].astype(str).agg(' '.join,axis=1)
    df['new'] = df['new'].apply(lambda s: ''.join(re.sub(r"\bnan\b", "", s)))
    df.to_csv('resource/result_version_1/clean_data/setup&method.csv', index=False)
    return df


def preprocess_stem(s):
    ps = PorterStemmer()
    s = word_tokenize(s)
    stopwords_set = set(stopwords.words())
    stop_dict = {s: 1 for s in stopwords_set}
    s = [w for w in s if w not in stop_dict]
    s = [ps.stem(w) for w in s]
    s = ' '.join(s)
    return s


def tokenizer(data):
    list = []
    data.apply(lambda s: list.append([x.strip() for x in s.split()]))
    return list


cleandata()
