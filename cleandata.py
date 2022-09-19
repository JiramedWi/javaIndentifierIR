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

classFile = 'resource/result_version_1/class.txt'
methodFile = 'resource/result_version_1/method.txt'
setupFile = 'resource/result_version_1/setUp.txt'
methodWithoutSetupFile = 'resource/result_version_1/methodWithoutSetup.txt'
class_and_setup_clean = 'resource/result_version_1/clean_data/setup_with_class_clean_concat.csv'
class_and_setup_clean_df = pd.read_table(class_and_setup_clean, sep=',',index_col=0)

setupData = pd.read_table(setupFile, header=None, names=['setup'])
classData = pd.read_table(classFile, header=None, names=['class'])
methodNoSetup = pd.read_table(methodWithoutSetupFile, header=None, names=['methodNoSetup'])
class_and_setup_df = pd.concat([classData['class'], setupData['setup']], axis=1)


def cle():
    data = class_and_setup_df['setup']
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
    clean_data.to_csv('resource/result_version_1/clean_data/setup_clean_not_concat.csv', index=False)
    clean_data = pd.concat([classData, clean_data], axis=1)
    clean_data.to_csv('resource/result_version_1/clean_data/setup_with_class_clean_concat.csv')
    return clean_data


def cle_allmethod():
    data = methodNoSetup['methodNoSetup']
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
    clean_data.to_csv('resource/result_version_1/clean_data/methodNoSetup.csv', index=False)
    clean_data = pd.concat([class_and_setup_clean_df, clean_data], axis=1)
    clean_data.to_csv('resource/result_version_1/clean_data/methodNoSetup_concat_clean.csv')
    return clean_data
