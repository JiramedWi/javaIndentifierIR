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

filePath = 'resource/i_think_should_be_the_last_time/ClassAndMethod.txt'
dfMultiIndex = pd.read_csv(filePath, sep='@!@', header=None, names=["class", "method", "detail"])

def clean_detail():
    data = dfMultiIndex['detail']
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
    clean_data = pd.concat([dfMultiIndex['class'],dfMultiIndex['method'],clean_data], axis=1)
    clean_data.to_pickle("resource/i_think_should_be_the_last_time/CleanDetailDataframe.pkl")
    return clean_data