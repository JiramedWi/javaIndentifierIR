import pandas as pd
import string
import requests
import numpy as np

setuoWithOutClass = 'resource/result_version_1/clean_data/methodNoSetup_concat_clean.csv'
setupWithClass = 'resource/result_version_1/clean_data/setup_with_class_clean_concat.csv'
# this is an old one */
# --------------------------------------------------------------

# tfidf_no_stem = "resource/tfidf_df_no_stem.csv"
# tfidf_stemmed = "resource/tfidf_df_stem.csv"
# tf_no_stem = "resource/tf_df_no_stem.csv"
# tf_stemmed = "resource/tf_df_stemmed.csv"
#
# tf_df_no_stem = pd.read_csv(tf_no_stem,index_col=0)
# tf_df_stemmed = pd.read_csv(tf_stemmed,index_col=0)
#
# tfidf_df_no_stem = pd.read_csv(tfidf_no_stem,index_col=0)
# tfidf_df_stemmed = pd.read_csv(tfidf_stemmed,index_col=0)

# --------------------------------------------------------------

reader = pd.read_csv(setuoWithOutClass, index_col=0, header=0)
reader2 = pd.read_csv(setupWithClass, index_col=0)