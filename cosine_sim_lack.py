import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
# from main import cleandata, preprocess_stem, tokenizer
from scipy.spatial.distance import cosine

# tfidf_no_stem = "resource/result_version_1/tf_idf/Lackcohesion_tfidf_df.csv"
tfidf_no_stem = "resource/result_version_1/tf_idf/demo_Lack_cohesion_tfidf_df.csv"
# tfidf_stemmed = "resource/tfidf_df_stem.csv"
# tf_no_stem = "resource/tf_df_no_stem.csv"
# tf_stemmed = "resource/tf_df_stemmed.csv"

# tf_df_no_stem = pd.read_csv(tf_no_stem,index_col=0)
# tf_df_stemmed = pd.read_csv(tf_stemmed,index_col=0)

tfidf_df_no_stem = pd.read_csv(tfidf_no_stem, index_col=0).dropna()
# tfidf_df_stemmed = pd.read_csv(tfidf_stemmed,index_col=0)

# Y = tfidf_df_no_stem.iloc[12]
max_length = len(tfidf_df_no_stem.index)


def mean():
    sim = 0
    for i in range(max_length):
        for j in range(max_length):
            if i<j:
                A = tfidf_df_no_stem.iloc[i]
                B = tfidf_df_no_stem.iloc[j]

                sim = sim + 1 - cosine(A, B)

    return sim / 100


1 - mean()
# X_tf = tf_df_no_stem.iloc[1]
# cosine_similarity(tf_df_no_stem)
# cosine_similarity(pd.DataFrame(X), pd.DataFrame(Y))

# cosine(X, Y)
# 1 - cosine(X, Y)
