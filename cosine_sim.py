import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
# from main import cleandata, preprocess_stem, tokenizer
from scipy.spatial.distance import cosine

tfidf_no_stem = "resource/result_version_1/tf_idf/tfidf_df_no_stem.csv"

tfidf_df_no_stem = pd.read_csv(tfidf_no_stem, index_col=0)

# Y = tfidf_df_no_stem.iloc[12]
# X = tfidf_df_no_stem.iloc[11]
max_length = len(tfidf_df_no_stem.index)
a = 0

for x in range(max_length):
    if a < max_length-1:
        A = tfidf_df_no_stem.iloc[x]
        B = tfidf_df_no_stem.iloc[x + 1]
        a = a + 1
        print(1 - cosine(A, B))
        # result = (1 - cosine(A, B))/
        # print(1 - cosine(A, B), a, x, x+1)
    print("done")

# X_tf = tf_df_no_stem.iloc[1]
# cosine_similarity(tf_df_no_stem)
# cosine_similarity(pd.DataFrame(X), pd.DataFrame(Y))

# cosine(X, Y)
# 1 - cosine(X, Y)
