import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from main import cleandata, preprocess_stem, tokenizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from collections import defaultdict
from gensim import corpora
methodSetup = 'resource/result_version_1/clean_data/methodNoSetup_concat_clean.csv'
methodSetup_df = pd.read_csv(methodSetup, header=0, index_col=0)
methodSetup_df_setup = methodSetup_df['setup'].fillna('')
methodSetup_df_method = methodSetup_df['methodNoSetup'].fillna('')
dict = pd.concat([methodSetup_df_setup])



# vectorizer = TfidfVectorizer(preprocessor=preprocess_stem, use_idf=True)
# vector_stemmed = vectorizer.fit_transform(clean_df)
# tfidf_df_stemmed = pd.DataFrame(vector_stemmed.toarray(), index=clean_df, columns=vectorizer.get_feature_names())


vectorizer_no_stem = TfidfVectorizer(use_idf=True)
vectorizer_no_stem = vectorizer_no_stem.fit(dict)
vector_no_stem = vectorizer_no_stem.fit_transform(methodSetup_df_method)
tfidf_df_no_stem = pd.DataFrame(vector_no_stem.toarray(), index=methodSetup_df_method, columns=vectorizer_no_stem.get_feature_names())
tfidf_df_no_stem.shape
print(tfidf_df_no_stem)
tfidf_df_no_stem.to_csv('resource/result_version_1/tf_idf/tfidf_df_no_stem.csv', encoding='utf-8')

# To inspect the first document
# first_vectorizer = vector[0]
# df = pd.DataFrame(first_vectorizer.T.todense(), index=vectorizer.get_feature_names(), columns=["tfidf"])
# df = df.sort_values(by=['tfidf'], ascending=False)
# print(df)


# def make_corpus():
#     LSI_dict = tokenizer(clean_df)
#     dict = corpora.Dictionary(LSI_dict)
#     corpus = [dict.doc2bow(doc) for doc in LSI_dict]
#     # For check the index tokens and document
#     # print('Number of unique tokens: %d' % len(dict))
#     # print('Number of document: %d' % len(corpus))
#     return corpus, dict

