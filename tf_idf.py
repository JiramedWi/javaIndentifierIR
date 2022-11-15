import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from main import cleandata, preprocess_stem, tokenizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from collections import defaultdict
from gensim import corpora

filepath = 'resource/i_think_should_be_the_last_time/CleanDetailDataframe.pkl'
df = pd.read_pickle(filepath)
multiIndex_df = df.set_index(['class', 'method'])
setup = multiIndex_df.xs('setUp', level='method')
method = multiIndex_df.loc[multiIndex_df.index.get_level_values(1) != 'setUp']
detail_setup = setup['detail']
detail_method = method['detail']
detail_all_method = multiIndex_df['detail']

# methodSetup = 'resource/result_version_1/clean_data/methodNoSetup_concat_clean.csv'
# methodSetup_df = pd.read_csv(methodSetup, header=0, index_col=0)
# methodSetup_df_setup = methodSetup_df['setup'].fillna('')
# methodSetup_df_method = methodSetup_df['methodNoSetup'].fillna('')
# dict = pd.concat([methodSetup_df_setup])


# vectorizer = TfidfVectorizer(preprocessor=preprocess_stem, use_idf=True)
# vector_stemmed = vectorizer.fit_transform(clean_df)
# tfidf_df_stemmed = pd.DataFrame(vector_stemmed.toarray(), index=clean_df, columns=vectorizer.get_feature_names())


vectorizer_no_stem = TfidfVectorizer(use_idf=True, token_pattern=r'(?u)\b\w+\b')
vectorized_no_stem = vectorizer_no_stem.fit(detail_all_method)
vector_no_stem_setup = vectorized_no_stem.transform(detail_setup)
vector_no_stem_methods = vectorized_no_stem.transform(detail_method)
vector_no_stem_all = vectorized_no_stem.transform(detail_all_method)

vector_feature_name = vectorized_no_stem.get_feature_names_out()

tfidf_df_no_stem_method = pd.DataFrame(vector_no_stem_methods.toarray(), columns=vectorized_no_stem.get_feature_names_out())
tfidf_df_no_stem_method = tfidf_df_no_stem_method.set_index(
    [method.index.get_level_values(0), method.index.get_level_values(1)])
tfidf_df_no_stem_all = pd.DataFrame(vector_no_stem_all.toarray(), columns=vectorized_no_stem.get_feature_names_out())
tfidf_df_no_stem_all = tfidf_df_no_stem_all.set_index(
    [multiIndex_df.index.get_level_values(0), multiIndex_df.index.get_level_values(1)])

print(tfidf_df_no_stem_all)
tfidf_df_no_stem_method.to_pickle("resource/i_think_should_be_the_last_time/tfidf_methods.pkl")
tfidf_df_no_stem_all.to_pickle("resource/i_think_should_be_the_last_time/tfidf_all.pkl")
# tfidf_df_no_stem.to_csv('resource/result_version_1/tf_idf/tfidf_df_no_stem.csv', encoding='utf-8')

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
