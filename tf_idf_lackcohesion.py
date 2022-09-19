import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

file = 'resource/result_version_1/clean_data/setup&method.csv'
demo = 'resource/clean_data_demo_test_class'
# df = pd.read_csv(file, header=0, index_col=False)
df = pd.read_csv(demo, header=0, index_col=False)
KmethAndSetup = df['Method_detail'].fillna('')

vectorizer_no_stem = TfidfVectorizer(use_idf=True)
vector_no_stem = vectorizer_no_stem.fit_transform(methAndSetup)
tfidf_df_no_stem = pd.DataFrame(vector_no_stem.toarray(), index=methAndSetup, columns=vectorizer_no_stem.get_feature_names())
tfidf_df_no_stem.shape
# tfidf_df_no_stem.to_csv('resource/result_version_1/tf_idf/Lackcohesion_tfidf_df.csv', encoding='utf-8')
tfidf_df_no_stem.to_csv('resource/result_version_1/tf_idf/demo_Lack_cohesion_tfidf_df.csv', encoding='utf-8')