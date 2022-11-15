import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from main import cleandata
from main import preprocess_stem
import nltk

'''When you new of this project, run this below first'''
'''
nltk.download('punkt')
nltk.download('stopwords')
'''

filepath = 'resource/i_think_should_be_the_last_time/CleanDetailDataframe.pkl'
df = pd.read_pickle(filepath)
multiIndex_df = df.set_index(['class', 'method'])
setup = multiIndex_df.xs('setUp', level='method')
method = multiIndex_df.loc[multiIndex_df.index.get_level_values(1) != 'setUp']
detail_setup = setup['detail']
detail_method = method['detail']
# setup = df.groupby(['class'],['method']).
# methodSetup = 'resource/result_version_1/clean_data/methodNoSetup_concat_clean.csv'
# methodSetup_df = pd.read_csv(methodSetup, header=0, index_col=0)
# methodSetup_df_setup = methodSetup_df['setup'].fillna('')
# methodSetup_df_method = methodSetup_df['methodNoSetup'].fillna('')
# dict = pd.concat([methodSetup_df_setup])
# dict = detail
#

cv = CountVectorizer(token_pattern=r'(?u)\b\w+\b')
'''Create term with all word as a term'''
cv = cv.fit(multiIndex_df['detail'])
#
X_fit_setup = cv.transform(detail_setup)
X_fit_method = cv.transform(detail_method)
X_fit_all = cv.transform(multiIndex_df['detail'])
#
cv_feature_name = cv.get_feature_names_out()
setup_tf = pd.DataFrame(X_fit_setup.toarray())
method_tf = pd.DataFrame(X_fit_method.toarray(), columns=cv_feature_name)
all_method_tf = pd.DataFrame(X_fit_all.toarray())
all_method_tf = all_method_tf.set_index([multiIndex_df.index.get_level_values(0),multiIndex_df.index.get_level_values(1)])
# print(setup_tf)
# print(method_tf)
