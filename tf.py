import pandas
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from main import cleandata
from main import preprocess_stem
import nltk

# nltk.download('punkt')
# nltk.download('stopwords')
methodSetup = 'resource/result_version_1/clean_data/methodNoSetup_concat_clean.csv'
methodSetup_df = pd.read_csv(methodSetup, header=0, index_col=0)
methodSetup_df_setup = methodSetup_df['setup'].fillna('')
methodSetup_df_method = methodSetup_df['methodNoSetup'].fillna('')
dict = pd.concat([methodSetup_df_setup])

cv = CountVectorizer()
cv = cv.fit(dict)

# X_fit_setup = cv.transform(methodSetup_df_setup)
X_fit_method = cv.transform(methodSetup_df_method)

cv_feature_name = cv.get_feature_names_out()
# setup_tf = pd.DataFrame(X_fit_setup.toarray(), columns=cv_feature_name)
method_tf = pd.DataFrame(X_fit_method.toarray(), columns=cv_feature_name)
# print(setup_tf)
print(method_tf)
