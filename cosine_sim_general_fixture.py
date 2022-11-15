import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
# from main import cleandata, preprocess_stem, tokenizer
from scipy.spatial.distance import cosine

file_path_method = "resource/i_think_should_be_the_last_time/tfidf_methods.pkl"
file_path_all = "resource/i_think_should_be_the_last_time/tfidf_all.pkl"
df_method = pd.read_pickle(file_path_method)
df_all = pd.read_pickle(file_path_all)

df_setup_only_class_index = df_all.xs('setUp', level=1)
df_setup_only_class_index = df_setup_only_class_index.index
# df_setup_only_class_index = df_all.groupby(by=df_setup_only_class_index.index ,level=0)

disjoint_pairs_count = []
disjoint_pairs_total = []
# disjoint_pairs = []
df_disjoint_pairs = pd.DataFrame(columns=['class', 'pair_method', 'disjoint_score', 'total_disjoint_score',
                                          'total_disjoint_pair'])
for i, new_df in df_all.loc[df_setup_only_class_index].groupby(level=0):
    df_setup = new_df.loc[[(i, 'setUp')]]
    df_withOutSetup = new_df.drop(index='setUp', level=1)
    disjoint_count = 0
    pair_counter = 0
    for j in range(len(df_withOutSetup)):
        pairs_each_class = []
        for k in range(j + 1, len(df_withOutSetup)):
            pairs_each_class.append([df_withOutSetup.index.get_level_values(1)[j],
                                     df_withOutSetup.index.get_level_values(1)[k]])
            check_disjoint = 0
            pair_counter += 1
            if (cosine(j, k) == 0) and (cosine(j, df_setup) != 0) and (cosine(k, df_setup) != 0):
                check_disjoint = 1
                disjoint_count += 1
            class_pair_djScore = pd.DataFrame([[i, pairs_each_class, check_disjoint]],
                                              columns=['class', 'pair_method', 'disjoint_score'])
            disjoint_pairs_count.append(class_pair_djScore)
            check_disjoint = 0
            pairs_each_class = []
    class_totalDjScore = pd.DataFrame([[i, disjoint_count, pair_counter]], columns=['class', 'total_disjoint_score',
                                                                                    'total_pair_method'])
    disjoint_pairs_total.append(class_totalDjScore)
df_disjoint_pairs = pd.concat(disjoint_pairs_count).set_index('class')
df_disjoint_total = pd.concat(disjoint_pairs_total).set_index('class')
result = pd.merge(df_disjoint_pairs, df_disjoint_total, on='class')
# df_disjoint_total['total_disjoint_score'].sum()
# df_disjoint_pairs['disjoint_score'].sum()