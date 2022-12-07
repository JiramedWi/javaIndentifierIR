from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import os


class tf_idf_dump_to_pickle:
    def __init__(self, file_path):
        self.save_path = Path(os.path.abspath('')) / 'resource/' / 'v_3_of_Java_extraction'
        self.df = pd.read_pickle(file_path)
        self.multi_index = self.df.set_index(['class', 'method'])
        self.setup = self.multi_index.xs('setUp', level='method')
        self.test_method = self.multi_index.loc[self.multi_index.index.get_level_values(1) != 'setUp']
        self.setup_detail = self.setup['detail']
        self.test_method_detail = self.test_method['detail']
        self.all_method_detail = self.multi_index['detail']
        vectorizer = TfidfVectorizer(use_idf=True, token_pattern=r'(?u)\b\w+\b')
        self.vectorizer_all_method = vectorizer.fit(self.all_method_detail)
        self.get_feature_name = self.vectorizer_all_method.get_feature_names_out()

    def tf_idf(self):
        tranform_with_test_methods = self.vectorizer_all_method.transform(self.test_method_detail)
        tranform_with_all_methods = self.vectorizer_all_method.transform(self.all_method_detail)
        tf_idf_test_methods = pd.DataFrame(tranform_with_test_methods.toarray(), columns=self.get_feature_name)
        tf_idf_test_methods = tf_idf_test_methods.set_index([self.test_method.index.get_level_values(0),
                                                             self.test_method.index.get_level_values(1)])
        tf_idf_test_methods.to_pickle(self.save_path/'tfidf_test_methods.pkl')
        tf_idf_all_methods = pd.DataFrame(tranform_with_all_methods.toarray(), columns=self.get_feature_name)
        tf_idf_all_methods = tf_idf_all_methods.set_index([self.all_method_detail.index.get_level_values(0),
                                                           self.all_method_detail.index.get_level_values(1)])
        tf_idf_all_methods.to_pickle(self.save_path/'tfidf_all_method.pkl')
        tf_idf_setup = tf_idf_all_methods.xs('setUp', level=1)
        tf_idf_setup.to_pickle(self.save_path/'tfidf_setup.pkl')
        return tf_idf_setup, tf_idf_test_methods, tf_idf_all_methods


if __name__ == '__main__':
    s = tf_idf_dump_to_pickle('resource/v_3_of_Java_extraction/CleanDetailDataframe.pkl')
    df = s.tf_idf()
