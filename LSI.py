import pandas as pd
from gensim.test.utils import common_dictionary, common_corpus
from gensim.models import LsiModel
from tf_idf import make_corpus

clean_data_path = 'resource/clean_data_not_concat.csv'
clean_df = pd.read_csv(clean_data_path, header=0, index_col=False)
clean_df = clean_df['context']

corpus, dict = make_corpus()

lsi_model = LsiModel(corpus, id2word=dict)
vectorized_corpus = lsi_model[corpus]  # vectorize input corpus in BoW format

topic = lsi_model.print_topics(num_topics=-1)
lsi_model.show_topics()
sent_topics_df = pd.DataFrame()

for i, row in enumerate(vectorized_corpus):
    # Sort in index of keyword
    row = sorted(row, key=lambda x: (x[1]), reverse=True)
    # Get dominant topic, Perc Contribution and Keyword for each document
    for j, (topic_num, prob_topic) in enumerate(row):
        if j == 0:  # => dominant topic
            wp = lsi_model.show_topic(topic_num)
            topic_keywords = " ".join([word for word, prob in wp])
            sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prob_topic, 4), topic_keywords]),
                                                   ignore_index=True)
        else:
            break

sent_topics_df.columns = ['dominant_topic', 'perc_contribution', 'topic_keywords']
dataframe = pd.concat([clean_df, sent_topics_df], axis=1)
dataframe.to_csv('resource\lsi.csv')
