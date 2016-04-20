import pandas as pd
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer
import string
from sklearn.feature_extraction.text import CountVectorizer
import lda
import numpy as np
import pickle
from collections import defaultdict
import cPickle as pk

__author__ = 'Naheed'

stop = stopwords.words('english')
stem = SnowballStemmer('english').stem
punc = string.punctuation
tokenizer = RegexpTokenizer('\s+', gaps=True)

for i in punc:
    stop.append(unicode(i))


def stemmer(sent):
    if type(sent) == float:
        return unicode(sent)
    else:
        return ' '.join([stem(word) for word in word_tokenize(sent.lower()) if word not in stop])


loaded = True   # True, If autoload self.df_all from MergedProductInfo csv file, False if i want to make it on the fly.
NUM_TOPICS = 50  # Number of Topics i want to extract Out of the whole corpus of documents.
n_top_words = 10  # Number of words per topic having most probability.

class Lda:
    '''
        Improvements/TO DO:
            1. Stop considering numeric words into vocabulary

            2. Fix the issue of some products have description but no attributes. Have to include them as well. May be use join instead of merge
                FIXED
    '''

    def __init__(self):
        self.bin = []
        self.df_all = []  # Dataframe of (produid,merged description,attribute)
        #self.topic_words = []  # List of List of String

        if not loaded:
            df_pro_desc = pd.read_csv('data/product_descriptions.csv')
            '''
            #df_attr = pd.read_csv('data/attributes.csv', encoding="ISO-8859-1")
            #groups = df_attr.groupby(['product_uid'])
            # print len(groups)

            ht = []
            for name, gr in groups:
                ht.append((name, ' '.join([i for i in gr['value'] if type(i) != float])))

            df_attr = pd.DataFrame(ht, columns=['product_uid', 'value'])
            '''

            # print df_pro_desc.shape

            #df_all = pd.merge(df_attr, df_pro_desc, how='right', on='product_uid')
            df_all = df_pro_desc
            print df_all.shape


            #print 'stemming values'
            #df_all['value'] = df_all['value'].map(lambda x: stemmer(x))
            print 'stemming prod desc'
            df_all['product_description'] = df_all['product_description'].map(lambda x: stemmer(x))

            #df_all['allaboutproduct'] = df_all[['product_description', 'value']].apply(lambda x: '\t'.join(x), axis=1)
            df_all.to_csv('Allproductinfo.csv',encoding = "ISO-8859-1")
            self.df_all = df_all.drop(['product_uid'], axis=1)

        else:
            df_all  = pd.read_csv('Allproductinfo.csv', encoding="ISO-8859-1")
            df_extracted = df_all[['product_uid','product_description']]

            df_extracted.to_csv('src/MergedProductinfo.csv',encoding = "ISO-8859-1")
            del df_extracted
            del df_all
            self.df_all = pd.read_csv('src/MergedProductinfo.csv', encoding="ISO-8859-1")


    def runlda(self):
        # vectorizer = CountVectorizer(encoding = "ISO-8859-1",analyzer = 'word',tokenizer = None,preprocessor= None)
        vectorizer = CountVectorizer(encoding="ISO-8859-1", analyzer='word', tokenizer=None, \
                                             preprocessor=None, max_features=15000)

        '''
        vectorizer = CountVectorizer(encoding="ISO-8859-1", analyzer='word', tokenizer=None, \
                                     preprocessor=None, max_features=500)
        '''

        feat = vectorizer.fit_transform(self.df_all['product_description'])

        features = feat.toarray()
        # print features.shape
        model = lda.LDA(n_topics=NUM_TOPICS, n_iter=50, random_state=1)

        model.fit(features)
        topic_word = model.topic_word_
        # print topic_word
        vocab = vectorizer.get_feature_names()
        # print vocab[:10]
        '''
        self.topic_words = [[] for i in range(NUM_TOPICS)]
        for i, topic_dist in enumerate(topic_word):
            self.topic_words[i] = np.array(vocab)[np.argsort(topic_dist)][:-n_top_words - 1:-1]
            # print('Topic {}: {}'.format(i, ' '.join(topic_words)))
        '''
        self.doc_topic = model.doc_topic_

        '''
        print("shape: {}".format(doc_topic.shape))
        for n in range(5):
            sum_pr = sum(doc_topic[n,:])
            print("document: {} sum: {}".format(n, sum_pr))
        '''
        self.bin = [[] for i in range(features.shape[0])]  # The topic bins to be used in Phase 2

        for n in range(features.shape[0]):
            topic_most_pr = self.doc_topic[n].argmax()
            # print("doc: {} topic: {}\n{}...".format(n, topic_most_pr, self.df_all['allaboutproduct'][n].encode("utf8")))
            self.bin[topic_most_pr].append(self.df_all['product_uid'][n])

        # Dump the bin into a pickle file to make the phase one independent of phase two
        with open('topicbins.pkl', 'wb') as fp:
            pickle.dump(self.bin, fp)

        # Dump the weights into a pickle file, which is an dictionary
        d_t = defaultdict(list)
        for n in range(features.shape[0]):
            uid = int(self.df_all['product_uid'][n])
            d_t[uid] = self.doc_topic[n]
        f = file('doc_topic.pkl', 'wb')
        pickle.dump(d_t, f)
