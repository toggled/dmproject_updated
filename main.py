from __future__ import division
from src.Lda import Lda
from src.SecondPhase import SecondPhase
from collections import defaultdict
import cPickle as pk
import pandas
import numpy as np
from time import time
from sklearn.cross_validation import KFold
from sklearn.metrics import mean_squared_error
import csv
import math


__author__ = 'Naheed'

df_dir = 'data/'

# if equals 0, use random forest, if equals 1, use gaussian process regression
MODEL_TYPE = 1
# Ture means for running on the fly(usually for the first time), False for running on the local stored data.
realtime = True
# if weighted = 0, chose the model with highest probability, it weighted = 1, use weighted sum
WEIGHTED = 1

K_fold = 2
topic_bins = []
load_pickled = False  # Whether want to load pickled bins from pk file

TOPIC_BINS_BASE = 'topicbins'
DOC_TOPIC_BASE = 'doc_topic'
#BASEDIR = 'RF_vartopic(5-200)' # folder for storing topic sensitivity results
BASEDIR = 'varsample' # folder for storing sample size sensitivity result
#BASEDIR = 'test'
INC_TOPICS = 5
INC_ITER = 10
run = None
runf = lambda x:[str(x),''][x==None]

def run_phaseone():
    topicbins_path = BASEDIR+'/'+TOPIC_BINS_BASE+str(run)+'.pkl'
    docbins_path = BASEDIR+'/'+DOC_TOPIC_BASE+str(run)+'.pkl'
    if load_pickled:
        import pickle
        with open(topicbins_path) as f:
            topic_bins = pickle.load(f)

    else:
        ld = Lda(topicbins_path,docbins_path)
        ld.runlda(n_topics = 20+run*INC_TOPICS, n_iteration = 30, max_feat = 15000)
        topic_bins = ld.bin  # List of List(product_id)

    for i in topic_bins:
        if i:
            print len(i)


def run_phasetwo():
    topicbins_path = BASEDIR+'/'+TOPIC_BINS_BASE+str(run)+'.pkl'
    docbins_path = BASEDIR+'/'+DOC_TOPIC_BASE+str(run)+'.pkl'
    process = SecondPhase()
    timezero = time()
    num_train = 74067
    df_all = pandas.read_csv(df_dir + 'my_df_all.csv')
    # df_all = df_all.drop(['id','search_term', 'product_title', 'product_description', 'product_info', 'attr', 'brand'],
    #                      axis=1)

    """ Assume here that all the df_all only contain the train.csv data,
        since we don't use the test.csv for our project"""

    df = df_all[:num_train].drop(['Unnamed: 0'], axis=1)
    # df = df_all.drop(['Unnamed: 0'], axis = 1)

    models = defaultdict(object)
    errors = []
    # weight_d_t is the documenet-topic probability, which is a list
    f = file(docbins_path,'rb')
    weight_d_t = pk.load(f)
    f1 = file(topicbins_path, 'rb')
    bins = pk.load(f1)
    newbins = []
    total = 0

    for bin in bins:
        if len(bin) > 0:
            newbins.append(bin)
            total+=len(bin)
    print 'number of non-empty bins: ',len(newbins)
    print 'total products: ',total

    ######################
    ## Cross Validation ##
    ######################

    kf = KFold(df.shape[0], n_folds=K_fold)
    iteration = 0
    perc_confidence = []
    ## each iteration contains one fold CV, and the result is the RSME for this iteration
    for train_index, test_index in kf:
        result_matrix = []
        variance_matrix= []

        print '\nIteration ', iteration, ' starts'
        train_set = df.iloc[train_index]
        test_set = df.iloc[test_index]

        uids = test_set['product_uid'].values

        time0 = time()
        train_bin = process.train_data_construct(newbins, train_set, iteration, realtime)
        test_data = process.test_data_construct(test_set)
        print 'train bins prepared, time used: ', time() - time0
        time0_0 = time()
        print 'start to train models'
        for i in range(0, len(train_bin.keys())):
            time0 = time()
            X_train = train_bin[i][0]

            # xx = X_train.drop(['product_uid','id','relevance'],axis=1)
            try:
                # X_train = X_train.drop(['product_uid','Unnamed: 0','relevance'],axis=1).values
                X_train = X_train.drop(['product_uid', 'id', 'relevance'], axis=1).values
            except:
                print i
                continue

            y_train = train_bin[i][1].values

            clf = process.select_model(MODEL_TYPE,X_train,y_train)
            if MODEL_TYPE == 0:
                clf.fit(X_train, y_train)
            else:
                clf.BuildModel(model='sparse')

            # try using Gaussian Processing Regression:
            # if MODEL_TYPE == 1:
            #     clf = GPregression(X_train, y_train)
            #     clf.BuildModel(model='full')
            # else:
            #     clf.fit(X_train, y_train)


            models[i] = clf
            if MODEL_TYPE == 1:
                mean,var = clf.predict(test_data[0].values)
                result = mean.flatten()
                variance = var.flatten()
            else:
                result = clf.predict(test_data[0].values)

            result_matrix.append(result)
            if MODEL_TYPE == 1:
                variance_matrix.append(variance)
            print 'model ', i , ' trained and predicted, time used: ', time() - time0
        if MODEL_TYPE == 0:
            y_predicted = process.weighted_sum(np.matrix(result_matrix), uids, weight_d_t, WEIGHTED)
        else:
            y_predicted = process.weighted_sum(np.matrix(result_matrix), uids, weight_d_t, WEIGHTED)
            wt_variance_matrix = process.var_weighted_sum(np.matrix(variance_matrix),uids,weight_d_t)

        error = np.sqrt(mean_squared_error(y_predicted, test_data[1]))

        errors = errors + [error]

        print 'Iteration ', iteration, ': All models trained, time used:', time() - time0_0
        iteration += 1

        with open(BASEDIR+'/result'+runf(run)+'.csv', 'a') as f:
            j = 0
            conf_int_count = 0
            writer = csv.writer(f)
            for i in test_index:
                if MODEL_TYPE == 1:
                    sd = math.sqrt(wt_variance_matrix[j])
                    ll = [i, test_set['id'][j:j+1], weight_d_t[uids[j]], np.argmax(weight_d_t[uids[j]]), y_predicted[j], test_data[1].values[j], abs(y_predicted[j] - test_data[1].values[j]) \
                            ,wt_variance_matrix[j],(y_predicted[j]+wt_variance_matrix[j],y_predicted[j] - wt_variance_matrix[j])]
                    writer.writerow(ll)
                    if test_data[1].values[j] < y_predicted[j]+ 2*sd and test_data[1].values[j]> y_predicted[j] - 2*sd:
                        conf_int_count += 1
                else:
                    ll = [i, test_set['id'][j:j+1], weight_d_t[uids[j]], np.argmax(weight_d_t[uids[j]]), y_predicted[j], test_data[1].values[j], abs(y_predicted[j] - test_data[1].values[j])]
                    writer.writerow(ll)

                j += 1
            if MODEL_TYPE == 1:
                print conf_int_count, ' ', j, conf_int_count*100/j
                perc_confidence.append(conf_int_count*100/j)
    error_f = np.mean(errors)
    conf_f = np.mean(perc_confidence)

    print '\nJOB DONE: the ', K_fold, ' fold Cross Validation has completed, time used: ', time() - timezero
    print 'The mean of RMSE is: ', error_f
    return (error_f,conf_f)


if __name__ == '__main__':
    run = -4
    THRESHOLD  = 5
    rmse_run = []
    perc_confidence = []
    import os
    while run < THRESHOLD :
        #CURDIR = BASEDIR+str(run)
        if not os.path.exists(BASEDIR):
            os.mkdir(BASEDIR)
        #run_phaseone()

        er,conf = run_phasetwo()
        rmse_run.append(er)
        perc_confidence.append(conf)

        run+=1
    print rmse_run, perc_confidence

    run = -4
    with open(BASEDIR+'/'+BASEDIR+'.txt','a') as f:
        for item in zip(rmse_run,perc_confidence):
            f.write(str(50+run*INC_ITER)+' '+str(item[0])+' '+str(item[1])+"%"+'\n')
            run+=1
            if run == 0:
                INC_ITER = 50


