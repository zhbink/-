# -*- coding: utf-8 -*-
"""
Created on Mon Nov 26 20:58:20 2018

@author: zhaohaibo
"""

import pandas as pd
import numpy as np
import jieba
import jieba.analyse
from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer, TfidfTransformer,TfidfVectorizer

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold,StratifiedKFold
from scipy.sparse import csr_matrix, hstack
from sklearn.metrics import f1_score,accuracy_score,classification_report
import sklearn.metrics as metrics

from sklearn.preprocessing import LabelEncoder
from collections import Counter
import pickle as pk
from sklearn import svm
from sklearn.ensemble import VotingClassifier
from sklearn import tree
from sklearn.svm import SVC
from sklearn import svm
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression 
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier 


import os
os.chdir('C:/Users/zhaohaibo/Desktop/汽车观点主题预测/data/')


def get_data():
    train = pd.read_csv('train.csv')
    test = pd.read_csv('test_public.csv')

    train = train.sample(frac=1)
    train = train.reset_index(drop=True)
    
    data = pd.concat([train, test])
    
    lbl =  LabelEncoder()
    lbl.fit(train['subject'])
    nb_classes = len(list(lbl.classes_))
    
    pk.dump(lbl, open('label_encoder.sav','wb'))

    subject = lbl.transform(train['subject'])

    y = []
    for i in list(train['sentiment_value']):
        y.append(i)

    y1= []
    for i in subject:
        y1.append(i)

    # print(np.array(y).reshape(-1,1)[:,0])
    return data,train.shape[0],np.array(y).reshape(-1,1)[:,0],test['content_id'],np.array(y1).reshape(-1,1)[:,0]


def processing_data(data):
    word = jieba.cut(data)
    return ' '.join(word)

def pre_process():
  
    data,nrw_train,y,test_id,y1 = get_data()

    data['cut_comment'] = data['content'].map(processing_data)
    
    print('TfidfVectorizer...')
    tf = TfidfVectorizer(ngram_range=(1,2),analyzer='char')
    discuss_tf = tf.fit_transform(data['cut_comment'])

    print('HashingVectorizer...')
    ha = HashingVectorizer(ngram_range=(1,1),lowercase=False)
    discuss_ha = ha.fit_transform(data['cut_comment'])

    data = hstack((discuss_tf,discuss_ha)).tocsr()

    return data[:nrw_train],data[nrw_train:],y,test_id,y1


X,test,y,test_id,y1= pre_process()


N = 10
kf = StratifiedKFold(n_splits=N, random_state=2018).split(X,y)

#clf = LogisticRegression(C=0.5)
#clf = svm.LinearSVC(loss='hinge', tol=1e-4, C=0.6)
#clf = VotingClassifier(estimators=[
#    ('svm_clf', SVC(kernel = 'linear',probability=True)),
#    ('log_clf', LogisticRegression())
#    ],voting='soft')

clf = XGBClassifier(learning_rate =0.2)

y_train_oofp = np.zeros_like(y, dtype='float64')
y_train_oofp1 = np.zeros_like(y, dtype='float64')

y_test_oofp = np.zeros((test.shape[0], N))
y_test_oofp_1 = np.zeros((test.shape[0], N))



def micro_avg_f1(y_true, y_pred):
    return metrics.f1_score(y_true, y_pred, average='micro')


acc = 0
vcc = 0
for i ,(train_fold,test_fold) in enumerate(kf):
    X_train, X_validate, label_train, label_validate,  label_1_train, label_1_validate,= X[train_fold, :], X[test_fold, :], y[train_fold], y[test_fold], y1[train_fold], y1[test_fold]
    clf.fit(X_train, label_train)
    
    val_ = clf.predict(X_validate)
    y_train_oofp[test_fold] = val_
    print('sentiment_value_f1:%f' % micro_avg_f1(label_validate, val_))
    acc += micro_avg_f1(label_validate, val_)
    result = clf.predict(test)
    y_test_oofp[:, i] = result

    clf.fit(X_train, label_1_train)
    val_1 = clf.predict(X_validate)
    y_train_oofp1[test_fold] = val_

    vcc += micro_avg_f1(label_1_validate, val_1)
    result = clf.predict(test)
    y_test_oofp_1[:, i] = result

print(acc/N)
print(vcc/N)


lbl = pk.load(open('label_encoder.sav','rb'))
res_2 = []
for i in range(y_test_oofp_1.shape[0]):
    tmp = []
    for j in range(N):
        tmp.append(int(y_test_oofp_1[i][j]))
    word_counts = Counter(tmp)
    yes = word_counts.most_common(1)
    res_2.append(lbl.inverse_transform(yes[0][0]))
    
    
res = []
for i in range(y_test_oofp.shape[0]):
    tmp = []
    for j in range(N):
        tmp.append(y_test_oofp[i][j])
    res.append(max(set(tmp), key=tmp.count))
    
    
    
print(len(res))
result = pd.DataFrame()
result['content_id'] = list(test_id)

result['subject'] = list(res_2)
result['subject'] = result['subject']

result['sentiment_value'] = list(res)
result['sentiment_value'] = result['sentiment_value'].astype(int)

result['sentiment_word'] = ''
result.to_csv('submit4.csv',index=False)    
