import pandas as pd
import numpy as np
import re
import jieba

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression 
from sklearn.naive_bayes import MultinomialNB
from sklearn.multiclass import OneVsRestClassifier

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline

from sklearn import svm
from sklearn.naive_bayes import MultinomialNB

import os
os.chdir('C:\\Users\\zhaohaibo\\Desktop')


def split(data):
    data = data.reset_index(drop=True)
    split_content = data.content.apply(lambda x:re.sub(' ','',x))
    split_content = split_content.apply(lambda x:re.sub('\xa0','',x))
    split_content = split_content.astype('str').apply(lambda x: jieba.lcut(x))
    # 导入停用词
    stop = pd.read_csv('stop.txt', encoding = 'utf-8', sep = 'zhao', header = None,engine = 'python') #sep:分割符号（需要用一个确定不会出现在停用词表中的单词）
    document = []
    for i in split_content.index:
        temp = [k for k in split_content[i] if k not in stop.values]
        strr = ' '.join(temp)
        document.append(strr)
        if(i % 1000 == 0):
            print("Complete ---  {} / {} ".format(i, len(split_content)))
    return document



print("(1) load texts...")
X = pd.read_csv('train.csv', usecols=[0,1])
y = pd.read_csv('train.csv',usecols=[2])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=666, shuffle = False)# shuffle默认为True
X_test = X_test.reset_index(drop=True)
y_train = y_train.reset_index(drop=True)
y_test = y_test.reset_index(drop=True)
# X_train = pd.read_csv('train.csv', usecols=[1])
# y_train = pd.read_csv('train.csv',usecols=[2])
# X_test = pd.read_csv('test_public.csv',usecols=[1])
train_texts = split(X_train)
test_texts = split(X_test)
train_labels = y_train
test_labels = y_test

all_text = train_texts + test_texts

print ("(2) doc to var...")
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer   
count_v0= CountVectorizer();  
counts_all = count_v0.fit_transform(all_text);
count_v1= CountVectorizer(vocabulary=count_v0.vocabulary_);  
counts_train = count_v1.fit_transform(train_texts);   
print ("the shape of train is {}".format(counts_train.shape))
count_v2 = CountVectorizer(vocabulary=count_v0.vocabulary_);  
counts_test = count_v2.fit_transform(test_texts);  
print ("the shape of test is {}".format(counts_test.shape))
  
tfidftransformer = TfidfTransformer();    
train_data = tfidftransformer.fit(counts_train).transform(counts_train);
test_data = tfidftransformer.fit(counts_test).transform(counts_test); 

x_train = train_data
y_train = train_labels
x_test = test_data
y_test = test_labels


def GridsearchCV():
#    param_grid = [
#        {
#           'learning_rate':[0.01,0.05,0.1],
#           'n_estimators':[i for i in range(3000,5000,500)]
#        }]
#    clf =  XGBClassifier()
#    grid_search = GridSearchCV(clf, param_grid,n_jobs=-1)
#    grid_search.fit(X_train,y_train)
    param_grid = { 'max_depth':range(3,10,2), 'min_child_weight':range(1,6,2)}# param_grid = {#  'max_depth':[7,8],#  'min_child_weight':[4,5]# }
    gsearch1 = GridSearchCV(estimator = XGBClassifier(),param_grid=param_grid,cv=5)
    gsearch1.fit(X_train, y_train)
    grid_search.best_score_
    grid_search.best_estimator_


print ("(3) XGBoost...")
from xgboost import XGBClassifier
model = XGBClassifier(learning_rate =0.2)
model.fit(x_train, y_train)
preds = model.predict(x_test)
num = 0
preds = preds.tolist()
for i,pred in enumerate(preds):
   if(pred == y_test.subject[i]):
        num += 1
print ("\n\n precision_score:{}".format(float(num) / len(preds)))


preds = pd.Series(preds)
X_test['subject'] = pd.Series(preds)
X_test['sentiment_value'] = 0
X_test['sentiment_word'] = None
X_test = X_test.drop(['content'],axis=1)
X_test.to_csv('result_svm.csv',index=False)






        




