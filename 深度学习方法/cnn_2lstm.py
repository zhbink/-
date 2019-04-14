import os
import pickle
import random
import jieba
import numpy as np

from keras import backend as K
from keras.regularizers import l2
from keras.optimizers import SGD
from keras.utils import np_utils
from gensim.models import Word2Vec
from keras.engine.topology import Layer
from keras.models import Model,Sequential,load_model
from keras.preprocessing.sequence import pad_sequences
from keras import initializers, regularizers, constraints
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers import Bidirectional,Input, LSTM, Dense, merge, Conv1D, MaxPooling1D, Flatten, Embedding, Dropout

EMBEDDING_DIM = 100
    
def get_tokenizer():

    with open('./train_data/tok_char.pkl','rb') as f:
        tokenizer=pickle.load(f)
    word_index = tokenizer.word_index
    #print(len(word_index)) #151317
    nb_words=min(50000,len(word_index)+1)
    return nb_words,word_index

def create_model():
    
    print('load data')
    nb_words,word_index=get_tokenizer()
    print('create model')
    word2vec = Word2Vec.load('./train_data/w2v_char.mod')
    embedding_matrix = np.zeros((nb_words, EMBEDDING_DIM))
    for word, i in word_index.items():
        if word in word2vec.wv.vocab:
            try:
                embedding_matrix[i] = word2vec.wv.word_vec(word)
            except:
                pass

    embedding_layer = Embedding(nb_words, EMBEDDING_DIM, input_length=130, weights=[embedding_matrix], trainable=True)
    model = Sequential()
    model.add(embedding_layer)
    model.add(Conv1D(512, 6))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(BatchNormalization())
    model.add(MaxPooling1D())
    model.add(Dropout(0.5))
    model.add(BatchNormalization())
    #model.add(LSTM(256,return_sequences=True,W_regularizer=l2(0.001)))
    #model.add(Activation('relu'))#激活函数
    #model.add(Dropout(0.5))  
    #model.add(BatchNormalization())
    model.add(LSTM(256,W_regularizer=l2(0.001)))#指定输入数据的形状
    model.add(Activation('relu'))#激活函数
    model.add(Dropout(0.5))  
    model.add(BatchNormalization())
    #全连接
    model.add(Dense(128))           #添加128节点的全连接
    model.add(Activation('relu'))   #激活
    model.add(Dropout(0.5))
    model.add(Dense(11))            #添加输出38个节点
    model.add(Activation('softmax')) 
    
    sgd = SGD(lr=0.02, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy',
                    optimizer=sgd,
                    metrics=['accuracy'])
    return model

def train(i):

    print('train',i)
    model=create_model()
    data_train_1, data_train_2,data_test_1,data_test_2=[],[],[],[]
    data_1,data_2=np.load("./train_data/data_1_char.npy"),np.load("./train_data/data_2_1_char.npy")
    if i-1!=0:
        for k in range(0,(i-1)*1000):
            data_train_1.append(data_1[k])
            data_train_2.append(data_2[k]) 
        for k in range((i-1)*1000,i*1000):
            data_test_1.append(data_1[k])
            data_test_2.append(data_2[k])
        for k in range(i*1000,9940):
            data_train_1.append(data_1[k])
            data_train_2.append(data_2[k])
    else:       
        for k in range((i-1)*1000,i*1000):
            data_test_1.append(data_1[k])
            data_test_2.append(data_2[k])
        for k in range(i*1000,9940):
            data_train_1.append(data_1[k])
            data_train_2.append(data_2[k])

    bst_model_path = 'weights_cnn_2lstm_char'+str(i)
    early_stopping = EarlyStopping(monitor='val_loss', patience=5)
    model_checkpoint = ModelCheckpoint(bst_model_path, save_best_only=True, save_weights_only=True)
    data_train_1, data_train_2,data_test_1,data_test_2=np.array(data_train_1), np.array(data_train_2),np.array(data_test_1),np.array(data_test_2)        
    hist = model.fit(data_train_1, data_train_2,validation_data=(data_test_1,data_test_2), 
                     epochs=100, batch_size=50, shuffle=True, callbacks=[early_stopping, model_checkpoint], verbose=2)
    model.load_weights('weights_cnn_2lstm_char'+str(i))
    model.save('./model/model_cnn_2lstm_char'+str(i)+'.h5')
for j in range(1,11):
    train(j)

#cnn(512)+lstm(256)+lstm(128)+dense(128) dropout:0.5 0.5 0.5 0.2 0.6105 90
#cnn(512)+lstm(256)+dense(128) dropout:0.5 0.5 0.5  0.6020 43
[0.6,0.6205,0.596,0.6075,0.5865,0.603,0.603,0.6065,0.595,0.6045]
#0.5570
