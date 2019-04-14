import os
import pickle
import random
import jieba
import numpy as np

from gensim.models import Word2Vec
from keras.models import Sequential
from keras.layers import Embedding
from keras.layers import LSTM
from keras.layers import Bidirectional
from keras.layers import Dense
from keras.layers import TimeDistributed
from keras.layers import Dropout
from keras_contrib.layers.crf import CRF
from keras.callbacks import EarlyStopping, ModelCheckpoint
#from keras_contrib.utils import save_load_utils
    
def get_tokenizer():

    with open('e:/2018汽车行业文本处理/train_data/tok_char.pkl','rb') as f:
        tokenizer=pickle.load(f)
    word_index = tokenizer.word_index
    #print(len(word_index)) #16885
    nb_words=min(50000,len(word_index)+1)

    return nb_words,word_index

def create_model():
    
    print('load data')
    nb_words,word_index=get_tokenizer()
    print('create model')
    word2vec = Word2Vec.load('e:/2018汽车行业文本处理/train_data/w2v_char.mod')
    embedding_matrix = np.zeros((nb_words, 100))
    for word, i in word_index.items():
        if word in word2vec.wv.vocab:
            try:
                embedding_matrix[i] = word2vec.wv.word_vec(word)
            except:
                pass
    embedding_layer = Embedding(nb_words, 100, input_length=130, weights=[embedding_matrix], trainable=True)

    model = Sequential()
    model.add(Embedding(nb_words, output_dim=100, input_length=130))
    model.add(Bidirectional(LSTM(128, return_sequences=True)))
    model.add(Dropout(0.5))
    model.add(Bidirectional(LSTM(128, return_sequences=True)))
    model.add(Dropout(0.5))
    model.add(TimeDistributed(Dense(11)))
    crf_layer = CRF(11)
    model.add(crf_layer)
    model.compile('rmsprop', loss=crf_layer.loss_function, metrics=[crf_layer.accuracy])
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
        for k in range(i*1000,9948):
            data_train_1.append(data_1[k])
            data_train_2.append(data_2[k])
    else:       
        for k in range((i-1)*1000,i*1000):
            data_test_1.append(data_1[k])
            data_test_2.append(data_2[k])
        for k in range(i*1000,9948):
            data_train_1.append(data_1[k])
            data_train_2.append(data_2[k])

    bst_model_path = 'weights_lstm_char'+str(i)
    early_stopping = EarlyStopping(monitor='val_loss', patience=10)
    model_checkpoint = ModelCheckpoint(bst_model_path, save_best_only=True, save_weights_only=True)
    data_train_1, data_train_2,data_test_1,data_test_2=np.array(data_train_1), np.array(data_train_2),np.array(data_test_1),np.array(data_test_2)        
    hist = model.fit(data_train_1, data_train_2,validation_data=(data_test_1,data_test_2), 
                     epochs=100, batch_size=50, shuffle=True, callbacks=[early_stopping, model_checkpoint], verbose=2)
    model.load_weights('weights_lstm_char'+str(i))
    model.save('./model/model_lstm_crf_char_'+str(i)+'.h5')

for j in range(1,10):
    train(j)

