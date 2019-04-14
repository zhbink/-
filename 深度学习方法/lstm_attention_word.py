import os
import pickle
import random
import jieba
import numpy as np

from keras import backend as K
from keras.optimizers import SGD
from keras.regularizers import l2
from gensim.models import Word2Vec
from keras.engine.topology import Layer
from keras.layers import Bidirectional
from keras.models import Model,Sequential,load_model
from keras.preprocessing.sequence import pad_sequences
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import initializers, regularizers, constraints
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers import Input, LSTM, Dense, merge, Flatten, Embedding, Dropout,Bidirectional

EMBEDDING_DIM = 100

class Attention(Layer):
    def __init__(self, step_dim=80,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        self.supports_masking = True
        #self.init = initializations.get('glorot_uniform')
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0
        super(Attention, self).__init__(**kwargs)
    def build(self, input_shape):
        assert len(input_shape) == 3
        self.W = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        self.features_dim = input_shape[-1]
        if self.bias:
            self.b = self.add_weight((input_shape[1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None
        self.built = True
    def compute_mask(self, input, input_mask=None):
        return None

    def call(self, x, mask=None):
        features_dim = self.features_dim
        step_dim = self.step_dim
        eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)), K.reshape(self.W, (features_dim, 1))), (-1, step_dim))
        if self.bias:
            eij += self.b
        eij = K.tanh(eij)
        a = K.exp(eij)
        if mask is not None:
            a *= K.cast(mask, K.floatx())
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)
    def compute_output_shape(self, input_shape):
        return input_shape[0],  self.features_dim
    
def get_tokenizer():

    with open('e:/2018汽车行业文本处理/train_data/tok_word.pkl','rb') as f:
        tokenizer=pickle.load(f)
    word_index = tokenizer.word_index
    #print(len(word_index)) #16885
    nb_words=min(50000,len(word_index)+1)

    return nb_words,word_index

def create_model():
    
    print('load data')
    nb_words,word_index=get_tokenizer()
    print('create model')
    word2vec = Word2Vec.load('e:/2018汽车行业文本处理/train_data/w2v_word.mod')
    embedding_matrix = np.zeros((nb_words, EMBEDDING_DIM))
    for word, i in word_index.items():
        if word in word2vec.wv.vocab:
            try:
                embedding_matrix[i] = word2vec.wv.word_vec(word)
            except:
                pass
    embedding_layer = Embedding(nb_words, EMBEDDING_DIM, input_length=80, weights=[embedding_matrix], trainable=True)
    model = Sequential()
    model.add(embedding_layer)
    model.add(LSTM(256,return_sequences=True,W_regularizer=l2(0.001)))
    model.add(Activation('relu'))#激活函数
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Attention())
    #全连接
    model.add(Dense(128))#添加128节点的全连接
    model.add(Activation('relu'))   #激活
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(11))            #添加输出节点
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
    data_1,data_2=np.load("./train_data/data_1_word.npy"),np.load("./train_data/data_2_1_word.npy")
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

    bst_model_path = 'weights_lstm_word'+str(i)
    early_stopping = EarlyStopping(monitor='val_loss', patience=10)
    model_checkpoint = ModelCheckpoint(bst_model_path, save_best_only=True, save_weights_only=True)
    data_train_1, data_train_2,data_test_1,data_test_2=np.array(data_train_1), np.array(data_train_2),np.array(data_test_1),np.array(data_test_2)        
    hist = model.fit(data_train_1, data_train_2,validation_data=(data_test_1,data_test_2), 
                     epochs=100, batch_size=50, shuffle=True, callbacks=[early_stopping, model_checkpoint], verbose=2)
    model.load_weights('weights_lstm_word'+str(i))
    model.save('./model/model_lstm_attention_word_'+str(i)+'.h5')

for j in range(1,10):
    train(j)
    
import lstm_attention_char
import cnn_char
import cnn_word
