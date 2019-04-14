import pickle
import numpy as np

from keras.models import Model
from keras.layers import Embedding
from gensim.models import Word2Vec
from keras.layers import Dense, Input, Flatten
from keras.layers.normalization import BatchNormalization
from keras.layers import Conv1D, MaxPooling1D, Embedding, Merge, Dropout, LSTM, GRU, Bidirectional, TimeDistributed

from keras import initializers
from keras import backend as K
from keras.optimizers import SGD
from keras.engine.topology import Layer, InputSpec
from keras.callbacks import EarlyStopping, ModelCheckpoint

MAX_SENT_LENGTH = 130
MAX_SENTS = 1
EMBEDDING_DIM = 100

class AttLayer(Layer):
    def __init__(self, attention_dim):
        self.init = initializers.get('normal')
        self.supports_masking = True
        self.attention_dim = attention_dim
        super(AttLayer, self).__init__()
    def build(self, input_shape):
        assert len(input_shape) == 3
        self.W = K.variable(self.init((input_shape[-1], self.attention_dim)))
        self.b = K.variable(self.init((self.attention_dim, )))
        self.u = K.variable(self.init((self.attention_dim, 1)))
        self.trainable_weights = [self.W, self.b, self.u]
        super(AttLayer, self).build(input_shape)
    def compute_mask(self, inputs, mask=None):
        return mask
    def call(self, x, mask=None):
        uit = K.tanh(K.bias_add(K.dot(x, self.W), self.b))
        ait = K.dot(uit, self.u)
        ait = K.squeeze(ait, -1)
        ait = K.exp(ait)
        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            ait *= K.cast(mask, K.floatx())
        ait /= K.cast(K.sum(ait, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        ait = K.expand_dims(ait)
        weighted_input = x * ait
        output = K.sum(weighted_input, axis=1)
        return output
    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])

def get_tokenizer():

    with open('e:/2018汽车行业文本处理/train_data/tok_char.pkl','rb') as f:
        tokenizer=pickle.load(f)
    word_index = tokenizer.word_index
    nb_words=min(50000,len(word_index)+1)

    return nb_words,word_index

def create_model():

    print('load data')
    nb_words,word_index=get_tokenizer()
    print('create model')
    word2vec = Word2Vec.load('e:/2018汽车行业文本处理/train_data/w2v_char.mod')
    embedding_matrix = np.zeros((nb_words, EMBEDDING_DIM))
    for word, i in word_index.items():
        if word in word2vec.wv.vocab:
            try:
                embedding_matrix[i] = word2vec.wv.word_vec(word)
            except:
                pass
    embedding_layer = Embedding(nb_words, EMBEDDING_DIM, input_length=30, weights=[embedding_matrix], trainable=True)

    sentence_input = Input(shape=(MAX_SENT_LENGTH,), dtype='int32')
    embedded_sequences = embedding_layer(sentence_input)
    l_lstm = Bidirectional(LSTM(128, return_sequences=True))(embedded_sequences)
    l_lstm=(Dropout(0.5))(l_lstm)
    l_lstm=BatchNormalization()(l_lstm)
    l_att = AttLayer(100)(l_lstm)
    sentEncoder = Model(sentence_input, l_att)

    review_input = Input(shape=(MAX_SENTS, MAX_SENT_LENGTH), dtype='int32')quit
    review_encoder = TimeDistributed(sentEncoder)(review_input)
    l_lstm_sent = Bidirectional(LSTM(128, return_sequences=True))(review_encoder)
    l_lstm_sent=(Dropout(0.5))(l_lstm_sent)
    l_lstm_sent=BatchNormalization()(l_lstm_sent)
    l_att_sent = AttLayer(100)(l_lstm_sent)

    dense1=Dense(128,activation='relu')(l_att_sent)
    dense1=(Dropout(0.5))(dense1)
    dense1=BatchNormalization()(dense1)
    preds = Dense(11, activation='softmax')(dense1)
    model = Model(review_input, preds)

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
            data_train_1.append([data_1[k]])
            data_train_2.append(data_2[k]) 
        for k in range((i-1)*1000,i*1000):
            data_test_1.append([data_1[k]])
            data_test_2.append(data_2[k])
        for k in range(i*1000,9948):
            data_train_1.append([data_1[k]])
            data_train_2.append(data_2[k])
    else:       
        for k in range((i-1)*1000,i*1000):
            data_test_1.append([data_1[k]])
            data_test_2.append(data_2[k])
        for k in range(i*1000,9948):
            data_train_1.append([data_1[k]])
            data_train_2.append(data_2[k])

    bst_model_path = 'weights_2lstm_hatt_char'+str(i)
    early_stopping = EarlyStopping(monitor='val_loss', patience=3)
    model_checkpoint = ModelCheckpoint(bst_model_path, save_best_only=True, save_weights_only=True)
    data_train_1, data_train_2,data_test_1,data_test_2=np.array(data_train_1), np.array(data_train_2),np.array(data_test_1),np.array(data_test_2)        
    hist = model.fit(data_train_1, data_train_2,validation_data=(data_test_1,data_test_2), 
                     epochs=100, batch_size=50, shuffle=True, callbacks=[early_stopping, model_checkpoint], verbose=2)
    model.load_weights('weights_2lstm_hatt_char'+str(i))
    model.save('./model/model_2lstm_hatt_char_'+str(i)+'.h5')

for j in range(1,10):
    train(j)


    
