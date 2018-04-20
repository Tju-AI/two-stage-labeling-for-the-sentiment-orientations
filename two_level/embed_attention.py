# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 14:07:14 2017

@author: yt
"""

import yaml
import pickle
import multiprocessing
import numpy as np
import pandas as pd
from gensim.models.word2vec import Word2Vec
from gensim.corpora.dictionary import Dictionary
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.models import Model
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM 
from keras import callbacks
#from keras import backend as K
from sklearn.cross_validation import train_test_split
from keras.layers import Bidirectional,Input,Flatten,concatenate,multiply,RepeatVector,Permute,TimeDistributed,Lambda,merge
from keras.layers.core import Dense, Dropout,Activation
#from keras.models import model_from_yaml,load_model
np.random.seed(1500)  # For Reproducibility
import re
#import random
#import pandas as pd
import sys
#import csv
import importlib
import jieba
importlib.reload(sys)
sys.setrecursionlimit(1000000)
# set parameters:
size=20
par=0.8
vocab_dim = 100
maxlen = 15
n_iterations = 1  # ideally more..
min_count = 3
window_size = 5
batch_size = 32
n_epoch = 30
input_length = 15
cpu_count = multiprocessing.cpu_count()
filtrate = re.compile(u'[^\u4E00-\u9FA5，。,.？!?！]')
#加载训练文件
def loadfile(path2):
#    data=pd.read_excel(path1,index=None)
    phone=pd.read_excel(path2,index=None)
    count=len(phone['word'])
    phone_word=list(phone['word'])
    phone_score=list(phone['score'])
    deny_word=list(phone['deny_word'])
    inter_word=list(phone['inter_word'])
    assume_word=list(phone['assume_word'])
#    comment = np.array(data.loc[:,['comment']])
#    comment = comment.tolist()
#    comment_1 = [sample[0] for sample in comment] 
#    label = np.array(data.loc[:,['score']])
    return count,phone_word,phone_score,deny_word,inter_word,assume_word

def load_w2v_file(path):
    fr = open(path,'rb')  
    data1 = pickle.load(fr)
    score = pickle.load(fr)
    char = pickle.load(fr)
    score=np.array(score)  
    fr.close()   
    return data1,score,char
def list_word(comment):
    com=[]
    for i in range(len(comment)):
        string=comment[i]
        word=[]
        for sample in string:
            word=word+list(sample)
        com.append(word)
    return com
def pre_clean(comment):
    filtrate1 = re.compile(u'[^\u4E00-\u9FA5，。,.？!?！:：]')
    comment_done=[] 
    for i,t in enumerate(comment):
        if str(t) != 'nan':
            b=re.sub('\n', '。', t, count=0, flags=0)
            b=re.sub(' ', '', b, count=0, flags=0)           
            b = filtrate1.sub(r'', b)
            comment_done.append(b)
    return comment_done
#构造标签
def label(length,num):
    array = []
    for i in range(length):
        array.append(num)
    return array

def tokenizer(text):
    text = [jieba.lcut(document) for document in text]
    return text

def create_dictionaries(model=None,combined=None):

    if (combined is not None) and (model is not None):
        gensim_dict = Dictionary()
        gensim_dict.doc2bow(model.wv.vocab.keys(),allow_update=True)
        w2indx = {v: k+1 for k, v in gensim_dict.items()}#所有频数超过10的词语的索引
        w2vec = {word: model[word] for word in w2indx.keys()}#所有频数超过10的词语的词向量
        
        #句子转化为索引
        def parse_dataset(combined):
            ''' Words become integers
            '''
            data=[]
            for sentence in combined:
                new_txt = []
                for word in sentence:
                    try:
                        new_txt.append(w2indx[word])
                    except:
                        new_txt.append(0)
                data.append(new_txt)
            return data
                
        combined=parse_dataset(combined)
        combined= sequence.pad_sequences(combined, maxlen=maxlen)#索引长度对齐
#        combined= zero_pad(combined, maxlen)#索引长度对齐
        return w2indx, w2vec,combined
    else:
        print ('No data provided...')


#创建词语字典，并返回每个词语的索引，词向量，以及每个句子所对应的词语索引
def word2vec_train(combined,length):
    model = Word2Vec(size=vocab_dim,
                     min_count=min_count,
                     window=window_size,
                     workers=cpu_count,
                     iter=n_iterations)
    model.build_vocab(combined)
    model.train(combined,total_examples=length,epochs=50)
    model.save('model1/Word2vec_model.pkl')
    index_dict, word_vectors,combined = create_dictionaries(model=model,combined=combined)
    index_dicted= sorted(index_dict.items(), key=lambda d:d[1], reverse = False)
    a=np.random.normal(size=100,loc=0,scale=0.05)
    word_vectors['UNK']=a
    f = open('lstm_data/Word2vec_ci_sansum.txt', 'w', encoding='utf-8')
    for dic in index_dicted:
        string = ''
        for i in range(len(word_vectors[dic[0]])):
            string = string + ' ' +str(word_vectors[dic[0]][i])
        f.write(dic[0] + ' ' + string + '\n')
    f.close
    return   index_dict, word_vectors,combined

def get_w2v_weight():
    f = open('lstm_data/Word2vec_sansum.txt', 'r',encoding='utf-8')
    index_dict = {}
    word_vectors ={}
    word2vec = f.readlines()
    for i in range(len(word2vec)):
        one = word2vec[i].split()
        index_dict[one[0]]=i+1
        vec = np.array(one[1:],dtype='float32')
        word_vectors[one[0]] = vec
    f.close()
    return index_dict,word_vectors

def add_feature(sen):
    feature2=np.zeros((1,8*size))
    if sen in phone_word:
        dex=phone_word.index(sen)
        score=phone_score[dex]
        if score==0:
            feature2[0][2*size:3*size]=par#23
        else:
            if score==0.25:
                feature2[0][3*size:4*size]=par#34
            else:
                if score==0.75:
                    feature2[0][4*size:5*size]=par
                else:feature2[0][5*size:6*size]=par # 56     
    if sen in deny_word:
        feature2[0][1*size:2*size]=par
    if sen in inter_word:
        feature2[0][0:size]=par
    if sen in assume_word:
         feature2[0][6*size:7*size]=par#67
    else:feature2[0][7*size:8*size]=par#78
    return feature2
def char_feature(string):
    feature3=np.zeros((1,8*size))
    character=['n','a','v','r','d','p','w','ad']
    if string in character:
        ind=character.index(string)
        feature3[0][ind*size:(ind+1)*size]=par
    return feature3 
    
def embedding_zi(combine,char_):
    asp_weight = np.zeros((len(combine),maxlen,8*size))
    embed_weight=np.zeros((len(combine),maxlen,vocab_dim+8*size))
    for i,sentence in enumerate(combine):
        if i%100==0:
            print(i,end=',')
#        senten=jieba.lcut(sentence)
        j=0
        for n,word in enumerate(sentence):
            char_f=char_feature(char_[i][n])
#            word_f=add_feature(word)
#            feature=np.concatenate((word_f,char_f),axis = 1)
            for zi in list(word):
#                print(j,end=',')
                try:
                    embed_weight[i][j][:]=np.concatenate((word_vectors[zi],char_f[0]),axis = 0)
                    asp_weight[i][j][:] = char_f
                except:
                    embed_weight[i][j][:]=np.concatenate((word_vectors['UNK'],char_f[0]),axis = 0)
                    asp_weight[i][j][:] = char_f
                j=j+1
                if j == maxlen:
                    break
            if j == maxlen:
                break
    return embed_weight ,asp_weight     
def embedding_word(combine,char_):
#    w=[]
    asp_weight = np.zeros((len(combine),maxlen,8*size))
    embed_weight=np.zeros((len(combine),maxlen,vocab_dim+8*size))
#    index_dict,word_vectors = get_w2v_weight()
#    feature1=np.zeros((1,size))
    for i,sentence in enumerate(combine):
        if i%100==0:
            print(i,end=',')
#        senten=jieba.lcut(sentence)
        j=0
        for n,word in enumerate(sentence):
            char_f=char_feature(char_[i][n])
#            word_f=add_feature(word)
#            feature=np.concatenate((word_f[0],char_f[0]),axis = 0)
            try:
                embed_weight[i][j][:]=np.concatenate((word_vectors[word],char_f[0]),axis = 0)
                asp_weight[i][j][:] = char_f
            except:
                embed_weight[i][j][:]=np.concatenate((word_vectors['UNK'],char_f[0]),axis = 0)
                asp_weight[i][j][:] = char_f
            j=j+1
            if j == maxlen:
                break
        
    return embed_weight ,asp_weight  
def split_train_test(embed_weight,asp_weight,y):
    x_train,x_test,asp_train,asp_test, y_train, y_test = train_test_split(embed_weight,asp_weight,y, test_size=0.2)
    print(x_train.shape,y_train.shape)
    return x_train,x_test,asp_train,asp_test, y_train, y_test
#ASPECT ATTENTION
def train_lstm2(x_train,asp_train,y_train,asp_test,x_test,y_test):
    print(x_train.shape,y_train.shape)
    print('Defining a Simple Keras Model...')
    a = Input(shape=(maxlen,100+8*size))
    b = Input(shape=(maxlen,8*size))
    print('a:',np.shape(a))
    print('b:',np.shape(b))
    activations1 = Bidirectional(LSTM(units=50,return_sequences=True))(a)
#    print(np.shape(x))
    activations = Bidirectional(LSTM(units=50,return_sequences=True))(activations1)
    act = Bidirectional(LSTM(units=50))(activations1)
    print('l:',np.shape(act))
#    activations =LSTM(units=50,return_sequences=True)(a)
    print('lstm:',np.shape(activations)
    hid_b =concatenate([activations,b])
    print('hid_b:',np.shape(hid_b))
    attention1 = Dense(1, activation='softmax')(hid_b)
    
    print('attention1:',np.shape(attention1))
    attention2 = Flatten()(attention1)
    print('attention2:',np.shape(attention2))
    attention3 = Activation('softmax')(attention2)
    attention4 = RepeatVector(100)(attention3)
    print('attention4:',np.shape(attention4))
    attention = Permute([2, 1])(attention4)
    print('attention:',np.shape(attention))
    attention_mul = multiply([activations, attention])
    
    print('merge:',np.shape(attention_mul))
    at_done = Flatten()(attention_mul)
    print('out:',np.shape(at_done))
    output = Dropout(0.3)(at_done)
    output = Dense(1)(output)
    output = Activation('sigmoid')(output)
    print('output:',np.shape(output))
    model = Model(inputs=[a,b], outputs=output)
    print('Compiling the Model...')
    model.compile(loss='mean_squared_error',
                  optimizer='adam',metrics=['accuracy'])
    print("Train...")
    earlyStopping = callbacks.EarlyStopping(monitor='val_loss', patience=4, verbose=1, mode='auto')
    saveBestModel = callbacks.ModelCheckpoint('lstm_data/vision7_word.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
    model.fit([x_train,asp_train], y_train, batch_size=batch_size, epochs=n_epoch,verbose=1, validation_data=([x_test,asp_test], y_test),callbacks=[earlyStopping, saveBestModel]) 
    output = Model(inputs=a, outputs=act)
    output.save('lstm_data/output_100.h5')
def read_test_data(path):
    data=pd.read_csv(path,encoding='gbk')
    comment =np.array(data.loc[:,['comment']])
#    label = np.array(data.loc[:,['score']])
    comment = comment.tolist()
#    label = label.tolist()
    comment_1 = [sample[0] for sample in comment]
#    label_1 = [sample[0] for sample in label]
    return comment_1

def test_data_result(comment,y):
    def get_label(score):
        label_test = []
        for i in range(len(score)):
            if score[i][0] <=0.45:
                label_test.append(0)
            if score[i][0] >0.45 and score[i][0] <0.55:
                label_test.append(0.5)
            if score[i][0] >=0.55:
                label_test.append(1)
        return label_test
#    commented = [list(document) for document in comment]    
    embed_weight,asp_weight=embedding_word(comment)
    print('loading model......')
    model = load_model('lstm_data/sen_sansum_que.h5')
    score = model.predict([embed_weight,asp_weight])
    label_test = get_label(score)
    label_true = get_label(y)
    print('writing data1...')  
    sava_data = pd.DataFrame(comment, columns=['comment'])
    sava_data['true_score'] = y
    sava_data['test_score'] = score
    sava_data['label_true'] = label_true
    sava_data['label_test'] = label_test
    sava_data.to_excel('result/2test_result_2.25.xlsx',index=None)
    return comment,score

   
if __name__=='__main__':
    print('Loading Data...')
#    count,phone_word,phone_score,deny_word,inter_word,assume_word=loadfile('data/phone/phone_word.xlsx')
    combine_w2v,y,char=load_w2v_file('data/travel/train_sin_travel.txt')
#    combine_w2v_list=list_word( combine_w2v)
    print(len(combine_w2v),len(y))
    print('Training a Word2vec model...')
    index_dict, word_vectors,combined=word2vec_train(combine_w2v,len(combine_w2v))
    print('Embedding...')
    embed_weight,asp_weight=embedding_word(combine_w2v,char)
#    embed_weight,asp_weight=embedding_zi(combine_w2v,char)
    print('Setting up Arrays for Keras Embedding Layer...')
    x_train,x_test,asp_train,asp_test, y_train, y_test=split_train_test(embed_weight,asp_weight,y)
    train_lstm2(x_train,asp_train,y_train,asp_test,x_test,y_test)

#    coment_test,test_y,phone_word,phone_score,deny_word,inter_word,assume_word=loadfile('test.xlsx')
#    comment,score = test_data_result(coment_test,test_y)
   















