# -*- coding: utf-8 -*-
"""
Created on Thu Mar  1 20:30:14 2018

@author: Administrator
"""


import pandas as pd
import numpy as np
import re
#import yaml
import jieba
import pickle
from keras import callbacks
#from keras.models import model_from_yaml
from evaluate_result import rate_result
from sklearn.cross_validation import train_test_split
#from keras.preprocessing import sequence
from gensim.models.word2vec import Word2Vec
from gensim.corpora.dictionary import Dictionary
#from keras.models import Sequential
#from keras.preprocessing import sequence
from keras.models import Model,load_model
#from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM 
from keras.layers import Bidirectional,Input
from keras.layers.core import Dense, Dropout,Activation

#filtrate = re.compile(u'[^\u4E00-\u9FA5，。,.？!?！]')
filtrate = re.compile(u'[^\u4E00-\u9FA51-9a-zA-Z,.?!，。？！]')
# set parameters:
size=20
par=1
maxlen=15
batch_size = 32
n_epoch =30
vocab_dim=100
def loadfile(path):
    phone=pd.read_excel(path,index=None)
    count=len(phone['word'])
    phone_word=list(phone[0:int(0.5*count)]['word'])    
    phone_score=list(phone['score'])
    deny_word=list(phone['deny_word'])
    inter_word=list(phone['inter_word'])
    assume_word=list(phone['assume_word'])
    return phone_word,phone_score,deny_word,inter_word,assume_word
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
    
def embedding_zi(combine,senchar):
    asp_weight = np.zeros((len(combine),maxlen,8*size))
    embed_weight=np.zeros((len(combine),maxlen,vocab_dim+8*size))
    for i,sentence in enumerate(combine):
        if i%100==0:
            print(i,end=',')
#        senten=jieba.lcut(sentence)
        j=0
        for n,word in enumerate(sentence):
            char_f=char_feature(senchar[i][n])
#            word_f=add_feature(word)
#            feature=np.concatenate((word_f,char_f),axis = 1)
            for zi in list(word):
#                print(j,end=',')
                try:
                    embed_weight[i][j][:]=np.concatenate((word_vectors[zi],char_f[0]),axis = 0)
                    asp_weight[i][j][:] = char_f[0]
                except:
                    embed_weight[i][j][:]=np.concatenate((word_vectors['UNK'],char_f[0]),axis = 0)
                    asp_weight[i][j][:] = char_f[0]
                j=j+1
                if j == maxlen:
                    break
            if j == maxlen:
                break
    return embed_weight ,asp_weight     
def embedding_word(combine,senchar):
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
            char_f=char_feature(senchar[i][n])
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
def read_data(path):
    data=pd.read_excel(path,encoding='gbk')
    comment =np.array(data.loc[:,['comment']])
    label = np.array(data.loc[:,['score']])
    comment = comment.tolist()
    label = label.tolist()
    comment_1 = [sample[0] for sample in comment]
    label_1 = [sample[0] for sample in label]
    return comment_1,label_1
def load_train_file(path):
    fr = open(path,'rb')  
    data1 = pickle.load(fr) 
    score1 = pickle.load(fr)
    char = pickle.load(fr)
#    print(data1)  
    fr.close()   
    return data1,score1,char
def list_word(comment):
    com=[]
    for i in range(len(comment)):
        string=comment[i]
        word=[]
        for sample in string:
            word=word+list(sample)
        com.append(word)
    return com
def tokenizer(text):
    text = [jieba.lcut(document) for document in text]
    return text
#每个句子所含词语对应的索引，所以句子中含有频数小于10的词语，索引为0
def parse_dataset(combined,w2indx):
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
def get_w2v_2():
    modelWord2Vec=Word2Vec.load('model1/Word2vec_model.pkl')
    gensim_dict = Dictionary()
    gensim_dict.doc2bow(modelWord2Vec.wv.vocab.keys(),allow_update=True)
    w2indx = {v: k+1 for k, v in gensim_dict.items()}#所有频数超过10的词语的索引
    w2vec = {word: modelWord2Vec[word] for word in w2indx.keys()}#所有频数超过10的词语的词向量
    a=np.random.normal(size=100,loc=0,scale=0.05)
    w2vec['UNK']=a
#    combined=parse_dataset(combined,w2indx)
#    data= sequence.pad_sequences(combined, maxlen=maxlen )
    return w2vec      
def embedding_w2v(combine):
    f = open('lstm_data/word2vec7.txt', 'r',encoding='utf-8')
    index_dict = {}
    word_vectors ={}
    word2vec = f.readlines()
    for i in range(len(word2vec)):
        one = word2vec[i].split()
        index_dict[one[0]]=i+1
        vec = np.array(one[1:],dtype='float32')
        word_vectors[one[0]] = vec
    f.close()
    def parse_dataset(combine):
        ''' Words become integers
        '''
        data=[]
        for sentence in combine:
            new_txt = []
            for word in sentence:
                try:
                    new_txt.append(index_dict[word])
                except:
                    new_txt.append(index_dict['UNK'])
            data.append(new_txt)
        return data         
    combine=parse_dataset(combine)
#    combined= sequence.pad_sequences(combine, maxlen=maxlen)#索引长度对齐   
    combined= zero_pad(combine, maxlen)
#    model=Word2Vec.load('lstm_data/Word2vec_all_data.pkl')
#    index_dict, word_vectors,combined=create_dictionaries(model,combine)
    return   index_dict, word_vectors,combined
def zero_pad(X, seq_len):
    return np.array([x[:seq_len - 1]+[0] * max(seq_len - len(x), 1)  for x in X])        
def clearn_data(comment,label): 
    graph = []
    graph_label =[]
    for i in range(len(comment)):
        sentence =[]
        content=re.sub(' ', '', comment[i], count=0, flags=0) 
        content = filtrate.sub(r'', content)
        content=re.sub('[.|\?|!|，|。|！|？|、]', ', ', content, count=0, flags=0)
        one_sample = content.split(' ')
        for one in one_sample:
            if one != '' :
                if len(one)>20 :
#                    print(len(one))
                    part = int(len(one)/15) + 1
#                    print(part)
                    for j in range(part):
                        if j < part-1:
                            sentence.append(one[j*15:(j+1)*15])
                        else:
                            sentence.append(one[j*15:])
                else:
                    sentence.append(one)
        if len(sentence) <= 10 and len(sentence) >= 2 :
            graph.append(sentence)
            graph_label.append(label[i])
        if len(sentence) > 10 : 
            graph.append(sentence[:5]+sentence[-5:])
            graph_label.append(label[i])
    return graph,graph_label
def clearn_test_data(comment): 
    graph = []
    for i in range(len(comment)):
        sentence =[]
        content=re.sub(' ', '', comment[i], count=0, flags=0) 
        content = filtrate.sub(r'', content)
        content=re.sub('[.|\?|!|，|。|！|？|、]', ', ', content, count=0, flags=0)
        one_sample = content.split(' ')
        for one in one_sample:
            if one != '' :
                if len(one)>20 :
#                    print(len(one))
                    part = int(len(one)/15) + 1
#                    print(part)
                    for j in range(part):
                        if j < part-1:
                            sentence.append(one[j*15:(j+1)*15])
                        else:
                            sentence.append(one[j*15:])
                else:
                    sentence.append(one)
        if len(sentence) <= 10 :
            graph.append(sentence)
        if len(sentence) > 10 : 
            graph.append(sentence[:5]+sentence[-5:])
    return graph
 

def get_feature_2(graph): 
   
    f = open('lstm_data/connect_word3.txt','r',encoding='utf-8')
    connect = f.read()
    connect = connect.split()
    def creat_one_hot_2(sentence,connect):
        one_hot = np.zeros((len(connect),1))
        for i in range(len(connect)):
#            if len(connect)==1:
#                if sentence.find(connect[i],0,2) != -1:
#                    one_hot[i][0] = 1
#            else:
#                if sentence.find(connect[i]) != -1:
#                    one_hot[i][0] = 1
            if connect[i] in sentence:
                one_hot[i][0] = 1
        return one_hot
    creat_embedding = np.zeros((len(graph),len(connect)+1))        
    for i in range(len(graph)):
        feature = creat_one_hot_2(graph[i],connect)
        for j in range(len(feature)):
            creat_embedding[i][j]=feature[j][0]
        creat_embedding[i][-1] = len(graph[i])/15
    return creat_embedding

def score_feature(score):
    feature= np.zeros((len(score),size)) 
    for i in range(len(score)):
        feature[i][:]=score[i][0]
    return feature
def get_embedding_2(graph,graph_char):
    sentence_data = []
    sen_char=[]
    for i in range(len(graph)):
        for j in range(len(graph[i])):
            sentence_data.append(graph[i][j])
            sen_char.append(graph_char[i][j])
    special_f = get_feature_2(sentence_data)
    print('loading model......')
    model = load_model('lstm_data/vision7_word.h5')
    get_vec2 = Model(inputs=model.input,
                    outputs=model.get_layer('flatten_2').output)
    comment_embed,comment_aspect=embedding_word(sentence_data,sen_char)
#    comment_embed,comment_aspect=embedding_zi(sentence_data,sen_char)
    score = model.predict([comment_embed,comment_aspect])

    score_f=score_feature(score)
    feature_vec = get_vec.predict([comment_embed,comment_aspect]）
    new_feature = np.concatenate((feature_vec,score_f,special_f),axis=1)
    embedding = np.zeros((len(graph),10,new_feature.shape[1])) 
    count = 0
    one_graph_score = []
    for i in range(len(graph)):
        for j in range(len(graph[i])):
            if len(graph[i]) ==1:
                one_graph_score.append([i,new_feature[count][100]])               
            embedding[i][j][:] = new_feature[count][:]
            count = count + 1
    return embedding,one_graph_score

def test_file(path_test1,path2):
    comment,y_label = read_data(path_test1)
    fr = open(path2,'rb')  
    data1 = pickle.load(fr)
    _ = pickle.load(fr)
    test_char = pickle.load(fr)
    fr.close() 
#    test_graph = clearn_test_data(data1)
    test_graph_emb,one_graph_score = get_embedding_2(data1,test_char)
    print('Load graph model......')
    graph_model = load_model('lstm_data/graph_model.h5')
    predict_score = graph_model.predict(test_graph_emb)
    for i in range(len(one_graph_score)):
        predict_score[one_graph_score[i][0]][0] = one_graph_score[i][1]
    def get_label(score):
        label_test = []
        for i in range(len(score)):
            if score[i][0] <=0.4:
                label_test.append(0)
            if score[i][0] >0.4 and score[i][0] <0.6:
                label_test.append(0.5)
            if score[i][0] >=0.6:
                label_test.append(1)
        return label_test
    label_test = get_label(predict_score)
    print('writing data1...')  
    sava_data = pd.DataFrame(comment, columns=['comment'])
    sava_data['true_score'] = y_label
    sava_data['test_score'] = predict_score
    sava_data['label_test'] = label_test
    sava_data.to_excel('result/123.xlsx',index=None)
    print('ALL DONE!!')
def evaluate():
    data=pd.read_excel('result/123.xlsx')
    true_data=list(data['true_score'])
    predict_data=list(data['label_test'])
    p_0,p_half,p_1,p,r_0,r_half,r_1,r,f_0,f_half,f_1,f,accuracy=rate_result(true_data,predict_data)
    print("分类准确率是:%.1f%% ,%.1f%% ,%.1f%% ,%.1f%%" %(p_0*100,p_half*100,p_1*100,p*100))
    print("分类召回率是:%.1f%% ,%.1f%% ,%.1f%% ,%.1f%%" %(r_0*100,r_half*100,r_1*100,r*100))
    print("分类F1值:   %.1f%% ,%.1f%% ,%.1f%% ,%.1f%%" %(f_0*100,f_half*100,f_1*100,f*100))
    print("分类正确率是:%.1f%% " %(accuracy*100))
    return
if __name__=='__main__':
    phone_word,phone_score,deny_word,inter_word,assume_word=loadfile('data/travel/travel_word.xlsx')
#    comment,label = read_data('data/phone/phone_train.xlsx')
#    graph,graph_label = clearn_data(comment,label)
    word_vectors = get_w2v_2()
    graph,graph_label,char = load_train_file('data/travel/travel_train.txt')
    x_train, x_test,x_train_char, x_test_char, y_train, y_test = train_test_split(graph,char, graph_label, test_size=0.2)
    x_train_emb,_ = get_embedding_2(x_train,x_train_char)
    x_test_emb,_ = get_embedding_2(x_test, x_test_char)
    train_lstm_3(x_train_emb,np.array(y_train),x_test_emb,np.array(y_test))
    test_file('data/travel/travel_test.xlsx','data/travel/travel_test.txt')
    evaluate()
   

