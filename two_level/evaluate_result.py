import pandas as pd
def data_prepare(ture_data,predict_data):
    TP_0=0
    FN_0=0
    FP_0=0
    TN_0=0
    TP_half=0
    FN_half=0
    FP_half=0
    TN_half=0
    TP_1=0
    FN_1=0
    FP_1=0
    TN_1=0
    for i in range(len(ture_data)):
        if ture_data[i]==0:
            if predict_data[i]==ture_data[i]:
                TP_0=TP_0+1
            else:
                FN_0=FN_0+1
        if ture_data[i]!=0 and  predict_data[i]==0:
            FP_0=FP_0+1
        if ture_data[i]!=0 and  predict_data[i]!=0:
            TN_0=TN_0+1
        if ture_data[i]==0.5:
            if predict_data[i]==ture_data[i]:
                TP_half=TP_half+1
            else:
                FN_half=FN_half+1
        if ture_data[i]!=0.5 and  predict_data[i]==0.5:
            FP_half=FP_half+1
        if ture_data[i]!=0.5 and  predict_data[i]!=0.5:
            TN_half=TN_half+1
        if ture_data[i]==1:
            if predict_data[i]==ture_data[i]:
                TP_1=TP_1+1
            else:
                FN_1=FN_1+1
        if ture_data[i]!=1 and  predict_data[i]==1:
            FP_1=FP_1+1
        if ture_data[i]!=1 and  predict_data[i]!=1:
            TN_1=TN_1+1
    return TP_0,FN_0,FP_0,TN_0,TP_half,FN_half,FP_half,TN_half,TP_1,FN_1,FP_1,TN_1
def rate_result(ture_data,predict_data):
    TP_0,FN_0,FP_0,TN_0,TP_half,FN_half,FP_half,TN_half,TP_1,FN_1,FP_1,TN_1=data_prepare(ture_data,predict_data)
    p_0=TP_0/(TP_0+FP_0)
    p_half=TP_half/(TP_half+FP_half)
    p_1=TP_1/(TP_1+FP_1)
    r_0=TP_0/(TP_0+FN_0)
    r_half=TP_half/(TP_half+FN_half)
    r_1=TP_1/(TP_1+FN_1)
    f_0=2*TP_0/(2*TP_0+FP_0+FN_0)
    f_half=2*TP_half/(2*TP_half+FP_half+FN_half)
    f_1=2*TP_1/(2*TP_1+FP_1+FN_1)
    accuracy=(TP_0+TP_half+TP_1)/len(ture_data)
    p=(p_0+p_half+p_1)/3
    r=(r_0+r_half+r_1)/3
    f=(f_0+f_half+f_1)/3
    return p_0,p_half,p_1,p,r_0,r_half,r_1,r,f_0,f_half,f_1,f,accuracy
#    return p,r,f,accuracy
#data=pd.read_excel('result/lstm_zi.xlsx')
#true_data=list(data['label_true'])
#predict_data=list(data['label_test'])
#p,r,f,accuracy=rate_result(true_data,predict_data)