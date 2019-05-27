# -*- coding: utf-8 -*-
"""
Created on Mon May 14 10:28:37 2018

@author: ML112new
"""

import tensorflow as tf
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from pandas import ExcelWriter
#%%parameters
input_neuron = ['FaultDistance','Depth','Magnitude','EP_distance','interplate'
                ,'intraplate','oneactivity','twoactivity','questioned'
                ,'hangingwall','footwall','strike','dip','covered','DorFrequency'
                ,'dip_s','dip_c','sigma1_azi_s','sigma1_azi_c','sigma2_azi_s'
                ,'sigma2_azi_c','sigma3_azi_s','sigma3_azi_c','sigma1_plg_s'
                ,'sigma1_plg_c','sigma2_plg_s','sigma2_plg_c','sigma3_plg_s'
                ,'sigma3_plg_c','fault_strike_s','fault_strike_c','EP_azimuth_s'
                ,'EP_azimuth_c','GA','GB','GC','GD','GE','GF','GG','A','B','C'
                ,'D','E','thrust','normal','right','left']
#delindex = [0,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28
#,29,30,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,1,2,3,4,5,31,32] # unused input neuron
delindex = [3,4] # unused input neuron 'index'
delneuron = []
for i in delindex:
    delneuron.append(input_neuron[i])
for i in delneuron: 
    input_neuron.remove(i)
InputNeurons = len(input_neuron)
# output_neuron = ['AI_azimuth_s','AI_azimuth_c','PGA_azi_s','PGA_azi_c']

train_keep = 0.5  # dropout parameter, keep how many percentage
learning_rate = 5e-5
traing_epochs = 16000
printstep = 1000
StdThreshold=0.5
currentpath = os.getcwd()
subname = '_FM5'
filename = 'logistic_ML_data'+subname+'.xlsx'

#%% input parsed data
# read mldata.xlsx and put into array
df_xDirtrain = pd.read_excel(filename,sheetname='x_Dirtrain')
df_yDirtrain = pd.read_excel(filename,sheetname='y_Dirtrain')
df_xZDirtrain = pd.read_excel(filename,sheetname='x_ZDirtrain')
df_yZDirtrain = pd.read_excel(filename,sheetname='y_ZDirtrain')

x_Dirtrain = np.matrix(df_xDirtrain)
y_Dirtrain = np.matrix(df_yDirtrain)
x_ZDirtrain = np.matrix(df_xZDirtrain)
y_ZDirtrain = np.matrix(df_yZDirtrain)

df_xtrain = pd.read_excel(filename,sheetname='x_Eqtrain')
df_ytrain = pd.read_excel(filename,sheetname='y_Eqtrain')
df_xtest = pd.read_excel(filename,sheetname='x_test')
df_ytest = pd.read_excel(filename,sheetname='y_test')

x_train = np.matrix(df_xtrain.iloc[:,3:52])
y_train = np.matrix(df_ytrain)
x_test = np.matrix(df_xtest.iloc[:,3:52])
y_test = np.matrix(df_ytest)

# summary data of alldata or equalratio
df_trainStdMean = pd.read_excel(filename,sheetname='summary_trainEq')
df_testStdMean = pd.read_excel(filename,sheetname='summary_test')
trainDirhat = np.matrix.transpose(np.matrix(df_trainStdMean.trainDir))
testDirhat = np.matrix.transpose(np.matrix(df_testStdMean.testDir))
trainHighhat = np.matrix.transpose(np.matrix(df_trainStdMean.trainHigh))
trainLowhat = np.matrix.transpose(np.matrix(df_trainStdMean.trainLow))
testHighhat = np.matrix.transpose(np.matrix(df_testStdMean.testHigh))
testLowhat = np.matrix.transpose(np.matrix(df_testStdMean.testLow))

# delete neurons
x_Dirtrain = np.delete(x_Dirtrain,delindex,1)
x_ZDirtrain = np.delete(x_ZDirtrain,delindex,1)
x_train = np.delete(x_train,delindex,1)
x_test = np.delete(x_test,delindex,1)

#%% std, mean
def AriasDistribution(y,StdThreshold):
    # y represents the results of the prediction
    # y.shape = (# of data ,18)
    
    ## mirror direction
    distribution = np.zeros((y.shape[0],36))
    for i in range(y.shape[0]):
        ariasindex = np.argmax(y[i,:])
        forindex = np.argmax(y[i,:])+1
        backindex = np.argmax(y[i,:])-1
        if forindex==18:            
            if y[i,0]>y[i,backindex]:
                hibound = forindex
                lobound = ariasindex
            else:
                hibound = ariasindex
                lobound = backindex
        elif backindex==-1:
            if y[i,forindex]>y[i,17]:
                hibound = forindex
                lobound = ariasindex
            else:
                hibound = ariasindex
                lobound = backindex
        else:
            if y[i,forindex]>y[i,backindex]:
                hibound = forindex
                lobound = ariasindex
            else:
                hibound = ariasindex
                lobound = backindex
        for j in range(9):
            aidxp = hibound+j
            didxp = hibound+9+j
            aidxm = lobound-j
            didxm = lobound+9-j
            if aidxp>17:
                aidxp=aidxp-18
            elif aidxm<0:
                aidxm=aidxm+18
            distribution[i,didxp]=y[i,aidxp]
            distribution[i,didxm]=y[i,aidxm]
        
    ## Arias Mean
    ariasmean = np.zeros((y.shape[0],1))
    for i in range(y.shape[0]):
        for j in range(36):
            ariasmean[i,0] += distribution[i,j]*(j-9)*10
    
    ## standard deviation
    ariasstd = np.zeros((y.shape[0],1))
    ariasvar = np.zeros((y.shape[0],1))
    ariassum = np.zeros((y.shape[0],1))
    ariasdir = np.zeros((y.shape[0],1))
    ariasii = np.zeros((y.shape[0],1))
    for i in range(len(ariasmean)):
        for j in range(36):
            ariassum[i,0]+=((j-8)*10-ariasmean[i,0])**2*distribution[i,j]
        ariasvar[i,0] = (ariassum[i,0]/np.sum(distribution[i,:]))
        ariasstd[i,0] = (ariassum[i,0]/np.sum(distribution[i,:]))**0.5
        ariasii[i,0] = np.amin(y[i,:])/np.amax(y[i,:])
        if ariasii[i,0]<=StdThreshold:
            ariasdir[i,0]=1
    
    return ariasstd, ariasmean, ariasdir, distribution

#%% Define Accuracy
    # totalAccuracy = DirAccuracy*MeanAccuracy
   
def DirAccuracy(directivity,dirhat):
    # directivity is trainDir or testDir
    # dirhat is trainDirhat or testDirhat
    # only calculate the accuracy of the data with directivity
    correctidx=[]
    zcorrectidx=[]
    oneidx=[]
    zeroidx=[]
    for i in range(len(directivity)):
        if dirhat[i,0]==1:
            oneidx.append(i)
            if directivity[i,0]==1:
                correctidx.append(i)
        elif dirhat[i,0]==0:
            zeroidx.append(i)
            if directivity[i,0]==0:
                zcorrectidx.append(i)
    if len(oneidx)==0:
        DirAccuracy=None
    else:
        DirAccuracy = len(correctidx)/len(oneidx)
    if len(zeroidx)==0:
        zDirAccuracy=None
    else:
        zDirAccuracy = len(zcorrectidx)/len(zeroidx)
    return DirAccuracy,zDirAccuracy,correctidx

def MeanAccuracy(mean,low,high,corridx):
    # mean represents trainMean or testMean
    # low and high represents Lowhat and Highhat
    # corridx represents train_corridx or test_corridx
    correct=0
    for i in corridx:
        if low[i,0]<=mean[i,0]<=high[i,0]:
            correct+=1
        elif high[i,0]>180:
            cormean = mean[i,0]+180
            if low[i,0]<=cormean<=high[i,0]:
                correct+=1
        elif low[i,0]<0:
            cormean = mean[i,0]-180
            if low[i,0]<=cormean<=high[i,0]:
                correct+=1
    MeanAccuracy = correct/len(corridx)
    return MeanAccuracy

#%% Accuracy
def Accuracy(directivity,dirhat,mean,low,high):
    # directivity is trainDir or testDir
    # dirhat is trainDirhat or testDirhat
    # only calculate the accuracy of the data with directivity
    # mean represents trainMean or testMean
    # low and high represents Lowhat and Highhat
    # corridx represents train_corridx or test_corridx
    zcorrect=0
    zcoridx=[]
    correct=0
    coridx=[]
    for i in range(len(directivity)):
        if dirhat[i,0]==0:
            if directivity[i,0]==0:
                zcorrect+=1
                zcoridx.append(i)
        elif dirhat[i,0]==1:
            if directivity[i,0]==1:
                correct+=1
                coridx.append(i)
#                if low[i,0]<=mean[i,0]<=high[i,0]:
#                    correct+=1
#                elif high[i,0]>180:
#                    cormean = mean[i,0]+180
#                    if low[i,0]<=cormean<=high[i,0]:
#                        correct+=1
#                elif low[i,0]<0:
#                    cormean = mean[i,0]-180
#                    if low[i,0]<=cormean<=high[i,0]:
#                        correct+=1
    accuracy = correct/len(dirhat)
    zaccuracy = zcorrect/len(dirhat)
    totalaccuracy = (zcorrect+correct)/len(dirhat)
    return accuracy,zaccuracy,totalaccuracy,zcoridx,coridx

#%% define layer
def add_layer(inputs, in_size, out_size, keep_prob, activation_function=None, Wname=None, banme=None):
    # add one more layer and return the output of this layer
    Weights = tf.Variable(tf.random_normal([in_size, out_size],stddev=0.01))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    #Wx_plus_b = tf.nn.dropout(Wx_plus_b, keep_prob)
    if activation_function is None:
        outputs = Wx_plus_b
    elif activation_function=='maxout':
        outputs = tf.contrib.layers.maxout(Wx_plus_b,out_size)
    else:
        outputs = activation_function(Wx_plus_b)
    outputs = tf.nn.dropout(outputs, keep_prob)
    return outputs, Weights, biases

# define placeholder for inputs to network
keep_prob = tf.placeholder(tf.float32)
xs = tf.placeholder(tf.float32, [None, InputNeurons])
ys = tf.placeholder(tf.float32, [None, 18])
# add hidden layer
[l1,W1,b1] = add_layer(xs, InputNeurons, 500, keep_prob, activation_function=tf.nn.relu)
[l2,W2,b2] = add_layer(l1,500,500,keep_prob, activation_function=tf.nn.relu)
[l3,W3,b3] = add_layer(l2,500,500,keep_prob, activation_function=tf.nn.relu)
#[l4,W4,b4] = add_layer(l3,500,500,keep_prob, activation_function=tf.nn.relu)
#[l5,W5,b5] = add_layer(l4,100,100,activation_function=tf.nn.relu)
# add output layer
[Flogits,Wp,bp] = add_layer(l3, 500, 18, 1.0, activation_function=None)
prediction = tf.nn.softmax(Flogits)
## cross entropy
#loss = tf.nn.softmax_cross_entropy_with_logits(logits=Flogits,labels=ys)
#loss = tf.losses.softmax_cross_entropy(ys,prediction)
loss = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction), axis=1))
#loss = tf.reduce_mean(-tf.reduce_mean((ys * tf.log(prediction)+(1-ys)*tf.log(1-prediction)),reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
# with learning rate=learning_rate
# in tensorflow, the gradient descent is stochastic gradient descent

#%% training step
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
# use matplotlib to record loss data
TrainLoss=[]
TestLoss=[]
epoch = []

for i in range(traing_epochs):
    # training

#    Diridx = np.random.choice(np.arange(len(x_Dirtrain)), 32, replace=False)
#    ZDiridx = np.random.choice(np.arange(len(x_ZDirtrain)), 32, replace=False)
#    x_train_batch = np.concatenate((x_Dirtrain[Diridx],x_ZDirtrain[ZDiridx]),axis=0)
#    y_train_batch = np.concatenate((y_Dirtrain[Diridx],y_ZDirtrain[ZDiridx]),axis=0)

    idx = np.random.choice(np.arange(len(x_train)), 64, replace=False)
    x_train_batch = x_train[idx]
    y_train_batch = y_train[idx]
    sess.run(train_step, feed_dict={xs: x_train_batch, ys: y_train_batch, keep_prob: train_keep})
    if i % printstep == 0:    
        train_loss = sess.run(loss, feed_dict={xs: x_train, ys: y_train, keep_prob: 1})
        loss_test = sess.run(loss, feed_dict={xs: x_test, ys: y_test, keep_prob:1})
#        trainoutput = sess.run(prediction, feed_dict={xs: x_train, keep_prob:1})
#        testoutput = sess.run(prediction, feed_dict={xs: x_test, keep_prob:1})
        print("epoch",i,"Testing loss=", loss_test,"Train Loss=",train_loss)
        TrainLoss.append(train_loss)
        TestLoss.append(loss_test)
        epoch.append(i)

#%% print the training results
trainoutput = sess.run(prediction, feed_dict={xs: x_train, keep_prob:1})
testoutput = sess.run(prediction, feed_dict={xs: x_test, keep_prob:1})
W1 = sess.run(W1, feed_dict={xs: x_train, keep_prob:1})
W2 = sess.run(W2, feed_dict={xs: x_train, keep_prob:1})
W3 = sess.run(W3, feed_dict={xs: x_train, keep_prob:1})
Wp = sess.run(Wp, feed_dict={xs: x_train, keep_prob:1})
"""
del TrainLoss[0]
del TestLoss[0]
del epoch[0]
"""

#%% finished training
print("Optimization Finished!")
print("Training loss=", train_loss)
print("Testing loss=", loss_test)
[trainStd,trainMean,trainDir,trainDis] = AriasDistribution(trainoutput,StdThreshold)
[testStd,testMean,testDir,testDis] = AriasDistribution(testoutput,StdThreshold)

#%% Accuracy
[trainDirAcc, trainZDirAcc, TrainCoridx] = DirAccuracy(trainDir,trainDirhat)
[testDirAcc, testZDirAcc, TestCoridx] = DirAccuracy(testDir,testDirhat)
trainMeanAccuracy = MeanAccuracy(trainMean,trainLowhat,trainHighhat,TrainCoridx)
testMeanAccuracy = MeanAccuracy(testMean,testLowhat,testHighhat,TestCoridx)
trainaccuracy,trainZaccuracy,trainTaccuracy,trainzcoridx,traincoridx = Accuracy(trainDir,trainDirhat,trainMean,trainLowhat,trainHighhat)
testaccuracy,testZaccuracy,testTaccuracy,testzcoridx,testcoridx = Accuracy(testDir,testDirhat,testMean,testLowhat,testHighhat)
print('trainDirAcc=',trainDirAcc)
print('testDirAcc=',testDirAcc)
print('trainZDirAcc=',trainZDirAcc)
print('testZDirAcc=',testZDirAcc)
print('trainTaccuracy=',trainTaccuracy)
print('testTaccuracy=',testTaccuracy)
print('trainMeanAccuracy=',trainMeanAccuracy)
print('testMeanAccuracy=',testMeanAccuracy)

#%% Accurate stations and events
train_cor_data = df_xtrain.iloc[traincoridx,0:3]
#train_cor_event = df_xtrain.iloc[traincoridx,0]
#train_cor_station = df_xtrain.iloc[traincoridx,1]
#train_cor_fault = df_xtrain.iloc[traincoridx,2]
test_cor_data = df_xtest.iloc[testcoridx,0:3]
#test_cor_event = df_xtest.iloc[testcoridx,0]
#test_cor_station = df_xtest.iloc[testcoridx,1]
#test_cor_fault = df_xtest.iloc[testcoridx,2]
train_zcor_data = df_xtrain.iloc[trainzcoridx,0:3]
test_zcor_data = df_xtest.iloc[testzcoridx,0:3]

#%% plot loss
fig = plt.figure()
# 111代表在subplot圖中的位置
ax = fig.add_subplot(111)
ax.plot(epoch, TrainLoss, c='b', label='TrainLoss', linewidth=0.5)
ax.plot(epoch, TestLoss, c='r', label='TestLoss', linewidth=0.5)
leg = plt.legend()
# get the lines and texts inside legend box
leg_lines = leg.get_lines()
leg_texts = leg.get_texts()
# bulk-set the properties of all lines and texts
plt.setp(leg_lines, linewidth=6)
plt.setp(leg_texts, fontsize='x-large')
plt.savefig('logistic_result\\'+subname+'_losses.png', dpi=600)
plt.show()

#%% Method 1
w1_mean = np.mean(W1,axis=1)
df_w1mean=pd.DataFrame({'mean':w1_mean})
w1_absmean = np.absolute(np.mean(W1,axis=1))
df_w1absmean=pd.DataFrame({'absmean':w1_absmean})
w1_max = np.amax(W1,axis=1)
df_w1max=pd.DataFrame({'max':w1_max})
w1_absmax = np.amax(np.absolute(W1),axis=1)
df_w1absmax=pd.DataFrame({'absmax':w1_absmax})

workbook = ExcelWriter('weights'+subname+'_wointerintra'+'.xlsx')
df_w1mean.to_excel(workbook,'weights',index=False,startcol=0)
df_w1max.to_excel(workbook,'weights',index=False,startcol=1)
df_w1absmean.to_excel(workbook,'weights',index=False,startcol=2)
df_w1absmax.to_excel(workbook,'weights',index=False,startcol=3)
train_cor_data.to_excel(workbook,'traincor',index=False)
test_cor_data.to_excel(workbook,'testcor',index=False)
train_zcor_data.to_excel(workbook,'trainzcor',index=False)
test_zcor_data.to_excel(workbook,'testzcor',index=False)
workbook.save()

##%% Method 3
#ordpred=sess.run(prediction, feed_dict={xs:x_test[9,:],keep_prob:1})
#OrdPred = np.amax(ordpred)
#for i in range(45):
#    test_xtrain=np.zeros(x_test[9,:].shape)
#    for j in range(45):
#        test_xtrain[0,j]=np.asarray(x_test[9,j])
#    test_xtrain[0,i]=0
#    testprediction=sess.run(prediction, feed_dict={xs:test_xtrain,keep_prob:1})
#    TestPred = np.amax(testprediction)
#    delprob = (OrdPred-TestPred)/TestPred
#    #print(testprediction)
#    print(i,delprob,np.argmax(testprediction))
