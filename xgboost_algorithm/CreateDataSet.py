# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 11:09:33 2019

@author: 10365
"""


#CreateDataSet

import numpy as np 
import sys
sys.path.append('..\subway_system')
import TrainAndRoadCharacter as trc
import trainRunningModel as trm
import pandas as pds
import matplotlib.pyplot as plt
import atoController

def readSwitchPointSet(file):
    #返回值SwitchPointMat
    with open(file,mode='r',encoding='UTF-8-sig') as file_obj:
        contents=file_obj.readlines()
        row=len(contents)    #行数
        col=len(contents[0].split(','))      #列数
        SwitchPointMat=np.zeros((row,col))
        rIndex=0
        for line in contents:
            line.strip() #移除'\n'
            listFormline=line.split(',')
            cIndex=0
            for ele in listFormline:
                SwitchPointMat[rIndex,cIndex]=float(ele)
                cIndex+=1
            rIndex+=1
    return SwitchPointMat


def SaveDataSet(num,data,filedir):
    data.reset_index()
    data.to_csv(filedir+str(num)+'_dataSet.csv',index=False)
    return True


def ReadDataSet(num,filedir):
    with open(filedir+str(num)+'_dataSet.csv',mode='r',encoding='UTF-8-sig') as file_obj:
        contents=file_obj.readlines()
        line=contents[0]
        line.strip()
        strElems=line.split(',')
        row=len(contents)
        col=len(strElems)
        dataMat=np.zeros((row,col))        
        for i in range(0,row):
            line=contents[i]
            line.strip()
            strElems=line.split(',')
            for j in range(0,col):
                dataMat[i,j]=float(strElems[j])
    return dataMat



def TanslationBySimulation(switchPoint,index):
    #模拟列车运行，将工况控制点转换成（s,v,t,u)组合
    dt=0.2      #时间步长
    startPoint=trc.SLStartPoint[0]
    endPoint=trc.SLStartPoint[-1]
    sl=trc.getRoadspeedLimit(startPoint)
    gd=trc.getRoadGradinet(startPoint)
    nsl=trc.getNextSpeedLimit(startPoint)
    nsld=trc.getSpeedLimitEndPoint(startPoint)
    vList=[0]                   #速度
    sList=[startPoint]          #位置
    tList=[0]                   #时间
    uList=[1]                   #加速度
    gdList=[gd]                  #坡度  千分
    slList=[sl]                 #路段限速（m/s)
    nslList=[nsl]                  #下一限速的值（m/s）
    nsldList=[nsld]                 #下一限速的距离（m)
    train=trm.Train_model(startPoint,0,0.6,dt)
    PIDC=atoController.PidController(dt,8,10,1)
    trd=trc.TrainAndRoadData()
    t=0
    accList=[0]
    stateList=[2]
    state=2
    acc=0.1
    while True:        
        t=t+dt
        laststate=state
        state=trc.getRunState(sList[-1],switchPoint)
        if state==1 and laststate!=1:
            vbar=vList[-1]
        if state==2:
            #牵引
            acc=1
        if state==1:        
            #巡航
            acc=PIDC.Step(vbar,vList[-1])
        elif state==0:
            #惰行
            acc=0
        elif state==-1:
            #制动
            acc=-1
        if vList[-1]>trd.getEmerencyBrakeSpeed(sList[-1]):
            acc=-1
        out=train.Step(acc)
        trueAcc=out['acc']
        stateList.append(state)
        accList.append(acc)
        sl=trc.getRoadspeedLimit(out['S'])
        gd=trc.getRoadGradinet(out['S'])
        nsl=trc.getNextSpeedLimit(out['S'])
        nsld=trc.getSpeedLimitEndPoint(out['S'])
        vList.append(out['v'])
        sList.append(out['S'])
        tList.append(t)
        uList.append(acc)   
        gdList.append(gd)                  #坡度  千分
        slList.append(sl)                #路段限速（m/s)
        nslList.append(nsl)                  #下一限速的值（m/s）
        nsldList.append(nsld)                #下一限速的距离（m)
        if out['S']>endPoint or out['v']<0:
            break
    #保存数据
    plt.plot(sList,accList)
    plt.plot(sList,stateList)
    plt.show()
    trc.plotSpeedLimitRoadGrad('abstract')
    plt.plot(sList,vList)
    plt.show()
    plt.plot(tList,vList)
    plt.show()
    print('-------------------%d---------------------' %index)
    dataList=[]
    for i in range(0,len(vList)):
        s=sList[i]
#        t=tList[i]
        sr=endPoint-sList[i]
        t=tList[i]
        tr=tList[-1]-t
        vn=vList[i]
        un=uList[i]
        sr=round(sr,2)
        tr=round(tr,2)
        vn=round(vn,2)
        sl=slList[i]
        gd=gdList[i]
        nsl=nslList[i]
        nsld=nsldList[i]
        line=[s,sr,tList[-1],tr,sl,gd,nsl,nsld,un]
        #如果list是一维的，则是以列的形式来进行添加，如果list是二维的则是以行的形式进行添加的
        dataList.append(line)
    tC=np.mat([sList,vList])
    targetCurve=pds.DataFrame(data=tC.T,columns=['s','v'])
    pData=pds.DataFrame(data=dataList,columns=['s','sr','t','tr','sl','gd','nsl','nsld','un'])
    return pData, targetCurve, tList[-1]
        
def produceData(style):         
    if style=='train':
     #训练数据   
         swfile='TrainSwitchPointSet.csv'
         outputdir='./TrainningDataSet/'
     #测试数据  
    elif style=='test':
         swfile='TestSwitchPointSet.csv'
         outputdir='./TestingdataSet/'
    sps=readSwitchPointSet(swfile)
    print('开始生产数据')
    row=sps.shape[0]
    maxLevel=0
    for i in range(0,row):
        dataList,targetCurve,T=TanslationBySimulation(sps[i,:].tolist(),i) 
        SaveDataSet(i+1,dataList,outputdir)
        targetCurve.to_csv('./targetCurveDataSet/'+str(round(T,2))+'_Curve.csv',index=False)
        print(str(i))
        print(str(T))
    return maxLevel

if __name__ == '__main__':  
    ml=produceData('train')
    ml=produceData('test')