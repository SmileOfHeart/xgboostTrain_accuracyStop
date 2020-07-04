# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 14:20:18 2019

@author: 10365

"""

import sys
sys.path.append('..\subway_system')
import TrainAndRoadCharacter as trc
import atoController
import matplotlib.pyplot as plt
import trainRunningModel as trm
import pandas as pds
import numpy as np

    
def Run(style):
    #style:   控制器类型  "svm":SVM  "sw":switchPoint "c":Coast
    #仿真参数
    startPoint=trc.SLStartPoint[0]
    endPoint=trc.SLStartPoint[-1]
    #仿真变量
    s=startPoint     #位置
    dt=0.2           #仿真时间步长
    v=0.1
    t=0
    acc=0.6          #起始加速度    
    Energy=0              #能耗
    #统计变量
    train=trm.Train_model(startPoint,0.5,0.6,dt)
    SRecord=[0]       #position list with start position  
    vRecord=[v]       #speed list with start speed
    tRecord=[t]       #time list
    accList=[acc]           #acc list 
    DEList=[0]              #能耗
    vbarList=[0]            #推荐速度
    MisError=0       #停车误差
    jerk=0           #舒适度   
    vbar=0           #目标速度
    #控制器设置
    if style=='PID':
        controller=atoController.PIDATO(startPoint,endPoint,98,dt,'./targetCurveDataSet/115.0_Curve.csv') #PID
        controller.SetParameters(3,0.3,1)
    elif style=='xgboost':
        controller=atoController.XgbATO(startPoint,endPoint,115,dt) #专家系统
    else:
        controller=atoController.ExpertSystem(startPoint,endPoint,108,dt) #专家系统
    while s<endPoint+100 and v >0:
        #控制器
        if style=='PID':
            cmd_acc,vbar=controller.Step(s,v)
        elif style=='xgboost':
            cmd_acc,vbar=controller.Step(s,v,t,acc)
        else:
            cmd_acc,vbar=controller.Step(s,v,t)
        #列车模型
        trainState=train.Step(cmd_acc)
        #更新状态
        #acc=trainState['acc']
        acc=trainState['acc']
        v=trainState['v']
        dE=trainState['P']
        jerk=trainState['Jerk']
        Energy=Energy+dE*dt
        s=trainState['S']
        t=t+dt;
        #统计数据
        vRecord.append(v)
        tRecord.append(t)
        DEList.append(dE/100)
        SRecord.append(s-startPoint)
        accList.append(cmd_acc)
        vbarList.append(vbar)
        MisError=s-endPoint
    jerk=jerk/t
    varRecAry=np.mat([SRecord,vbarList,vRecord,tRecord,DEList,accList])
    varRec=pds.DataFrame(data=varRecAry.T,columns=['s','vbar','v','t','p','acc'])
    var=[t,Energy,jerk,MisError]
    return var,varRec

stl='xgboost'
#stl='PID'
var,res=Run(stl)
print('Simulation End')
res.to_csv(stl+'result.csv')
trc.plotSpeedLimitRoadGrad('relative')
plt.plot(res['s'],res['vbar'])  #画vbar-x 
plt.plot(res['s'],res['v'])  #画v-x 
plt.show()
plt.plot(res['s'],res['acc'])  #画acc
plt.show()
plt.plot(res['s'],res['vbar'])  #画vbar-x 
plt.show()