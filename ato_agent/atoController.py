# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 10:30:45 2019

@author: 10365
"""

#列车运行控制器
#atoController

import sys
sys.path.append('./subway_system')
sys.path.append('./xgboost_algorithm')
import TrainAndRoadCharacter as trc
import tool
import pandas as pds
import xgboost as xgb
import numpy as np

#控制器的基类
class controller():
    def __init__(self,timeStepLenth,initOutput):
        self.dt=timeStepLenth     #仿真时间步长
        self.out=initOutput    #控制器输出
    #控制器输入
    def Input(self):
        return
    
    #控制器运行
    def Run(self):
        return

    #控制器输出  
    def Output(self):
        #控制器输出
        return self.out  

    def Step(self):
        self.Input()
        self.Run()
        return self.Output()
#-------------------------------------------------------------------------------------------               
class PidController(controller):
    def __init__(self,Dt,KP,KI,KD):
        super().__init__(Dt,0)
        self.kp=KP         #比列因子
        self.ki=KI         #积分因子
        self.kd=KD         #微分因子
        self.eInteger=0    #误差积分
        self.ek=0          #e(k)
        self.e1=0          #e(k-1)
        self.target=0      #目标
        self.actual=0      #实际输入
    
    def SetParameters(self,KP,KI,KD):
        self.kp=KP         #比列因子
        self.ki=KI         #积分因子
        self.kd=KD         #微分因子
        
    def Run(self):
        #PIDIncrement
        self.ek=self.target-self.actual   
        self.eInteger=self.eInteger+self.ek*self.dt        
        self.out=self.kp*self.ek+self.ki*self.eInteger+self.kd*(self.ek-self.e1)*self.dt
        self.e1=self.ek
    
    def Input(self,target,actual):
        self.target=target      #目标
        self.actual=actual      #实际输入 
        return

    def Step(self,target,actual):
        self.Input(target,actual)
        self.Run()
        return self.Output()
#--------------------------------------------------------------------------------------------
class XgbATO(controller):   
    def __init__(self,start,destination,T,dt):
        super().__init__(dt,0)
        #离线信息
        self.tripStart=start #旅程起点
        self.tripDestination=destination #旅程终点
        self.tripTime=T
        self.dv=0.1                       #允许速度误差
        self.MaxAcc=1.5            #最大牵引加速度
        self.MinAcc=-1.5             #最大制动加速度        
        self.dadt=0.5                #加速度变化率，初始为0.5
        self.trd=trc.TrainAndRoadData()
        self.MinT=self.trd.getMinRestTime(self.tripDestination)
        #在线信息（状态）
        self.pos=self.tripStart    #位置
        self.veo=0                 #当前速度
        self.time=0                #仿真时间步
        self.trueAcc=0                 #当前的加速度acc
        self.vbar=0                #预测速度
        self.last_veo=0
        self.last_pos=start
        self.last_out=0
        self.step=0
        #xgboost训练好的模型
        self.bst=xgb.Booster(model_file='./xgboost_algorithm/xgb.model')
        
        
    def Input(self,pos,veo,time,trueAcc):
        self.last_veo=self.veo
        self.last_pos=self.pos
        self.pos=pos               #位置
        self.veo=veo                 #当前速度
        self.time=time             #仿真时间步
        self.trueAcc=trueAcc           #控制器输出的加速度
        self.last_out=self.out
        self.step+=1
        return
    
    def Run(self):
        distance=125
        #组装好的控制器，包含若干个子模块
        if self.pos!=self.last_pos:
            uc=(self.veo*self.veo-self.last_veo*self.last_veo)/(2*(self.pos-self.last_pos))  #测量计算得到的加速度
            detu0=uc-self.last_out                  #误差值
        else:
            detu0=0       
        self.vbar=self.MTDProPro(self.pos,self.time,detu0)
        self.out=0  #默认惰行
        #基于xgboost的智能学习算法
        s=self.pos
        sr=self.tripDestination-self.pos
        tr=self.tripTime-self.time
        sl=trc.getRoadspeedLimit(self.pos)  #路段限速
        gd=trc.getRoadGradinet(self.pos)    #道路坡度
        nsl=trc.getNextSpeedLimit(self.pos)   #下一限速
        nsld=trc.getSpeedLimitEndPoint(self.pos)  #当前限速区间剩余距离
        if self.pos<self.tripDestination-distance:
            character=[s,sr,self.tripTime,tr,sl,gd,nsl,nsld]
            xdata=xgb.DMatrix(np.mat(character))
            preds = self.bst.predict(xdata)
            self.out=max(preds[0],-1)   #计算阻力误差加速度  
            if preds[0]>0.99:
                self.out=1
            elif preds[0]<0.01 and preds[0]>-0.1:
                self.out=0.0
#            elif preds[0]<-0.75:
#                    self.out=-1   
        #安全保护算法  
        ebkv=self.trd.getEmerencyBrakeSpeed(self.pos)     #紧急制动速度
        if self.veo<self.vbar and self.pos<self.tripDestination-distance and self.out<=0:
            self.out=self.last_out+self.dadt*self.dt
        if self.veo>ebkv and self.pos<self.tripDestination-distance:
            #超速紧急制动
            self.out=-1   
        #精准停车算法
        if self.pos>self.tripDestination-distance and self.pos<self.tripDestination:
            up=-1*self.veo*self.veo/(2*sr)     #目标加速度
            self.out=up-detu0  #计算阻力误差加速度
            self.out=up
            self.out=max(-1.0,self.out)
        if  self.pos>self.tripDestination:
            self.out = -1
        if self.out>1:
            self.out=1
        return 
    
    def Output(self): 
        return self.out,self.vbar
        
    def MTD(self,pos,time):
        # minimal-time distribution  algorithm
        #子模块1：时间分配算法
        tr=(self.tripTime-time)  #真实剩余时间
        vlimEndPoint=trc.getSpeedLimitEndPoint(pos) #获得限速的终点
        tnMin=self.trd.getCurrentSectionMinTime(pos) #当前点到区间终点的获得花的最小时间
        tpMin=self.trd.getMinRestTime(pos)  #当前点到旅程终点最小预测时间
        if tpMin<3:
            tpMin=3
        trbar=tr*tnMin/tpMin #预测当前点到区间终点应分配的时间
        if trbar<3:
            trbar=3
        vbar=(vlimEndPoint-pos)/trbar
        return vbar-5
    
    def MTDProPro(self,pos,time,detaAcc):
        # minimal-time distribution  algorithm
        #子模块1：时间分配算法
        tr=(self.tripTime-time)  #真实剩余时间
        vlimEndPoint=trc.getSpeedLimitEndPoint(pos) #获得限速的终点
        tnMin=self.trd.getCurrentSectionMinTime(pos) #当前点到区间终点的获得花的最小时间
        tpMin=self.trd.getMinRestTime(pos)  #当前点到旅程终点最小预测时间
        if tpMin<3:
            tpMin=3
        trbar=tr*tnMin/tpMin #预测当前点到区间终点应分配的时间
        if trbar<3:
            trbar=3
        S=vlimEndPoint-pos 
        vbar=0.5*detaAcc*trbar+S/trbar
        return vbar/3
    
    def MTDPro(self,pos,time):
        # minimal-time distribution  algorithm
        #子模块1：时间分配算法
        tr=(self.tripTime-time)  #真实剩余时间
        vlimEndPoint=trc.getSpeedLimitEndPoint(pos) #获得限速的终点      
        if self.trd.getEmerencyBrakeSpeed(vlimEndPoint)<self.veo:
            ta=0
        else:
            ta=(self.trd.getEmerencyBrakeSpeed(vlimEndPoint)-self.veo)/self.MaxAcc
        trMax=tr-self.trd.getMinRestTime(vlimEndPoint)-ta #当前点到区间终点的充裕时间
        if trMax<5:
            trMax=5
        vbar=(vlimEndPoint-pos)/trMax
        return vbar 


    def Step(self,pos,veo,time,trueAcc):
        self.Input(pos,veo,time,trueAcc)
        self.Run()
        return self.Output() 

#------------------------------------------------------------------------
class PIDATO(controller):   
    def __init__(self,start,destination,T,dt,filename):
        super().__init__(dt,0)
        #离线信息
        self.tripStart=start #旅程起点
        self.tripDestination=destination #旅程终点
        self.tripTime=T
        targetCurve=pds.read_csv(filename)
        self.posList=targetCurve['s']
        self.veoList=targetCurve['v']
        self.pidCtrl=PidController(dt,0,0,0)
        self.pos=start 
        self.tarVeo=0       #目标
        self.actVe0=0       #实际输入 
    
    def SetParameters(self,KP,KI,KD):
        self.pidCtrl.SetParameters(KP,KI,KD)
        
    def Run(self):
        self.tarVeo=tool.findAtInter(self.posList,self.veoList,self.pos)
        self.out=self.pidCtrl.Step(self.tarVeo,self.actVe0)
        #精准停车算法
        sr=self.tripDestination-self.pos
        if self.pos>self.tripDestination-110 and sr > 0:
            up=-1*self.actVe0*self.actVe0/(2*sr)     #目标加速度
            self.out=up #计算阻力误差加速度
            self.out=max(-1.0,self.out)
        if self.out>1:
            self.out=1
        self.out=min(1.0,self.out)
        self.out=max(-1.0,self.out)
    
    def Input(self,pos,actualVeo):
        self.pos=pos               #位置
        self.actVe0=actualVeo      #速度
        return
    
    def Output(self): 
        return self.out,self.tarVeo

    def Step(self,target,actual):
        self.Input(target,actual)
        self.Run()
        return self.Output()
                

class ExpertSystem(controller):
    def __init__(self,start,destination,T,dt):
        super().__init__(dt,0)
        #离线信息
        self.tripStart=start #旅程起点
        self.tripDestination=destination #旅程终点
        self.tripTime=T
        self.dv=0.1                       #允许速度误差
        self.MaxAcc=1.5            #最大牵引加速度
        self.MinAcc=-1.5             #最大制动加速度        
        self.dadt=0.5                #加速度变化率，初始为0.5
        self.trd=trc.TrainAndRoadData()
        self.MinT=self.trd.getMinRestTime(self.tripDestination)
        #在线信息（状态）
        self.pos=self.tripStart    #位置
        self.veo=0                 #当前速度
        self.time=0                #仿真时间步
        self.vbar=0                #预测速度
        
        
    def Input(self,pos,veo,time):
        self.pos=pos               #位置
        self.veo=veo                 #当前速度
        self.time=time             #仿真时间步
        return
    
    def Run(self):
        #组装好的控制器，包含若干个子模块
        self.vbar=self.MTDPro(self.pos,self.time)
        self.out=0  #默认惰行
        ebkv=self.trd.getEmerencyBrakeSpeed(self.pos)     #紧急制动速度
        if self.veo<self.vbar:
            self.out=1
        if self.veo>ebkv:
            #超速紧急制动
            self.out=-1     
        return 
    
    def Output(self): 
        return self.out,self.vbar
        
    def MTD(self,pos,time):
        # minimal-time distribution  algorithm
        #子模块1：时间分配算法
        tr=(self.tripTime-time)*1.2  #真实剩余时间
        vlimEndPoint=trc.getSpeedLimitEndPoint(pos) #获得限速的终点
        tnMin=self.trd.getCurrentSectionMinTime(pos) #当前点到区间终点的获得花的最小时间
        tpMin=self.trd.getMinRestTime(pos)  #当前点到旅程终点最小预测时间
        if tpMin<3:
            tpMin=3
        trbar=tr*tnMin/tpMin #预测当前点到区间终点应分配的时间
        if trbar<3:
            trbar=3
        vbar=(vlimEndPoint-pos)/trbar
        return vbar 
    
    def MTDPro(self,pos,time):
        # minimal-time distribution  algorithm
        #子模块1：时间分配算法
        tr=(self.tripTime-time)  #真实剩余时间
        vlimEndPoint=trc.getSpeedLimitEndPoint(pos) #获得限速的终点      
        if self.trd.getEmerencyBrakeSpeed(vlimEndPoint)<self.veo:
            ta=0
        else:
            ta=(self.trd.getEmerencyBrakeSpeed(vlimEndPoint)-self.veo)/self.MaxAcc
        trMax=tr-self.trd.getMinRestTime(vlimEndPoint)-ta #当前点到区间终点的充裕时间
        if trMax<5:
            trMax=5
        vbar=(vlimEndPoint-pos)/trMax
        return vbar 


    def Step(self,pos,veo,time):
        self.Input(pos,veo,time)
        self.Run()
        return self.Output()     
    
