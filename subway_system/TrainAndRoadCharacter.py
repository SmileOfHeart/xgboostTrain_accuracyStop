# -*- coding: utf-8 -*-
"""
Created on Sun Mar 31 14:54:49 2019

@author: 10365
"""

#TrainAndRoadCharacter
#定义列车的基本信息如牵引力和制动力以及运行阻力等
#定义道路的基本信息如道路坡度和路段限速等
from tool import findIndex
import tool
import matplotlib.pyplot as plt
import numpy as np 
import pandas as pds
import sys


##宋家庄道路坡度
#gradStartPoint=[0,335,	535,	865,	1525,	2055,	2425,	2625]
#gradList=[-2,	-3,	12,	3.15,	-8,	3,	4.25,	-2]
##宋家庄路段限速
#SLStartPoint=[173.7,451.2,695.3,1264.6,2686,2806]
#speedLimit=[50,80,65,80,55,0]  #km/h
#stateTable=[2,0,2,0,2,0,-1]

#次渠道路坡度
gradStartPoint=[19950,20295,20970,21405]
gradList=[2,-3,3,2]
#gradList=[2,-3,12,8]
#次渠路段限速
SLStartPoint=[20103,20283,21449,21569]
speedLimit=[55, 80,	55,	0]  #km/h
stateTable=[2,0,2,0,-1]

#列车质量（t)
M=194.295  

def getRoadGradinet(pos):
    #道路坡度信息
    #Pos:位置（m) 返回值：千分度
    grad=0
    for i in range(0,len(gradStartPoint)):
        if pos<gradStartPoint[i]:
            grad=gradList[i-1]
            break 
    return grad


def getTrateForce(veo):
    #UNTITLED3 根据牵引曲线计算牵引力大小
    #  传入速度单位为m/s
    f=0
    u=veo*3.6 #单位换算
    if u<51.5:
        f=203
    elif u<80:
        f=-0.002032*u*u*u+0.4928*u*u-42.13*u+1343  
    return f


def getBrakeForce(veo):
    #UNTITLED3 根据制动曲线计算阻力大小
    #  传入速度单位为m/s
    #  阻力单位为KN
    f=0
    u=veo*3.6 #单位换算
    if u<77:
        f=166 
    elif u<80:
        f=0.1314*u*u-25.07*u+1300  
    return f


def getRoadspeedLimit(pos):
    #UNTITLED 最原始的限速信息
    # 返回值单位：m/s
    global SPDLIMARRAY
    vLimit=0
    i=findIndex(SLStartPoint,pos)
    vLimit=speedLimit[i]
    return vLimit/3.6


def getAntiForce(veo,pos):
    #UNTITLED6 计算附加阻力
    #单位是KN
    #传入速度单位为m/s,位置为m
    u=veo*3.6 
    w0=2.031+0.0622*u+0.001807*u*u 
    wi=getRoadGradinet(pos) 
    #wr
    #ws
    f=(w0+wi)*M*9.8/1000 
    return f

'''
    绘制限速和线路坡度
'''
def plotSpeedLimitRoadGrad(style):
    #style:relative :相对坐标(相对于起点)
    #      abstract: 绝对坐标
    Mat=tool.ReadCSVMat('./subway_system/speedLimit_ciqu')
    startPoint=Mat[:,0]
    speedLimit=Mat[:,1]*3.6
    n=len(startPoint)
    x=np.zeros(2*n-2)
    y=np.zeros(2*n-2)
    x[0]=startPoint[0]
    y[0]=speedLimit[0]/3.6
    #中间的点都重复两次
    for i in range(1,n-1):
        x[2*i-1]=startPoint[i]
        y[2*i-1]=speedLimit[i-1]/3.6
        x[2*i]=startPoint[i]
        y[2*i]=speedLimit[i]/3.6      
    x[2*n-3]=startPoint[n-1]
    y[2*n-3]=speedLimit[n-2]/3.6
    if style=='relative':
        x=x-np.ones(2*n-2)*startPoint[0]
    slx=pds.DataFrame(x.T)
    sly=pds.DataFrame(y.T)
    sl=pds.concat([slx,sly],axis=1)  #对行操作，水平连接
    sl.columns=['s','speedlimit']
    sl.to_csv('./simulation_output/speedlimit.csv')
    plt.plot(x,y)
    
    rate=0.3
    startPoint=gradStartPoint
    n=len(startPoint)
    x=np.zeros(2*n)
    y=np.zeros(2*n)
    x[0]=startPoint[0]
    y[0]=gradList[0]*rate
    #中间的点都重复两次
    for i in range(1,n):
        x[2*i-1]=startPoint[i]
        y[2*i-1]=gradList[i-1]*rate
        x[2*i]=startPoint[i]
        y[2*i]=gradList[i]*rate   
    #以车站作为坡度的终点
    x[2*n-1]=SLStartPoint[-1]
    y[2*n-1]=gradList[n-1]*rate
    if style=='relative':
        x=x-np.ones(2*n)*startPoint[0]
        slx=pds.DataFrame(x.T)
    rdx=pds.DataFrame(x.T)
    rdy=pds.DataFrame(y.T)
    rd=pds.concat([rdx,rdy],axis=1)  #对行操作，水平连接
    rd.columns=['s','gd']
    rd.to_csv('./simulation_output/roadgradient.csv')
    plt.plot(x,y)
    

'''
    读取制动限速曲线
'''
def ReadBrakeSpeedLimitCurve():
    #读取存储在文本中的紧急制动曲线信息,返回值 BLKDic
    with open('./subway_system/speedLimit_ciqu.csv',mode='r',encoding='UTF-8-sig') as file_obj:
        contents=file_obj.readlines()
        BLKDic={}  #字典存储 {位置（key)，限速(value))}对
        for line in contents:
            line.strip() #移除'\n'
            listFormline=line.split(',')
            key=float(listFormline[0])
            value=float(listFormline[1])
            BLKDic[key]=value
        #print(BLKDic)
    return BLKDic

'''
    读取最小时间速度曲线
'''        
def ReadMinTimeCurve():
    #读取存储在文本中的紧急制动曲线信息,返回值 BLKDic
    with open('./subway_system/minTimeCurve_ciqu.csv',mode='r',encoding='UTF-8-sig') as file_obj:
        contents=file_obj.readlines()
        MTCDic={}  #字典存储 {位置（key)，限速(value))}对
        for line in contents:
            line.strip() #移除'\n'
            listFormline=line.split(',')
            key=float(listFormline[0])
            value=float(listFormline[1])
            MTCDic[key]=value
        #print(MTCDic)
    return MTCDic

'''
    获取运行工况,一般用于设置好的工况和工况转换点的仿真
'''    
def getRunState(pos,switchPoint):
    #根据列车位置和工况转换点列表获得列车运行工况
    # 2:牵引 1：巡航 0：惰行 -1：制动 
    index=findIndex(switchPoint,pos)+1
    state=stateTable[index]
    return state

 
def getSpeedLimitEndPoint(pos):
	#获得当前限速的终点
    #输入单位m/s
    point=SLStartPoint[-1]
    try:
        point=SLStartPoint[tool.findIndex(SLStartPoint,pos)+1] 
    except IndexError:
        print(pos)
    return point
    
def getNextSpeedLimit(pos):
   #获得下一区段限速限速值
   # 返回值单位：m/s
    global SPDLIMARRAY
    vLimit=0
    i=findIndex(SLStartPoint,pos)
    if i<len(speedLimit)-1:
        vLimit=speedLimit[i+1]
    else:
        vLimit=speedLimit[i]    
    return vLimit/3.6
	
'''
    加载道路和列车运行限制条件数据
'''
class TrainAndRoadData():
    def __init__(self):
        self.BLKDic=ReadBrakeSpeedLimitCurve()
        self.MTCDic=ReadMinTimeCurve()
        MT=list(self.MTCDic.values())
        self.tnMin=MT[-1]
    
    def getEmerencyBrakeSpeed(self,pos):
        pos=min(pos,SLStartPoint[-1])
        value=tool.findAtInter(self.BLKDic.keys(),self.BLKDic.values(),pos)
        return value
    
    def getMinTime(self,pos):
        #当前位置在最短时间运行曲线上的时间点
        pos=min(pos,SLStartPoint[-1])
        tMin=tool.findAtInter(self.MTCDic.keys(),self.MTCDic.values(),pos)         
        return tMin
    
    def getMinRestTime(self,pos):
        #当前位置在最短时间运行曲线上的时间点
        pos=min(pos,SLStartPoint[-1])
        tMin=self.getMinTime(pos)       
        return self.tnMin-tMin
    
    
    def getCurrentSectionMinTime(self,pos):
        #当前限速区间的最小运行时间
        pos=min(pos,SLStartPoint[-1])
        vlimEndPoint=getSpeedLimitEndPoint(pos) #获得限速的终点
        value=self.getMinTime(vlimEndPoint)-self.getMinTime(pos) #当前区间最小运行时间
        return value
    
    def PlotMinTimeCurve(self):
        #画出最小运行时间曲线
        plt.plot(list(self.MTCDic.keys()),list(self.MTCDic.values()))
    
    def PlotEmerencyBrakeCurve(self):
        plt.plot(list(self.BLKDic.keys()),list(self.BLKDic.values()))

if __name__ == '__main__':   
    plotSpeedLimitRoadGrad()    
    trd=TrainAndRoadData() 
    trd.PlotMinTimeCurve()