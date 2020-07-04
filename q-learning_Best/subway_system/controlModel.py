# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 17:30:57 2019

@author: 10365
"""

import math

from queue import Queue
from matplotlib import pyplot as plt
#时延环节
class DelayModel():     
    def __init__(self,delayTime,SimulinkDetaTime,InitValue):
        self.Buff=Queue()
        N=int(delayTime/SimulinkDetaTime)
        for i in range(N):
            self.Buff.put(InitValue)
    
    def Step(self,value):
        self.Buff.put(value)
        return self.Buff.get()

#惯性环节
class InertiaModel():
    def __init__(self,SimulinkDetaTime,tao):
        self.sum=0
        self.tao=tao
        self.dt=SimulinkDetaTime
    
    def Step(self,value):
        self.sum+=(value-self.sum)*self.dt/self.tao
        return self.sum  
        

if __name__ == "__main__": 
    dt=0.1
    model=InertiaModel(dt,1) 
    delay=DelayModel(1,dt,0)
    tl=[]
    MaxStep=100
    outlist=[]
    inlist=[math.sin(i*dt) for i in range(0,100)]
#    inlist=[1 for i in range(0,MaxStep)]
    for i in range(MaxStep):
        tl.append(i*dt)
        in_v=inlist[i]
        outlist.append(delay.Step(in_v))
    plt.plot(tl,outlist)
    plt.plot(tl,inlist)
    plt.show()