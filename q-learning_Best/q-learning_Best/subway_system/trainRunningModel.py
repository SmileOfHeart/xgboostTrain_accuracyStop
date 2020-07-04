


import TrainAndRoadCharacter as trc
import numpy as np
from queue import Queue 
import controlModel as ctl

class Train_model():
    #根据simulink仿真平台的仿真图搭建的
    def __init__(self,startPos,initSpeed,initAcc,dt):
        #仿真参数
        self.deltaTime=dt        #仿真时间步长
        self.weight=trc.M        #列车质量
        self.tao=0.4         #传输延时常数
        self.sigma=6         #传输延时6*0.2=1.2s
        #列车状态
        self.speed=initSpeed     #列车速度 
        self.postion=startPos    #列车位置 
        self.trueAcc=0        #真实加速度
        self.TractionPower=0  #牵引功率
        self.RegenerativePower=0 #再生制动功率
        self.Jerk=0            #冲击率
        self.true_traction_acc=0    #
        self.true_brake_acc=0       #
        self.dadt=1
        #输入命令
        self.aim_acc=initAcc
        #存储buff
        self.delay=0.8   #延迟环节参数
        self.delayM=ctl.DelayModel(self.delay,dt,0.6)
        self.Inertia=ctl.InertiaModel(dt,0.6)
        
    def InputPort(self,cmd):
        self.aim_acc=cmd
    
    def RefreshTrainState(self):        
        #计算牵引力和制动力
        acc=0
        if self.aim_acc>0:
            #延时和一阶惯性系统 
            self.aim_acc=self.delayM.Step(self.aim_acc)
            traction_acc=self.Inertia.Step(self.aim_acc)      
            #限制环节
            max_traction_acc=trc.getTrateForce(self.speed)/self.weight
            traction_acc=min(max_traction_acc,traction_acc)
            self.true_traction_acc=traction_acc
        else:
            traction_acc=0
            self.true_traction_acc=traction_acc
        if self.aim_acc<0:    
            self.aim_acc=self.aim_acc*(-1)
            #延时和一阶惯性系统  
            self.aim_acc=self.delayM.Step(self.aim_acc)
            braking_acc=self.Inertia.Step(self.aim_acc)            
            #限制环节
            max_braking_acc=trc.getBrakeForce(self.speed)/self.weight
            braking_acc=min(max_braking_acc,braking_acc)  
            self.true_brake_acc=braking_acc
        else:
            braking_acc=0
            self.true_brake_acc=braking_acc
        #基本阻力和附加阻力对系统造成的干扰    
        anti_acc=trc.getAntiForce(self.speed,self.postion)/self.weight
        acc=traction_acc-anti_acc-braking_acc
        #更新列车状态
        self.TractionPower=traction_acc*self.weight*self.speed
        self.RegenerativePower=traction_acc*self.weight*self.speed
        self.RegenerativePower=braking_acc*self.weight*self.speed
        dacc_dt=(self.trueAcc-acc)/self.deltaTime
        self.Jerk=self.Jerk+abs(dacc_dt)*self.deltaTime
        self.trueAcc=acc
        dS=self.speed*self.deltaTime+0.5*self.trueAcc*self.deltaTime**2
        self.speed=self.speed+self.trueAcc*self.deltaTime
        self.postion=self.postion+dS
        
    def OutputPort(self):
        out={
		'acc':self.trueAcc,
		'v':self.speed,
		'S':self.postion,
		'Jerk':self.Jerk,
		'P':self.TractionPower,
		'R':self.RegenerativePower
		}
        return out
    
    def Step(self,cmd):
        self.InputPort(cmd)
        self.RefreshTrainState()
        return self.OutputPort()
        