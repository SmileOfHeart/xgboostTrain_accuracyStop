# -*- coding: utf-8 -*-
"""
Created on Sun Mar 31 16:17:37 2019

@author: 10365
"""
import numpy as np

def findAt(keyList,valueList,key):
    #UNTITLED 给出key列表，value列表，表示一个离散化的函数，想从中找出某个自变量对应的值
    #并给出一个key，查找在key列表，找出对应的值value
    #key列表按从小到大排列
    keyList=list(keyList)
    valueList=list(valueList)
    low=1
    high=len(keyList)-1
    while high-low>1:
         i=int(np.floor((low+high)/2+0.5))
         if key>keyList[i]:
             low=i
         else:
             high=i 
    i=max(np.floor((low+high)/2-0.5),1)
    value=valueList[i]
    return value


def findAtInter(keyList,valueList,key):
    #UNTITLED 给出key列表，value列表，表示一个离散化的函数，想从中找出某个自变量对应的值
    #并给出一个key，查找在key列表，找出最近的值value
    #key列表按从小到大排列
    keyList=list(keyList)
    valueList=list(valueList)
    low=1
    high=len(keyList)-1
    count=0
    while high-low>1:
         i=int(np.floor((low+high)/2+0.5))
         if key>keyList[i]:
             low=i
         else:
             high=i 
         count+=1
         if count>100:
             print('%d %d' %(high,low))
             break
    #线性插值法
    if abs(keyList[high]-keyList[low])>0.1:
        value=(valueList[high]-valueList[low])*(key-keyList[low])/(keyList[high]-keyList[low])+valueList[low]
    else:
        value=valueList[low]
    if value<-10:
        print(value)
    return value


def findIndex(valueList,value):
    #找出某个值落在那一个区间
    #valueList值按从小到大排列
    #返回下界
    valueList=list(valueList)
    low=0
    up=len(valueList)-1
    if value<valueList[low]:
        return -1
    if value>valueList[up]:
        index=up
        return index
    if  len(valueList)==0:
        print("valueList大小为0")
        return  0 
    while up-low>1:
         index=int(np.floor((low+up)/2+0.5))
         if value>valueList[index]:
             low=index
         else:
             up=index
    index=low
    return index

def SaveTable(table,flieName,style):
    #保存一张表到csv文件中，表的结构是[[],[],[]...,[]]
    #style解释每个子list代表一列还是代表一行，可选值是字符串'row'和"col"
    with open(flieName+'.csv',mode='w',encoding='UTF-8-sig') as file_obj:
        if style=='row':
            #行模式
            for sublist in table:
                file_obj.write(" ,")
                for elem in sublist:
                    file_obj.write(str(elem)+",")
                file_obj.write("\n") 
        elif style=='col':
            rowNum=len(table[0])
            colNum=len(table)
            for j in range(0,colNum):
                    file_obj.write(" "+",")
            file_obj.write("\n")
            for i in range(0,rowNum):
                for j in range(0,colNum):
                    file_obj.write(str(table[j][i])+",")
                file_obj.write("\n")
        else:
            print("SaveTable:style 参数错误")

def ReadCSVMat(flieName):
    #读取一个矩阵到csv文件中  
    with open(flieName+'.csv',mode='r',encoding='UTF-8-sig') as file_obj:
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


