# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 20:33:36 2019

@author: 10365
"""
n=4
nums=[1,2,3,4]
output=nums.copy()
def pailie(numl,index,num_e):
    if(num_e==0):
        print(output)     
        return
    for i in range(num_e):
        output[index]=numl[i]
        numl_n=numl.copy()
        numl_n.pop(i)
        pailie(numl_n,index+1,num_e-1)
    return

pailie(nums,0,n)
        
        