# -*- coding: utf-8 -*-
"""
Created on Fri May 17 10:06:10 2019

@author: H.S.Wang
"""

import numpy as np
import PeriodicOrbit as PO
import Parameters as Pa
import GetX0 as GX0
import GetOE2 as GE
import matplotlib.pyplot as plt
import pandas as pd

def etheta(a0,h,Num):
    
    
    aL = []
    eL = []
    eS = []
    aS = []
    thetaL = []
    thetaS = []
    
    T = 100
    co = np.array(['black','red'])
    i = 0
    a = a0
    while i < Num:
        
        print(i)
        index = 0
        x0,T0 = GX0.GetXLS(a,0,index)
        xx0,yy0,E,sol0 = PO.Plot_orbit(x0,T,co[index])
        aAry0,eAry0,tAry0 = GE.LoopOE(sol0)
        thetaL.append(max(sol0.y[1,:])*180/np.pi)
        aL.append(max(aAry0))
        eL.append(max(eAry0))
        
        index = 1
        x0,T0 = GX0.GetXLS(a,0,index)
        xx0,yy0,E,sol1 = PO.Plot_orbit(x0,T,co[index])
        aAry1,eAry1,tAry1 = GE.LoopOE(sol1)
        thetaS.append(max(sol1.y[1,:])*180/np.pi)
        aS.append(max(aAry1))
        eS.append(max(eAry1))
        i+=1
        a = a0+i*h
        
    
    return aL,eL,aS,eS,thetaL,thetaS

def etheta1(a,T,index):
    
    co = np.array(['black','red'])
    x0,T0 = GX0.GetXLS(a,0,index)
    #xx0,yy0,E,sol0 = PO.Plot_orbit(x0,T,co[index])
    x1,T1,flag1 =PO.PeriOrbit_method(T0,x0,PO.YHC,PO.EOM)
    xx0,yy0,E,sol0 = PO.Plot_orbit(x1,T,co[index])
    aAry0,eAry0,tAry0 = GE.LoopOE(sol0)
    
    #plt.figure()
    plt.scatter(eAry0,sol0.y[1,:],color=co[index])
    plt.xlabel('e')
    plt.ylabel('theta')
    plt.show()
    
    return aAry0,eAry0,sol0.y[1,:]
    
    

def Plot(aL,eL,aS,eS,thetaL,thetaS):
    
    plt.figure()
    plt.scatter(eS,thetaL)
    plt.xlabel('eS')
    plt.ylabel('thetaL')
    plt.show()

if __name__=="__main__":
    
    Num = 10
    a0 = 0.0
    at = 0.03
    h = (at-a0)/Num
#    aL,eL,aS,eS,thetaL,thetaS = etheta(a0,h,Num)
#    Plot(aL,eL,aS,eS,thetaL,thetaS)
    index = 1
    a,e,theta = etheta1(0.005,500,index)
    dfa = pd.DataFrame(a)
    dfe = pd.DataFrame(e)
    dft = pd.DataFrame(theta)
    
#    dfa.to_csv(r"E:\learning\硕士\NJUBachelorPaper\data1\a1.txt")
#    dfe.to_csv(r"E:\learning\硕士\NJUBachelorPaper\data1\e1.txt")
#    dft.to_csv(r"E:\learning\硕士\NJUBachelorPaper\data1\theta1.txt")