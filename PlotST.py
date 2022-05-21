# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 23:20:34 2019

@author: H.S.Wang
"""
import numpy as np
import PeriodicOrbit as PO
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
import Parameters as Pa
import GetX0 as GX0
import Monodromy as Mo
import Convert2 as Con
from myOdeSolver import OdeSolver
import GetOE2 as GE
from numba import jit
import datetime

m,A,B,IzA,K,s = Pa.Parameters()

def f_1(S):
    theta = 0.0
    return S*(K/(m*S**2+IzA))**2-1/S**2-3/2/S**4*(A+B*np.cos(2*theta))

def PO_Ana(a0,at,Num,LS):
    
    x0,T0 = GX0.GetXLS(a0,0,LS)
    #T0 = 70.25
    #x0 = np.array([5.05557,0.0,0.0,0.00602878])
    S = [x0[0]]
    S0 = fsolve(f_1,[2.0])[0]
    TT = [T0]
    Re = [0]
    Re_list = []
    h = (at-a0)/Num/10
    m = 1
    i = 0
    index = []
    x = []
    y = []
    theta = []
    E = []
    aa = []
    ee = []
    t_OE=[]
    
    while i<Num:
        
        print('m =',m,';','i =',i)
        try:
            test = index[-3:]
            if (test[0][1]==test[1][1]) and (test[2][1]==test[1][1]):
                print('Failed')
                break
        except IndexError:
            pass
        
            
        if m==1:

            index.append([m,i])
            print('x0 = ',x0,';','T0 = ',T0)
            x1,T1,flag1 = PO.PeriOrbit_method(T0,x0,PO.YHC,PO.EOM)
            
            if flag1 == 0:
                m = 2
                continue
            else:
                if abs(Re[i]-2)<2:
                    xx,yy,EE,sol = PO.Plot_orbit(x1,2*T1,'blue')
                    x.append(list(xx))
                    y.append(list(yy))
                    theta.append(max(sol.y[1,:]))
                    E.append(EE)
                    aAry,eAry,tAry = GE.LoopOE(sol)
                    aa.append(aAry)
                    ee.append(eAry)
                    t_OE.append(tAry)
                elif abs(Re[i]-2)>=2:
                    xx,yy,EE,sol = PO.Plot_orbit(x1,2*T1,'red')
                    x.append(list(xx))
                    y.append(list(yy))
                    theta.append(max(sol.y[1,:]))
                    E.append(EE)
                    aAry,eAry,tAry = GE.LoopOE(sol)
                    aa.append(aAry)
                    ee.append(eAry)
                    t_OE.append(tAry)
                avector,_,_,M = Mo.Mon(x1,2*T1,Mo.YHC_2)
                Re_list.append(avector)
                Re.append(np.trace(M))
                i = i+1
                a = a0+i*h
                S.append(x1[0])
                TT.append(T1)
                x0,_ = GX0.GetXLS(S[i]+a-S0,0,LS)
                T0 = T1
             
                
        if m == 2:
            
            index.append([m,i])
            print('x0 = ',x0,';','T0 = ',T0)
            x1,T1,flag2 = PO.PeriOrbit_method2(T0,x0,PO.YHC,PO.EOM)
            
            if flag2 == 0 or abs(x1[0]-S0)<h/10:
                m = 1
                h = h/10
                continue
            else:
                if abs(Re[i]-2)<2:
                    xx,yy,EE,sol = PO.Plot_orbit(x1,2*T1,'blue')
                    x.append(list(xx))
                    y.append(list(yy))
                    theta.append(max(sol.y[1,:]))
                    E.append(EE)
                    aAry,eAry,tAry = GE.LoopOE(sol)
                    aa.append(aAry)
                    ee.append(eAry)
                    t_OE.append(tAry)
                elif abs(Re[i]-2)>=2:
                    xx,yy,EE,sol = PO.Plot_orbit(x1,2*T1,'red')
                    x.append(list(xx))
                    y.append(list(yy))
                    theta.append(max(sol.y[1,:]))
                    E.append(EE)
                    aAry,eAry,tAry = GE.LoopOE(sol)
                    aa.append(aAry)
                    ee.append(eAry)
                    t_OE.append(tAry)
                avector,_,_,M = Mo.Mon(x1,2*T1,Mo.YHC_2)
                Re_list.append(avector)
                Re.append(np.trace(M))
                i = i+1
                S.append(x1[0])
                TT.append(T1)
                x0,_ = GX0.GetXLS(x1[0]-S0,0,LS)
                T0 = T1+h
        
    x = list(x)
    y = list(y)
    
    return S,TT,Re,theta,aa,ee,t_OE,E,Re_list,x,y,index


def AnaPlot(S,TT,Re,theta,aa,ee,t_OE,E,Re_list,index):
    
    plt.figure()
    plt.title('S-T')
    plt.xlabel('S')
    plt.ylabel('T')
    plt.scatter(S[2:],T[2:])
    plt.show()
    
    plt.figure()
    plt.title('S-E')
    plt.xlabel('S')
    plt.ylabel('E')
    plt.scatter(S[1:],E)
    plt.show()
    
    plt.figure()
    plt.title('S-theta')
    plt.xlabel('S')
    plt.ylabel('theta')
    plt.scatter(S[1:],theta)
    plt.show()
    
    Re = np.array(Re)-2
    yup = np.linspace(2,2,len(S)-2)
    ydown = np.linspace(-2,-2,len(S)-2)
    Re_list = np.array(Re_list)
    co = ['blue','yellow']
            
    #画图
    plt.figure()
    plt.title('Stability')
    plt.xlabel('S')
    plt.ylabel('Re')
    plt.scatter(S[2:],Re[2:],color=co[index])
    plt.plot(S[2:],yup,color= 'red')
    plt.plot(S[2:],ydown,color= 'red')
    plt.show()

def Savedata(S,TT,Re,theta,aa,ee,t_OE,E,Re_list,x,y,LS):
    
    import pandas as pd
    
    dfx = pd.DataFrame(x)
    dfy = pd.DataFrame(y)
    dfa = pd.DataFrame(aa)
    dfe = pd.DataFrame(ee)
    dft = pd.DataFrame(t_OE)
    data = []
    data.append(S)
    data.append(T)
    data.append(E)
    data.append(Re)
    data.append(theta)
    dfdata = pd.DataFrame(data)
    
    if LS == 1:
        dfx.to_csv(r"E:\learning\硕士\NJUBachelorPaper\data\POS\POSx.txt")
        dfy.to_csv(r"E:\learning\硕士\NJUBachelorPaper\data\POS\POSy.txt")
        dfa.to_csv(r"E:\learning\硕士\NJUBachelorPaper\data\POS\a.txt")
        dfe.to_csv(r"E:\learning\硕士\NJUBachelorPaper\data\POS\e.txt")
        dfdata.to_csv(r"E:\learning\硕士\NJUBachelorPaper\data\POS\STERetheta.txt")
        dft.to_csv(r"E:\learning\硕士\NJUBachelorPaper\data\POS\tOE.txt")
        
    elif LS == 0:
        dfx.to_csv(r"E:\learning\硕士\NJUBachelorPaper\data\POL\POLx2.txt")
        dfy.to_csv(r"E:\learning\硕士\NJUBachelorPaper\data\POL\POLy2.txt")
        dfa.to_csv(r"E:\learning\硕士\NJUBachelorPaper\data\POL\a2.txt")
        dfe.to_csv(r"E:\learning\硕士\NJUBachelorPaper\data\POL\e2.txt")
        dfdata.to_csv(r"E:\learning\硕士\NJUBachelorPaper\data\POL\STERetheta2.txt")
        dft.to_csv(r"E:\learning\硕士\NJUBachelorPaper\data\POL\tOE2.txt")
    

if __name__=="__main__":
    
    
    begin = datetime.datetime.now()
    Num = 10
    LS = 0
    #S,T,Re,theta,E,Re_list,x,y,index = PO_Ana(0.01,0.035,Num,LS)
    S,T,Re,theta,aa,ee,t_OE,E,Re_list,x,y,index = PO_Ana(0.0,1,Num,LS)
    
    end = datetime.datetime.now()
    print('time = ',end-begin)
    #AnaPlot(S,T,Re,theta,aa,ee,t_OE,E,Re_list,LS)
    #Savedata(S,T,Re,theta,aa,ee,t_OE,E,Re_list,x,y,LS)
    
    plt.figure()
    plt.title('S-T')
    plt.xlabel('S')
    plt.ylabel('T')
    plt.scatter(S[1:],T[1:])
    plt.show()
    
    
    
    
    
            
            
                
                
            
    
        
    