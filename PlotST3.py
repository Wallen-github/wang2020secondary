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


m,A,B,IzA,K,s = Pa.Parameters()

def f_1(S):
    theta = 0.0
    return S*(K/(m*S**2+IzA))**2-1/S**2-3/2/S**4*(A+B*np.cos(2*theta))

def PO_Ana(x0,T0,h,Num,LS):
    
    m = 1
    i = 0
    index = []
    S = [x0[0]]
    S0 = fsolve(f_1,[1.0])[0]
    TT = [T0]
    Re = [0]
    Re_list = []

    
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
                avector,b,_,M = Mo.Mon(x1,2*T1,Mo.YHC_2)
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
                avector,b,_,M = Mo.Mon(x1,2*T1,Mo.YHC_2)
                Re_list.append(avector)
                Re.append(np.trace(M))
                i = i+1
                S.append(x1[0])
                TT.append(T1)
                x0,_ = GX0.GetXLS(x1[0]-S0,0,LS)
                T0 = T1+h
        
    
    return S,TT,Re,Re_list,index


def AnaPlot(S,TT,LS):
    
    co = ['black','red']
    #plt.figure()
    plt.title('S-T')
    plt.xlabel('S')
    plt.ylabel('T')
    plt.scatter(S[2:],T[2:],color=co[LS])
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
        dfx.to_csv(r"E:\learning\硕士\NJUBachelorPaper\data1\POS\POSx.txt")
        dfy.to_csv(r"E:\learning\硕士\NJUBachelorPaper\data1\POS\POSy.txt")
        dfa.to_csv(r"E:\learning\硕士\NJUBachelorPaper\data1\POS\a.txt")
        dfe.to_csv(r"E:\learning\硕士\NJUBachelorPaper\data1\POS\e.txt")
        dfdata.to_csv(r"E:\learning\硕士\NJUBachelorPaper\data1\POS\STERetheta.txt")
        dft.to_csv(r"E:\learning\硕士\NJUBachelorPaper\data1\POS\tOE.txt")
        
    elif LS == 0:
        dfx.to_csv(r"E:\learning\硕士\NJUBachelorPaper\data1\POL\POLx3.txt")
        dfy.to_csv(r"E:\learning\硕士\NJUBachelorPaper\data1\POL\POLy3.txt")
        dfa.to_csv(r"E:\learning\硕士\NJUBachelorPaper\data1\POL\a3.txt")
        dfe.to_csv(r"E:\learning\硕士\NJUBachelorPaper\data1\POL\e3.txt")
        dfdata.to_csv(r"E:\learning\硕士\NJUBachelorPaper\data1\POL\STERetheta3.txt")
        dft.to_csv(r"E:\learning\硕士\NJUBachelorPaper\data1\POL\tOE3.txt")
    

if __name__=="__main__":
    
    Num = 100
    a0 = 0.0
    at = 2
    h = (at-a0)/Num
    LS = 0
    #x0,T0 = GX0.GetXLS(a0,0,LS)
    x0 = np.array([2.44206447,0,0,0.1333075])
    T0 = 19.69853296422553
    S,T,Re,Re_list,index = PO_Ana(x0,T0,h,Num,LS)
    AnaPlot(S,T,LS)
    #Savedata(S,T,Re,theta,aa,ee,t_OE,E,Re_list,x,y,LS)
    

    
    
    
    
            
            
                
                
            
    
        
    