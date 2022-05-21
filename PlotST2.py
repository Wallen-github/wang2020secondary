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

m,A,B,IzA,K,s = Pa.Parameters()

def f_1(S):
    theta = 0.0
    return S*(K/(m*S**2+IzA))**2-1/S**2-3/2/S**4*(A+B*np.cos(2*theta))

def PO_Ana(x0,T0,h,Num,LS):
    
    m = 2
    i = 0
    index = []
    x = []
    y = []
    S = [x0[0]]
    S0 = fsolve(f_1,[1.0])[0]
    TT = [T0]
    Re = [0]
    Re_list = []
    theta = [0]
    E = [0]
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
                avector,b,_,M = Mo.Mon(x1,2*T1,Mo.YHC_2)
                Re_list.append(avector)
                Re.append(np.trace(M))
                i = i+1
                #a = a0+i*h
                S.append(x1[0])
                TT.append(T1)
                x0,_ = GX0.GetXLS(S[i]+h-S0,0,LS)
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
                avector,b,_,M = Mo.Mon(x1,2*T1,Mo.YHC_2)
                Re_list.append(avector)
                Re.append(np.trace(M))
                i = i+1
                S.append(x1[0])
                TT.append(T1)
                x0,_ = GX0.GetXLS(x1[0]-S0,0,LS)
                T0 = T1+h/60
        
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
    plt.scatter(S[1:],E[1:])
    plt.show()
    
    plt.figure()
    plt.title('S-theta')
    plt.xlabel('S')
    plt.ylabel('theta')
    plt.scatter(S[1:],theta[1:])
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
        dfx.to_csv(r"E:\learning\硕士\NJUBachelorPaper\data1\POS\POSx.txt")
        dfy.to_csv(r"E:\learning\硕士\NJUBachelorPaper\data1\POS\POSy.txt")
        dfa.to_csv(r"E:\learning\硕士\NJUBachelorPaper\data1\POS\a.txt")
        dfe.to_csv(r"E:\learning\硕士\NJUBachelorPaper\data1\POS\e.txt")
        dfdata.to_csv(r"E:\learning\硕士\NJUBachelorPaper\data1\POS\STERetheta.txt")
        dft.to_csv(r"E:\learning\硕士\NJUBachelorPaper\data1\POS\tOE.txt")
        
    elif LS == 0:
        dfx.to_csv(r"E:\learning\硕士\NJUBachelorPaper\data1\POLdetail\POLxd.txt")
        dfy.to_csv(r"E:\learning\硕士\NJUBachelorPaper\data1\POLdetail\POLyd.txt")
        dfa.to_csv(r"E:\learning\硕士\NJUBachelorPaper\data1\POLdetail\ad.txt")
        dfe.to_csv(r"E:\learning\硕士\NJUBachelorPaper\data1\POLdetail\ed.txt")
        dfdata.to_csv(r"E:\learning\硕士\NJUBachelorPaper\data1\POLdetail\STERethetad.txt")
        dft.to_csv(r"E:\learning\硕士\NJUBachelorPaper\data1\POLdetail\tOEd.txt")
    

if __name__=="__main__":
    
    Num = 200
    LS = 0
    a0 = 0.0
    at = 4
    h = (at-a0)/Num
    #x0,T0 = GX0.GetXLS(a0,0,LS)
    x0 = np.array([4.45332937,0,0,6.28517997])
    T0 =  33.54091517146147
    S,T,Re,theta,aa,ee,t_OE,E,Re_list,x,y,index = PO_Ana(x0,T0,h,Num,LS)
    AnaPlot(S,T,Re,theta,aa,ee,t_OE,E,Re_list,LS)
    #Savedata(S,T,Re,theta,aa,ee,t_OE,E,Re_list,x,y,LS)

    
    
    
    
            
            
                
                
            
    
        
    