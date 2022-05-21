# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 11:36:52 2019

@author: H.S.Wang
"""

import numpy as np
from scipy.optimize import fsolve
from myOdeSolver import OdeSolver
import Monodromy as Mo
import matplotlib.pyplot as plt
import Parameters as Pa
import GetX0 as GX0

m,A,B,IzA,K,s = Pa.Parameters()

def PeriOrbit_method(T,x,RF,RF2,EPS=1e-15):
    """
    周期轨子函数
    Parameters
    ---------------------------------------------------------
    T0：float
        粗略估计的一个周期时间
    x0：float，array(N)
        初始状态量
    RF：def
        一维化的Monodromy矩阵和运动方程构成的右函数
    RF2：def
        运动方程构成的右函数
    TOF：float
        误差限
    
    Return
    ---------------------------------------------------------
    x：float，array(N)
        修正后的状态量
    T：float
        修正后的一个周期时间

    """
    flag = 1
    loopcount = 0
    TOF = 1.0
    x0 = x.copy()
    T0 = T
    beta = 1
    
    while TOF>EPS:
        
        a,b,xT,M = Mo.Mon(x0,T0,YHC)
        N = len(x0)
        
        x1 = EOM(T0,xT[0:N])
        dF = x0-xT
        
        dtheta1 = (x1[2]*dF[1]-x1[1]*dF[2])/(x1[2]*M[1,3]-x1[1]*M[2,3])
        dT = (M[2,3]*dF[1]-M[1,3]*dF[2])/(M[2,3]*x1[1]-M[1,3]*x1[2])
        ep = np.sqrt(dtheta1**2+dT**2)
        x0[3] = x0[3]+dtheta1/(1+beta*ep)
        T0 = T0+dT/(1+beta*ep)
        
        TOF = np.sqrt(dF[1]**2+dF[2]**2)
        
        loopcount +=1
        #print('TOF=',TOF,';','T0=',T0)
        #print('x0 = ',x0,';','T0 = ',T0,';','TOF=',TOF)
        if loopcount > int(50*(1+beta*ep)):
            flag = 0
            print('Iteration Failed')
            #print('x0 = ',x0,';','T0 = ',T0)
            return x,T,flag
            break
    if x0[0]<=0 or T0<=0:
        flag = 0
        print('initial value S <=0 or T<=0')
        #print('x0 = ',x0,';','T0 = ',T0)
        return x,T,flag
    
    return x0,T0,flag

def PeriOrbit_method2(T,x,RF,RF2,EPS=1e-14):
    """
    周期轨子函数
    Parameters
    ---------------------------------------------------------
    T：float
        粗略估计的一个周期时间
    x0：float，array(N)
        初始状态量
    RF：def
        一维化的Monodromy矩阵和运动方程构成的右函数
    RF2：def
        运动方程构成的右函数
    TOF：float
        误差限
    
    Return
    ---------------------------------------------------------
    x：float，array(N)
        修正后的状态量
    T：float
        修正后的一个周期时间

    """
    flag = 1
    loopcount = 0
    TOF = 1.0
    x0 = x.copy()
    T0 = T
    beta = 1
    
    while TOF>EPS:
        
        a,b,xT,M = Mo.Mon(x0,T0,YHC)
        
        dF = x0-xT
        
        dtheta1 = (M[1,0]*dF[2]-M[2,0]*dF[1])/(M[1,0]*M[2,3]-M[2,0]*M[1,3])
        dS = (M[2,3]*dF[1]-M[1,3]*dF[2])/(M[2,3]*M[1,0]-M[1,3]*M[2,0])
        ep = np.sqrt(dtheta1**2+dS**2)
        x0[3] = x0[3]+dtheta1/(1+beta*ep)
        x0[0] = x0[0]+dS/(1+beta*ep)
        
        TOF = np.sqrt(dF[1]**2+dF[2]**2)
        
        loopcount +=1
        #print('Loopcount = ',loopcount,';','TOF =',TOF)
        #print('x0 = ',x0,';','T0 =',T0,'TOF =',TOF)
        if loopcount > int(50*(1+beta*ep)) :
            flag = 0
            print('Iteration Failed')
            #print('x0 = ',x0,';','T0 = ',T0)
            return x,T,flag
            break
    if x0[0]<=0 or T<=0:
        flag = 0
        print('initial value S <=0 or T<=0')
        #print('x0 = ',x0,';','T0 = ',T0)
        return x,T,flag
    
    return x0,T0,flag
 
def EOM(t,x):
    """
    2自由度的运动方程
    Parameters
    ------------------------------------------------------
    t：float
        积分时间
    X：array(N*N+N)
        状态量
    
    Return
    ----------------------------------------------------
    Y：array(N*N+N)
        状态量
    """
    
    y = np.zeros(len(x))
    
    S = x[0]
    theta = x[1]
    S1 = x[2]
    theta1 = x[3]

    
    y[0] = S1
    y[1] = theta1
    y[2] = S*(IzA*theta1+K)**2/(IzA+m*S**2)**2-1/S**2-3/2/S**4*(A+B*np.cos(2*theta))
    y[3] = -2*S1*(IzA*theta1+K)/S/(IzA+m*S**2)-B*np.sin(2*theta)*(m*S**2+IzA)/S**5/IzA
 
    return y

def YHC(t,x):
    """
    一维化的变分矩阵矩阵和运动方程构成的右函数，消去thetaA的运动方程，仅两维
    Parameters
    ------------------------------------------------------
    t：float
        积分时间
    X：array(N*N+N)
        状态量
    
    Return
    ----------------------------------------------------
    Y：array(N*N+N)
        状态量
    """
    
    S = x[0]
    theta = x[1]
    S1 = x[2]
    theta1 = x[3]
    
    N=4
    AA=np.zeros((N,N))
    Y=np.zeros(N*N+N)
    
    for i in range(2):
        AA[i,i+2]=1.0
     
    Izt = IzA+m*S**2
    AA[2,0] = (IzA*theta1+K)**2/Izt**2*(1-4*m*S**2/Izt)+2/S**3+6*(A+B*np.cos(2*theta))/S**5
    AA[2,1] = 3*B*np.sin(2*theta)/S**4
    AA[2,2] = 0
    AA[2,3] = 2*S*(IzA*theta1+K)*IzA/Izt**2
    AA[3,0] = 2*S1*(IzA*theta1+K)*(2*m*S**2+Izt)/(Izt*S)**2+B*np.sin(2*theta)*(3*m*S**2+5*IzA)/(IzA*S**6)
    AA[3,1] = -2*B*Izt*np.cos(2*theta)/(IzA*S**5)
    AA[3,2] = -2*(IzA*theta1+K)/Izt/S
    AA[3,3] = -2*S1*IzA/Izt/S
    
    M=np.array(x[N:N*N+N]).reshape(N,N)
    Y[N:N*N+N]=np.dot(AA,M).flatten()
    
    Y[0] = S1
    Y[1] = theta1
    Y[2] = S*(IzA*theta1+K)**2/(IzA+m*S**2)**2-1/S**2-3/2/S**4*(A+B*np.cos(2*theta))
    Y[3] = -2*S1*(IzA*theta1+K)/S/(IzA+m*S**2)-B*np.sin(2*theta)*(m*S**2+IzA)/S**5/IzA
    
    return Y

def Plot_orbit(x0,T,co):
    
    #积分运动方程
    sol = OdeSolver(EOM,[0,T],x0)

    N=sol.y.shape[1]
    
    #定义极坐标到平面坐标的变量以及能量角动量
    xx=np.zeros(N)
    yy=np.zeros(N)
    EE=np.zeros(N)

    
    #转换自变量
    S = sol.y[0,:]
    theta = sol.y[1,:]
    S1 = sol.y[2,:] 
    theta1 = sol.y[3,:]
    S0 = fsolve(f_1,[1.0])[0]

    #将S，theta，thetaA转换到平面坐标系，以及求解能量和角动量
    for i in range(N):
        xx[i] = S[i]*np.cos(theta[i])
        yy[i] = S[i]*np.sin(theta[i])
        EE[i]=1/2*K**2/(m*S[i]**2+IzA)+1/2*IzA*m*S[i]**2*theta1[i]**2/(IzA+m*S[i]**2)+1/2*m*S1[i]**2-m/S[i]-m*(A+B*np.cos(2*theta[i]))/(2*S[i]**3)

    E = np.mean(EE)
     #画图
    #plt.figure()
    plt.plot(xx,yy,color=co)
    plt.scatter(S0,0,color='black')
    plt.title('Orbit')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
    
    #    #画图
    #    plt.figure()
    #    plt.plot(theta)
    #    plt.title('Orbit_1')
    #    plt.ylabel('theta')
    #    plt.show()
    #    
    #    #画图
    #    plt.figure()
    #    plt.plot(EE)
    #    plt.title('energy')
    #    plt.ylabel('E')
    #    plt.show()
    
    return xx,yy,E,sol

def f_1(S):
    theta = 0.0
    return S*(K/(m*S**2+IzA))**2-1/S**2-3/2/S**4*(A+B*np.cos(2*theta))


if __name__=="__main__":
    
    flag1 = flag2 = 0
    
    index = 0
    a = 0.1#短周期1.6是极限#长周期0.03是极限
    b = 0.0
    co = np.array(['blue','orange','brown'])
    x0,T0 = GX0.GetXLS(a,b,index)
    xx0,yy0,E,sol0 = Plot_orbit(x0,2*T0,'black')
    x1,T1,flag1 =PeriOrbit_method(T0+1,x0,YHC,EOM)
    print('x0 = ',x0)
    print('T0 = ',T0)
    print('x1 = ',x1)
    print('T1 = ',T1)
    
    
#    print('x1 = ',x1,';','T1 = ',T1)
#    a,b,yy,M = Mo.Mon(x1,2*T1,Mo.YHC_2)
#    print('Tr(M)-2=',np.trace(M)-2)
    #x2,T2,flag2 = PeriOrbit_method2(T1,x0,YHC,EOM)
    
    if flag1 == 1:
        xx,yy,E,sol = Plot_orbit(x1,2*T1,co[0])
    if flag2 == 1:
        xx,yy,E,sol = Plot_orbit(x2,2*T2,co[1])
  
    #导出轨道数据画图
#xL = []
#yL = []
#xL.append(list(xx))
#yL.append(list(yy))
#import pandas as pd
#dfx = pd.DataFrame(xL)
#dfy = pd.DataFrame(yL)
#dfx.to_csv(r"E:\learning\硕士\NJUBachelorPaper\data\xS.txt")
#dfy.to_csv(r"E:\learning\硕士\NJUBachelorPaper\data\yS.txt")
#dfx = pd.DataFrame(xL0)
#dfy = pd.DataFrame(yL0)
#dfx.to_csv(r"E:\learning\硕士\NJUBachelorPaper\data\xS0.txt")
#dfy.to_csv(r"E:\learning\硕士\NJUBachelorPaper\data\yS0.txt")
    
#xL0 = []
#yL0 = []
#xL0.append(list(xx0))
#yL0.append(list(yy0))

    
    