# -*- coding: utf-8 -*-
"""
Created on Thu Dec 20 16:42:22 2018


此程序用于计算Monodromy矩阵

@author: H.S.Wang
"""


import numpy as np
from myOdeSolver import OdeSolver
import matplotlib.pyplot as plt
import PeriodicOrbit as PO
import Parameters as Pa
import GetX0 as GX0

m,A,B,IzA,K,s = Pa.Parameters()

def Mon(x0,T,YHC):
    """
    Monodromy矩阵计算
    Parameters
    ------------------------------------------------------
    T：float
        积分时间
    x0：array(N*N+N)
        初始状态量
    YHC：def
        右函数
    
    Return
    ----------------------------------------------------
    a：array(float)
        Monodromy矩阵特征值
    b：array(float)
        Monodromy矩阵特征向量
    yy：array(N)
        积分后状态量
    M：array()
        Monodromy矩阵
    
    """
    N=len(x0)#完整运动方程个数
    X=np.zeros(N*N+N)
    M0=np.eye(N)
   
    X[0:N]=x0.copy()
    X[N:]=M0.flatten()
    
    sol = OdeSolver(YHC,[0,T],X)
    y=sol.y[:,-1]
    M=np.array(y[N:N*N+N]).reshape(N,N)
    yy=y[0:N]
    
    #特征值a和特征向量b
    a, b = np.linalg.eig(M)
    
    return a,b,yy,M

def YHC(t,x):
    """
    一维化的Monodromy矩阵和运动方程构成的右函数，3个参数，6个运动方程
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
    thetaA = x[2]
    S1 = x[3]
    theta1 = x[4]
    thetaA1 = x[5]
    
    N=6
    AA=np.zeros((N,N))
    Y=np.zeros(N*N+N)
    
    AA[0,3]=1.0
    AA[1,4]=1.0
    AA[2,5]=1.0
    AA[3,0]=(theta1+thetaA1)**2+2/S**3+(6*(A+B*np.cos(2*theta)))/S**5
    AA[3,1]=3*B*np.sin(2*theta)/S**4
    AA[3,2]=0.0
    AA[3,3]=0.0
    AA[3,4]=AA[3,5]=2*S*(theta1+thetaA1)
    AA[4,0]=5*B*np.sin(2*theta)/S**6+2*S1*(theta1+thetaA1)/S**2+3*m*B*np.sin(2*theta)/(IzA*S**4)
    AA[4,1]=-2*B*np.cos(2*theta)/S**5-2*m*B*np.cos(2*theta)/(IzA*S**3)
    AA[4,2]=0.0
    AA[4,3]=-(2*(theta1+thetaA1))/S
    AA[4,4]=AA[4,5]=-2*S1/S
    AA[5,0]=-3*m*B*np.sin(2*theta)/(IzA*S**4)
    AA[5,1]=2*m*B*np.cos(2*theta)/(IzA*S**3)
    AA[5,2]=AA[5,3]=AA[5,4]=AA[5,5]=0.0
    
    M=np.array(x[N:N*N+N]).reshape(N,N)
    Y[N:N*N+N]=np.dot(AA,M).flatten()
    
    Y[0]=S1
    Y[1]=theta1
    Y[2]=thetaA1
    Y[3]=S*(theta1+thetaA1)**2-1/S**2-(3/2)*(A+B*np.cos(2*theta))/S**4
    Y[4]=-B*np.sin(2*theta)/S**5-2*S1*(theta1+thetaA1)/S-m*B*np.sin(2*theta)/(IzA*S**3)
    Y[5]=m*B*np.sin(2*theta)/(IzA*S**3)
                
    
    return Y

def YHC_2(t,x):
    """
    一维化的Monodromy矩阵和运动方程构成的右函数,2个参数，4个运动方程
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
    
    AA[0,2] = 1.0
    AA[1,3] = 1.0
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

def Ana(a0,at,Num,index):
    
    h = (at-a0)/Num
    Re = []
    dS = []
    Re_list = []
    flag2 = 1
    for i in range(Num):
        print(i)
        x0,T0 = GX0.GetXLS(a0+i*h,0,index)
        x1,T1,flag1 = PO.PeriOrbit_method(T0,x0,PO.YHC,PO.EOM)
        a,b,yy,M = Mon(x1,2*T1,YHC_2)
        Re_list.append(a)
        dS.append(x1[0])
        Re.append(np.trace(M))
    Re = np.array(Re)-2
    yup = np.linspace(2,2,len(dS))
    ydown = np.linspace(-2,-2,len(dS))
    Re_list = np.array(Re_list)
    co = ['blue','yellow']
            
    #画图
    plt.figure()
    plt.title('Stability')
    plt.xlabel('S')
    plt.ylabel('Re')
    plt.scatter(dS[1:],Re[1:],color=co[index])
    plt.plot(dS,yup,color= 'red')
    plt.plot(dS,ydown,color= 'red')
    plt.show()
    #Re = Re.real()
    
    return Re,Re_list,dS

if __name__=="__main__":  
    
    
    index = 0
    x0,T0 = GX0.GetXLS(0.0,0,index)
    #x1,T1,flag = PO.PeriOrbit_method(T0,x0,PO.YHC,PO.EOM)
    a,b,yy,M = Mon(x0,2*np.pi,YHC_2)
    print('eig = ',a,'Tr(M)=',np.trace(M))
    #print('exp(s*2pi/j) = ',np.cos(s[index]*2*np.pi),np.sin(s[index]*2*np.pi))
    
    #Re,Re_list,dS = Ana(0.0,0.03,10,index)
    
    #np.savetxt('data.txt',(M))