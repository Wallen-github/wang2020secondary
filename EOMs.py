# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 11:15:03 2019

运动方程子函数，积分运动轨道和绘制轨道图，能量守恒，角动量守恒
@author: H.S.Wang
"""

import numpy as np
from myOdeSolver import OdeSolver
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
import Parameters as Para

m,A,B,IzA,K,s = Para.Parameters()

def YHC_3(t,x):
    """
    3自由度的运动方程
    Parameters
    ------------------------------------------------------
    t：float
        积分时间
    X：array(4)
        状态量
    
    Return
    ----------------------------------------------------
    Y：array(4)
        状态量
    """
    y = np.zeros(len(x))
    
    S = x[0]
    theta = x[1]
    thetaA = x[2]
    S1 = x[3]
    theta1 = x[4]
    thetaA1 = x[5]
    
    y[0]=S1
    y[1]=theta1
    y[2]=thetaA1
    y[3]=S*(theta1+thetaA1)**2-1/S**2-(3/2)*(A+B*np.cos(2*theta))/S**4
    y[4]=-B*np.sin(2*theta)/S**5-2*S1*(theta1+thetaA1)/S-m*B*np.sin(2*theta)/(IzA*S**3)
    y[5]=m*B*np.sin(2*theta)/(IzA*S**3)
    
    return y




def EOMs_3(T,x):
    """
    3自由度的运动方程下，求解运动轨道，能量和角动量。
    Parameters
    ------------------------------------------------------
    T：float
        积分时间
    x：array(4)
        状态量
        
    Return
    ------------------------------------------------------
    E：array()
        能量数组
    K：array()
        角动量数组
    sol：OdeResult(5)
        积分结果
    """  

    #积分运动方程
    sol = OdeSolver(YHC_3,[0,T],x)
    
    #定义极坐标到平面坐标的变量以及能量角动量
    N=sol.y.shape[1]
    E=np.zeros(N)
    K=np.zeros(N)
    
    #转换自变量
    S = sol.y[0,:]
    theta = sol.y[1,:]
    thetaA = sol.y[2,:]
    S1 = sol.y[3,:] 
    theta1 = sol.y[4,:]
    thetaA1 = sol.y[5,:]

    #将S，theta，thetaA转换到平面坐标系，以及求解能量和角动量
    for i in range(N):
        E[i]=m/2*(S1[i]**2+S[i]**2*(theta1[i]+thetaA1[i])**2)+1/2*IzA*thetaA1[i]**2-m/S[i]-m/(2*S[i]**3)*(A+B*np.cos(2*theta[i]))
        K[i]=m*S[i]**2*(thetaA1[i]+theta1[i])+IzA*thetaA1[i]
      
    return E,K,sol
    
    
def YHC_2(t,x):
    """
    2自由度的运动方程
    Parameters
    ------------------------------------------------------
    t：float
        积分时间
    X：array(4)
        状态量
    
    Return
    ----------------------------------------------------
    Y：array(4)
        状态量
    """
    
    y = np.zeros(len(x))
    
    S = x[0]
    theta = x[1]
    S1 = x[2]
    theta1 = x[3]
    
    y[0]=S1
    y[1]=theta1
    y[2]=S*(IzA*theta1+K)**2/(m*S**2+IzA)**2-1/S**2-3/(2*S**4)*(A+B*np.cos(2*theta))
    y[3]=-2*S1*(IzA*theta1+K)/(S*(m*S**2+IzA))-B*np.sin(2*theta)*(m*S**2+IzA)/(IzA*S**5)

    return y

def EOMs_2(T,x):
    """
    2自由度的运动方程下，求解运动轨道和能量。
    Parameters
    ------------------------------------------------------
    T：float
        积分时间
    x：array(4)
        状态量
    K：float
        角动量
        
    Return
    ------------------------------------------------------
    E：array()
        能量数组
    K：float
        角动量
    sol：OdeResult(5)
        积分结果
    """ 
    
    #积分运动方程
    sol = OdeSolver(YHC_2,[0,T],x)
    
    #转换自变量
    S = sol.y[0,:]
    theta = sol.y[1,:]
    S1 = sol.y[2,:]
    theta1 = sol.y[3,:]
    
    #求解系统能量    
    N=sol.y.shape[1]
    E=np.zeros(N)
    for i in range(N):
        E[i]=1/2*K**2/(m*S[i]**2+IzA)+1/2*IzA*m*S[i]**2*theta1[i]**2/(IzA+m*S[i]**2)+1/2*m*S1[i]**2-m/S[i]-m*(A+B*np.cos(2*theta[i]))/(2*S[i]**3)
        
    return E,K,sol

def EOMplot(E,K,sol,flag = np.array(['Oxy','OSt','E','K'])):
    """
    2自由度的运动方程下，求解运动轨道和能量。
    Parameters
    ------------------------------------------------------
    E：array(float)
        能量数组
    K：float
        角动量
    sol：OdeResult(5)
        积分结果
    flag：array(str)
        绘图指示器，选择要绘制那些图片
        flag=np.array(['Oxy','OSt','E','K'])
        Oxy：画汇合坐标x-y轨道图
        OSt：绘制S-theta轨道图
        E：能量守恒
        K：动量守恒
        
    Return
    ------------------------------------------------------
    图片
    
    """ 
    
    N=sol.y.shape[1]
    
    #定义极坐标到平面坐标的变量以及能量角动量
    xx=np.zeros(N)
    yy=np.zeros(N)
    
    
    #转换自变量
    S = sol.y[0,:]
    theta = sol.y[1,:]    
    
    if len(sol.y[:,0]) == 6:#(转换至x-y的惯性坐标系下)
        
        thetaA = sol.y[2,:]
        #将S，theta，thetaA转换到平面坐标系，以及求解能量和角动量
        for i in range(N):
            xx[i] = S[i]*np.cos(theta[i]+thetaA[i])
            yy[i] = S[i]*np.sin(theta[i]+thetaA[i])
    elif len(sol.y[:,0]) == 4:#(转换至x-y的汇合坐标系下)
        
        #将S，theta，thetaA转换到平面坐标系，以及求解能量和角动量
        for i in range(N):
            xx[i] = S[i]*np.cos(theta[i])
            yy[i] = S[i]*np.sin(theta[i])
    
        
    if 'E' in flag:
        
        #绘制能量守恒
        plt.figure()
        plt.plot(E,label='Energy')
        plt.title('Energy')
        plt.xlabel('time')
        plt.ylabel('E')
        plt.show()
        
    if 'K' in flag:
    
        #绘制角动量守恒
        plt.figure()
        plt.plot(K,label='Angular momentum')
        plt.title('Angular momentum')
        plt.xlabel('time')
        plt.ylabel('K')
        plt.show()
    if 'OSt' in flag:
        
        #绘制S-theta轨道图
        plt.figure()
        plt.plot(S,theta)
        plt.title('Orbit_St')
        plt.xlabel('S')
        plt.ylabel('theta')
        plt.legend()
        plt.show()
        
    if 'Oxy' in flag:
        
        #画汇合坐标x-y轨道图
        plt.figure()
        plt.plot(xx,yy)
        #plt.scatter(S0,0,color='black')
        plt.title('Orbit')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        plt.show()
    
    if len(flag) == 0:
        print('No pic will be show')
        
    return xx,yy
        
def f_1(S):
    theta = 0.0
    return S*(K/(m*S**2+IzA))**2-1/S**2-3/2/S**4*(A+B*np.cos(2*theta))
    
 
if __name__=="__main__":
    
    
    
#    x=np.zeros(4)
#    x[0] = 5
#    x[1] = 0.1#np.pi/2
#    x[2] = 0.0
#    x[3] = 0.0
#
#    T=1000
#    E,K,sol = EOMs_2(T,x)
#    flag=np.array(['E','Oxy'])
#    xx,yy = EOMplot(E,K,sol,flag)
    
    x=np.zeros(6)
    x[0] = 5
    x[1] = 2.398481534+0.01
    x[2] = 0.0
    x[3] = 0.0
    x[4] = 0.0
    x[5] = 0.1
    T=1000
    E,K,sol = EOMs_3(T,x)
    flag=np.array(['E','Oxy','K'])
    EOMplot(E,K,sol,flag)
    
    #数据导出
#    file = open(r"E:\learning\硕士\小行星双星系统\2019.4.22Summary\data\EOM_Case1.txt",'w')
#    lst = []
#    for i in range(len(sol.y[0,:])):
#        lst.append("{} {} {}\n".format(xx[i],yy[i],sol.t[i]))
#        file.writelines(lst)
#    file.close()




