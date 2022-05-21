# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 16:02:11 2019

此程序是完整的球-椭球1:1共振模型的计算程序
是一个类


@author: H.S.Wang
"""
import numpy as np
from scipy.optimize import fsolve
import cmath
from myOdeSolver import OdeSolver
import matplotlib.pyplot as plt

class SE_Model():

    """
    球-椭球模型的1:1轨旋共振类
    ----------------------------------
        _init_ : callable
                初始化参数

        GetwS0 : callable
                计算长短周期频率和平动点位置函数

        f : callable
                计算平动点位置的函数
    """

    def __init__(self,K,aA,bA,cA,p,rs,rp,theta):

        """
        初始化
        Parameters
        ------------------------------
            K : float
                角动量
                
            aA : float
                椭球半长轴比例

            bA : float
                椭球半长轴比例

            cA : float
                椭球半长轴比例
                
            p : float
                主星和卫星密度

            rs : float
                卫星椭球半径单位    

            rp : float
                主星半径单位

            theta : float
                主星与卫星体固坐标轴夹角 

            a_bar : float
                椭球平均半径

            a : float
                归一化长度单位  

            alpha : float
                归一化的椭球平均半径  

            MA : float
                椭球卫星质量

            MB : float
                球星主星质量

            mu : float
                归一化的椭球卫星质量

            J2 : float
                归一化的非球形引力项
            
            J22 : float
                归一化的非球形引力项

            A : float
                与J2有关的自定变量
            
            B : float
                与J22有关的自定变量

            IzA : float
                z方向的转动惯量
        """

        self.K = K
        self.aA = aA*rs
        self.bA = bA*rs
        self.cA = cA*rs
        self.p = p
        self.rs = rs
        self.rp = rp
        self.theta = theta

        self.a_bar = (self.aA*self.bA*self.cA)**(1/3)
        self.a = rp+self.aA
        self.MA = 4.0/3*np.pi*(self.aA*self.bA*self.cA)*p
        self.MB = 4.0/3*np.pi*self.rp**3*self.p
        self.mu = self.MA/(self.MA+self.MB)
        self.m = self.MA*self.MB/(self.MA+self.MB)**2
        self.alpha = self.a_bar/self.a
        self.J2 = (self.aA**2+self.bA**2-2*self.cA**2)/(10*self.a_bar**2)
        self.J22 = (self.aA**2-self.bA**2)/(20*self.a_bar**2)
        self.A = self.alpha**2*self.J2
        self.B = 6*self.alpha**2*self.J22
        self.IzA = self.mu*(self.aA**2+self.bA**2)/(5*self.a**2)

    def GetwS0(self):
        """
        计算长短周期频率和平动点位置
        function
        ------------------------------
            S0 : float
                平动点位置
            
            w0 : ndarray,float
                长短周期轨频率
            
            PP : float
                系数
            
            QQ : float
                系数

            RR : float
                系数

            SS : float
                系数
        """
        S0 = fsolve(self.f,[1.0])[0] 
        PP = self.K**2/(S0**2*self.m+self.IzA)**2-4*S0**2*self.K**2*self.m/(S0**2*self.m+self.IzA)**3+2/S0**3+(6*(self.A+self.B*np.cos(2*self.theta)))/S0**5
        QQ = 2*S0*self.K*self.IzA/(S0**2*self.m+self.IzA)**2
        RR = -2*self.B*np.cos(2*self.theta)*(S0**2*self.m+self.IzA)/(self.IzA*S0**5)
        SS = -2*self.K/(S0*(S0**2*self.m+self.IzA))
    
        w0 = np.zeros(2)
        w0[0] = cmath.sqrt((PP+RR+QQ*SS+np.sqrt((QQ*SS+PP+RR)**2-4*PP*RR))*(1/2)).imag
        w0[1] = cmath.sqrt((PP+RR+QQ*SS-np.sqrt((QQ*SS+PP+RR)**2-4*PP*RR))*(1/2)).imag

        return w0,S0

    def f(self,S):
        """
        计算平动点位置的方程
        function
        ------------------------------
        """
        return S*(self.K/(self.m*S**2+self.IzA))**2-1/S**2-3/2/S**4*(self.A+self.B*np.cos(2*self.theta))

    def GetXLS(self,a,b,index):
        """
        利用线性化模型计算初始值
        function
        ------------------------------
            Izt : float
                系统总角动量

            x0 : ndarray(4),float
                初始位置速度矢量

            T0 : ndarray(2),float
                初始周期
        Return
        ----------------------------------------------------
            x0 : array(4)
                状态量
            T0 : float
                初始时刻
        """
        w0,S0 = self.GetwS0()
        Izt = self.IzA+self.m*S0**2
        a42 = -2*self.B*Izt/(self.IzA*S0**5)
        a43 = -2*self.K/(S0*Izt)
        a31 = self.K**2*(self.IzA-3*self.m*S0**2)/Izt**3+2/S0**3+6*(self.A+self.B)/S0**5
        a34 = 2*S0*self.K*self.IzA/Izt**2
        
        alpha = -a43*w0[index]/(a42+w0[index]**2)#(a31+s1**2)/(a34*s1) 
        x0 = np.array([S0+a,b,b*w0[index]/alpha,-a*alpha*w0[index]])
        T0 = np.array([np.pi/w0[0],np.pi/w0[1]])
        
        return x0,T0[index]

    def YHC_3(self,t,x):
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
        y[3]=S*(theta1+thetaA1)**2-1/S**2-(3/2)*(self.A+self.B*np.cos(2*theta))/S**4
        y[4]=-self.B*np.sin(2*theta)/S**5-2*S1*(theta1+thetaA1)/S-self.m*self.B*np.sin(2*theta)/(self.IzA*S**3)
        y[5]=self.m*self.B*np.sin(2*theta)/(self.IzA*S**3)
        
        return y

    def test_YHC3(self,T,x,flag = np.array(['Oxy','OSt','E','K'])):
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
            E ：ndarray
                能量数组

            K ：ndarray
                角动量数组

            sol ：OdeResult(5)
                积分结果

            xx ：ndarray
                惯性坐标系下横轴
            
            yy ：ndarray
                惯性坐标系下纵轴
        """  

        #积分运动方程
        sol = OdeSolver(self.YHC_3,[0,T],x)
        
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
            E[i]=self.m/2*(S1[i]**2+S[i]**2*(theta1[i]+thetaA1[i])**2)+1/2*self.IzA*thetaA1[i]**2-self.m/S[i]-self.m/(2*S[i]**3)*(self.A+self.B*np.cos(2*theta[i]))
            K[i]=self.m*S[i]**2*(thetaA1[i]+theta1[i])+self.IzA*thetaA1[i]
        
        N=sol.y.shape[1]
    
        #定义极坐标到平面坐标的变量以及能量角动量
        xx=np.zeros(N)
        yy=np.zeros(N)
        
        
        #转换自变量
        S = sol.y[0,:]
        theta = sol.y[1,:]    
        
            
        thetaA = sol.y[2,:]
        #将S，theta，thetaA转换到平面坐标系，以及求解能量和角动量
        for i in range(N):
            xx[i] = S[i]*np.cos(theta[i]+thetaA[i])
            yy[i] = S[i]*np.sin(theta[i]+thetaA[i])
        
            
        if 'E' in flag:
            
            #绘制能量守恒
            plt.figure()
            plt.plot(E,label='Energy')
            plt.title('Energy')
            plt.xlabel('time')
            plt.ylabel('E')
            #plt.show()
            
        if 'K' in flag:
        
            #绘制角动量守恒
            plt.figure()
            plt.plot(K,label='Angular momentum')
            plt.title('Angular momentum')
            plt.xlabel('time')
            plt.ylabel('K')
            #plt.show()

        if 'OSt' in flag:
            
            #绘制S-theta轨道图
            plt.figure()
            plt.plot(S,theta)
            plt.title('Orbit_St')
            plt.xlabel('S')
            plt.ylabel('theta')
            plt.legend()
            #plt.show()
            
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
        
        return E,K,sol,xx,yy

    def YHC_2(self,t,x):
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
        y[2]=S*(self.IzA*theta1+self.K)**2/(self.m*S**2+self.IzA)**2-1/S**2-3/(2*S**4)*(self.A+self.B*np.cos(2*theta))
        y[3]=-2*S1*(self.IzA*theta1+self.K)/(S*(self.m*S**2+self.IzA))-self.B*np.sin(2*theta)*(self.m*S**2+self.IzA)/(self.IzA*S**5)

        return y

    def test_YHC2(self,T,x,flag = np.array(['Oxy','OSt','E'])):
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
        sol = OdeSolver(self.YHC_2,[0,T],x)
        
        #转换自变量
        S = sol.y[0,:]
        theta = sol.y[1,:]
        S1 = sol.y[2,:]
        theta1 = sol.y[3,:]
        
        #求解系统能量    
        N=sol.y.shape[1]
        E=np.zeros(N)
        for i in range(N):
            E[i]=1/2*self.K**2/(self.m*S[i]**2+self.IzA)+1/2*self.IzA*self.m*S[i]**2*theta1[i]**2/(self.IzA+self.m*S[i]**2)+1/2*self.m*S1[i]**2-self.m/S[i]-self.m*(self.A+self.B*np.cos(2*theta[i]))/(2*S[i]**3)
        
        N=sol.y.shape[1]
    
        #定义极坐标到平面坐标的变量以及能量角动量
        xx=np.zeros(N)
        yy=np.zeros(N)
        
        
        #转换自变量
        S = sol.y[0,:]
        theta = sol.y[1,:]    
    
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
            #plt.show()
            
        if 'OSt' in flag:
            
            #绘制S-theta轨道图
            plt.figure()
            plt.plot(S,theta)
            plt.title('Orbit_St')
            plt.xlabel('S')
            plt.ylabel('theta')
            plt.legend()
            #plt.show()
            
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

        return E,sol,xx,yy

    def Mon(self,x0,T,YHC):
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

    def YHC_M2(self,t,x):
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
        Izt = self.IzA+self.m*S**2
        AA[2,0] = (self.IzA*theta1+K)**2/Izt**2*(1-4*self.m*S**2/Izt)+2/S**3+6*(self.A+self.B*np.cos(2*theta))/S**5
        AA[2,1] = 3*self.B*np.sin(2*theta)/S**4
        AA[2,2] = 0
        AA[2,3] = 2*S*(self.IzA*theta1+K)*self.IzA/Izt**2
        AA[3,0] = 2*S1*(self.IzA*theta1+K)*(2*self.m*S**2+Izt)/(Izt*S)**2+self.B*np.sin(2*theta)*(3*self.m*S**2+5*self.IzA)/(self.IzA*S**6)
        AA[3,1] = -2*self.B*Izt*np.cos(2*theta)/(self.IzA*S**5)
        AA[3,2] = -2*(self.IzA*theta1+K)/Izt/S
        AA[3,3] = -2*S1*self.IzA/Izt/S
        
        M=np.array(x[N:N*N+N]).reshape(N,N)
        Y[N:N*N+N]=np.dot(AA,M).flatten()
        
        Y[0] = S1
        Y[1] = theta1
        Y[2] = S*(self.IzA*theta1+K)**2/(self.IzA+self.m*S**2)**2-1/S**2-3/2/S**4*(self.A+self.B*np.cos(2*theta))
        Y[3] = -2*S1*(self.IzA*theta1+K)/S/(self.IzA+self.m*S**2)-self.B*np.sin(2*theta)*(self.m*S**2+self.IzA)/S**5/self.IzA
        
        return Y

    def PeriOrbit_method(self,T,x,RF,RF_M,EPS=1e-15):
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
            
            a,b,xT,M = self.Mon(x0,T0,RF_M)
            N = len(x0)
            
            x1 = RF(T0,xT[0:N])
            dF = x0-xT
            
            dtheta1 = (x1[2]*dF[1]-x1[1]*dF[2])/(x1[2]*M[1,3]-x1[1]*M[2,3])
            dT = (M[2,3]*dF[1]-M[1,3]*dF[2])/(M[2,3]*x1[1]-M[1,3]*x1[2])
            ep = np.sqrt(dtheta1**2+dT**2)
            x0[3] = x0[3]+dtheta1/(1+beta*ep)
            T0 = T0+dT/(1+beta*ep)
            
            TOF = np.sqrt(dF[1]**2+dF[2]**2)
            
            loopcount +=1

            if loopcount > int(50*(1+beta*ep)):
                flag = 0
                print('Iteration Failed')

                return x,T,flag
                
        if x0[0]<=0 or T0<=0:
            flag = 0
            print('initial value S <=0 or T<=0')

            return x,T,flag
        
        return x0,T0,flag

    def PeriOrbit_method2(self,T,x,RF,RF_M,EPS=1e-14):
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
            
            a,b,xT,M = self.Mon(x0,T0,RF_M)
            
            dF = x0-xT
            
            dtheta1 = (M[1,0]*dF[2]-M[2,0]*dF[1])/(M[1,0]*M[2,3]-M[2,0]*M[1,3])
            dS = (M[2,3]*dF[1]-M[1,3]*dF[2])/(M[2,3]*M[1,0]-M[1,3]*M[2,0])
            ep = np.sqrt(dtheta1**2+dS**2)
            x0[3] = x0[3]+dtheta1/(1+beta*ep)
            x0[0] = x0[0]+dS/(1+beta*ep)
            
            TOF = np.sqrt(dF[1]**2+dF[2]**2)
            
            loopcount +=1

            if loopcount > int(50*(1+beta*ep)) :
                flag = 0
                print('Iteration Failed')

                return x,T,flag
                
        if x0[0]<=0 or T<=0:
            flag = 0
            print('initial value S <=0 or T<=0')

            return x,T,flag
        
        return x0,T0,flag

    def Plot_orbit(self,x0,T,co):
    
        #积分运动方程
        sol = OdeSolver(self.YHC_2,[0,T],x0)

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
        S0 = fsolve(self.f,[1.0])[0]

        #将S，theta，thetaA转换到平面坐标系，以及求解能量和角动量
        for i in range(N):
            xx[i] = S[i]*np.cos(theta[i])
            yy[i] = S[i]*np.sin(theta[i])
            EE[i]=1/2*self.K**2/(self.m*S[i]**2+self.IzA)+1/2*self.IzA*self.m*S[i]**2*theta1[i]**2/(self.IzA+self.m*S[i]**2)+1/2*self.m*S1[i]**2-self.m/S[i]-self.m*(self.A+self.B*np.cos(2*theta[i]))/(2*S[i]**3)

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
        
if __name__=="__main__":

#    K = 0.15
#    aA = 6.0
#    bA = 5.0
#    cA = 4.0
#    p = 2000.0
#    rs = 100.0
#    rp = 1000.0

    rs = 163.0/1.2
    rp = 780.0
    p = 2100.0
    aA = 1.44
    bA = 1.2
    cA = 1.0
    R = 1180.0
    m = (aA*bA*cA)*rs**3*rp**3/(aA*bA*cA*rs**3+rp**3)**2
    mu = aA*bA*cA*rs**3/(aA*bA*cA+rp**3)
    n = 2*np.pi/(11.920*24*60*60)
    Iz = mu*(aA**2+bA**2)*rs**2/(5*R**2)
    K = 0.01#m*R**2*n+Iz*n
    
    theta = 0.0
    SE = SE_Model(K,aA,bA,cA,p,rs,rp,theta)
    w0,S0 = SE.GetwS0()
    x0,T0 = SE.GetXLS(0.01,0.0,0)
    x1,T1,flag1 = SE.PeriOrbit_method(T0,x0,SE.YHC_2,SE.YHC_M2)
    co = np.array(['blue','orange','brown'])
    xx,yy,E,sol = SE.Plot_orbit(x1,2*T1,co[0])
    print('x0 = ',x0)
    print('T0 = ',T0)
    print('x1 = ',x1)
    print('T1 = ',T1)

"""
    a,b,yy,M = SE.Mon(x0,2*np.pi,SE.YHC_M2)
    print('eig = ',a,'Tr(M)=',np.trace(M))

    x=np.zeros(6)
    x[0] = 5
    x[1] = 2.398481534+0.01
    x[2] = 0.0
    x[3] = 0.0
    x[4] = 0.0
    x[5] = 0.1
    T=1000
    E,K,sol,xx,yy = SE.test_YHC3(T,x)
"""
    


