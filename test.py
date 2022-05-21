# -*- coding: utf-8 -*-
"""
Created on Sat Jun 15 08:56:08 2019

@author: H.S.Wang
"""
import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import cmath



class Funexpr():
    def __init__(self,m,K,IzA,A,B,theta):
        self.m = m
        self.K = K
        self.IzA = IzA
        self.A = A
        self.B = B
        self.theta = theta
        
    def f(self,S):
        return S*(self.K/(self.m*S**2+self.IzA))**2-1/S**2-3/2/S**4*(self.A+self.B*np.cos(2*self.theta))


Num = 100
record = []
for i in range(Num):
    rs = 100#1+i*100/Num
    
    aA = 6*rs
    bA = 5*rs
    cA = 4*rs
    p = 2000
    a_bar = (aA*bA*cA)**(1/3)
    rp = 1000
    a = rp+aA
    K = 0.15
    MB = 4/3*np.pi*rp**3*p
    MA = 4/3*np.pi*(aA*bA*cA)*p
    m = MA*MB/(MA+MB)**2
    miu = MA/(MA+MB)
    
    alpha = a_bar/a
    J2 = (aA**2+bA**2-2*cA**2)/(10*a_bar**2)
    J22 = (aA**2-bA**2)/(20*a_bar**2)
    A = alpha**2*J2
    B = 6*alpha**2*J22
    IzA = miu*(aA**2+bA**2)/(5*a**2)
    e = 0.01
    
    theta = 0
    Fun = Funexpr(m,K,IzA,A,B,theta)
    S0 = fsolve(Fun.f,[1.0])[0]
    PP = K**2/(S0**2*m+IzA)**2-4*S0**2*K**2*m/(S0**2*m+IzA)**3+2/S0**3+(6*(A+B*np.cos(2*theta)))/S0**5
    QQ = 2*S0*K*IzA/(S0**2*m+IzA)**2
    RR = -2*B*np.cos(2*theta)*(S0**2*m+IzA)/(IzA*S0**5)
    SS = -2*K/(S0*(S0**2*m+IzA))
    
    w = np.zeros(2)
    w[0] = cmath.sqrt((PP+RR+QQ*SS+np.sqrt((QQ*SS+PP+RR)**2-4*PP*RR))*(1/2)).imag
    w[1] = cmath.sqrt((PP+RR+QQ*SS-np.sqrt((QQ*SS+PP+RR)**2-4*PP*RR))*(1/2)).imag
    gamma = 2*w[1]*e/(w[1]-w[0]**2)
    
    alpha1 = aA/bA
    w02 = 3*(1-miu)*(1-alpha1**2)/(S0**3*(1+alpha1**2))
    if i==0:
        print(alpha1)
        print(S0)
    n = np.sqrt((1+miu)/S0**3)
    gammaA = 2*w02*e/(w02-n**2)
    
    record.append([miu,w[0],w[1],w02])
    
recordAry = np.array(record)
plt.plot(recordAry[:,0],recordAry[:,1],color = 'black')
plt.plot(recordAry[:,0],recordAry[:,2],color = 'red')
plt.plot(recordAry[:,0],recordAry[:,3],color = 'orange')
