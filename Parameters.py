# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 17:26:57 2019

@author: H.S.Wang
"""

import numpy as np
import cmath
from scipy.optimize import fsolve
import sympy as sp

def Parameters0():
    
    m = 0.9257
    K = 2.8382
    IzA = I2z_bar = 0.3434
    I2x_bar = 0.1973
    I2y_bar = 0.2913
    Is_bar = 2.1175
    I1z_bar = 2.4034
    #A = I1z_bar-Is_bar-1/2*(I2x_bar+I2y_bar)+I2z_bar
    A = -1/2*(I2x_bar+I2y_bar)+I2z_bar
    B = 3/2*(-I2x_bar+I2y_bar)
    
    s = np.array([0.03013768121,0.03633157591])
    
    return m,A,B,IzA,K,s

def Parameters1():
    K = 0.35

    aA = 1000
    bA = 900
    cA = 900
    p = 2000
    a_bar = (aA*bA*cA)**(1/3)
    rp = 1500
    a = rp+aA
    
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
    
    s = np.array([0.04485104374,0.08970334021])
    
    return m,A,B,IzA,K,s

def Parameters():
    K = 0.15

    aA = 600
    bA = 500
    cA = 400
    p = 2000
    a_bar = (aA*bA*cA)**(1/3)
    rp = 1000
    a = rp+aA
    
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
    
    s = np.array([0.1814085992,0.2728381413])
    
    return m,A,B,IzA,K,s

def Parameters2():
    K = 0.1

    aA = 600
    bA = 500
    cA = 400
    p = 2000
    a_bar = (aA*bA*cA)**(1/3)
    rp = 1000
    a = rp+aA
    
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
    
    s = np.array([0.6289772586,1.195963894])
    
    return m,A,B,IzA,K,s

def Parameters3():
    K = 0.2

    aA = 600
    bA = 500
    cA = 400
    p = 2000
    a_bar = (aA*bA*cA)**(1/3)
    rp = 1000
    a = rp+aA
    
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
    
    s = np.array([0.07619787753,0.1111579003])
    
    return m,A,B,IzA,K,s

def Parameters4():
    K = 0.25

    aA = 600
    bA = 500
    cA = 400
    p = 2000
    a_bar = (aA*bA*cA)**(1/3)
    rp = 1000
    a = rp+aA
    
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
    
    s = np.array([0.03896966659,0.05638759695])
    
    return m,A,B,IzA,K,s




if __name__=="__main__":
    
    print(Parameters())
    print(Parameters0())
    