# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 23:42:31 2019

@author: H.S.Wang
"""

import numpy as np
from sympy import *

#S = symbols('S')
#so = solve(0.225e-1*S/(0.9566326527e-1*S**2+0.5106026784e-2)**2-1/S**2-0.3632812500e-1/S**4, S)

miu = 0.2086353708
alpha = 6.0/5

S0 = 0.8800540903
w02 = 3*(1-miu)*(1-alpha**2)/(S0**3*(1+alpha**2))
e = 0.01
n = np.sqrt((1+miu)/S0**3)

gammaA = 2*w02*e/(w02-n**2)

gm.append([miu,gammaA])

