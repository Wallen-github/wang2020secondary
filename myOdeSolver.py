# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 20:40:05 2018

常微分方程的求解，采用RKF78方法，单步长积分，可变步长
"""    

from scipy.optimize import OptimizeResult
import numpy as np

class OdeResult(OptimizeResult):
    pass

###############################################################################
class RKF:
    """
    龙格库塔积分器类(缺省使用RKF78积分器)
    如果使用其他积分器，则继承此类，并覆盖相应的方法
    """

    def __init__(self, fun, t0, x0, t_bound, h_abs, rtol):
        """
        初始化
        Parameters
        ------------------------------
        fun : callable
            ODE 右函数 fun(t,x), x: ndarray,n维向量
            dx / dt = f(t, x)
            x(t0) = x0
            
        t0 : float
            初始时刻  
        x0 : nadrray,shape(n,)            
            自变量初值
        t_bound : float 
            积分终止时刻
            
        h_abs : float
            积分初始步长            
        rtol : float
            积分1步的相对精度            
        """
        self.yhc = fun
        self.t = t0
        self.t_bound = t_bound        
        self.x = np.array(x0)
        self.h_abs = h_abs 
        self.rtol = rtol        
        
        
        # 积分的方向(1：向前；-1：向后)
        self.direction = 1.0
        if t_bound < t0: self.direction =-1        
        
    def run_one_step(self, t,x,h):
        """
        积分1步，步长固定为h(缺省调用rkf78)
        如需其他积分器，可覆盖积分器   
        
        Parameters
        ------------------------------
        yhc : callable
            常微分方程右函数: yhc(t,x)
            t 为时间,
            x为积分自变量
            yhc(t,x)返回 shape(n,)
        
        t : float
            当前时刻
            
        x : ndarray,shape(n,)
            积分自变量
           
        h : float
            步长        
        
        Returns
        -----------------------------
        x_new : ndarray,shape(n,)
            步长h后的新的自变量
        error : float
            本步长，7阶与8阶的差值        
        """
        yhc = self.yhc
        
        y0 = yhc(t, x)
        y1 = yhc(t + (2./27.) * h, x + h * (2. * y0)/27.)
        y2 = yhc(t + (1./9.) * h, x + h * (y0 + 3. * y1)/36.)
        y3 = yhc(t + (1./6.) * h, x + h * (y0 + 3. * y2)/24.)
        y4 = yhc(t + (5./12.) * h, x + h * (20. * y0 + 75. * (-y2 + y3))/48.)
        y5 = yhc(t + (1./2.) * h, x + h * (y0 + 5. * y3 + 4. * y4)/20.)
        y6 = yhc(t + (5./6.) * h, x + h * (-25. * y0 + 125. * y3 - 260. * y4 + 250. * y5)/108.)
        y7 = yhc(t + (1./6.) * h, x + h * (93. * y0 + 244. * y4 - 200. * y5 + 13. * y6)/900.)
        y8 = yhc(t + (2./3.) * h, x + h * (180. * y0 - 795. * y3 + 1408. * y4 - 1070. * y5 + 67. * y6 + 270. * y7)/90.)
        y9 = yhc(t + (1./3.) * h, x + h * (-455. * y0 + 115. * y3 - 3904. * y4 + 3110. * y5 - 171. * y6 + 1530. * y7 - 45. * y8)/540.)
        y10 = yhc(t + h, x + h * (2383. * y0 - 8525. * y3 + 17984. * y4 - 15050. * y5 + 2133. * y6 + 2250. * y7 + 1125. * y8 + 1800. * y9)/4100.)
        y11 = yhc(t, x + h * (60. * y0 - 600. * y5 - 60. * y6 - 300. * y7 + 300. * y8 + 600. * y9)/4100.)
        y12 = yhc(t + h, x + h * (-1777. * y0 - 8525. * y3 + 17984. * y4 - 14450. * y5 + 2193. * y6 + 2550. * y7 + 825. * y8 + 1200. * y9 + 4100. * y11)/4100.)
    
        # 7阶的新的自变量数值
        #x_new = x + h * ( 41./840. * y0 + 34./105. * y5 + 9./35. * y6 + 9./35. * y7 + 9./280. * y8 + 9./280. * y9 + 41./840. * y10)
        # 8阶的新的自变量数值
        x_new = x + h * (272. * y5 + 216. * y6 + 216. * y7 + 27. * y8 + 27. * y9 + 41. * y11 + 41. * y12)/840.
        
        # 7阶与8阶的差值
        error = (y0 + y10 -y11 -y12) * h * 41.0/840.0
        
        return x_new, error

    def step_impl(self):
        """
        变步长积分1步，如果不满足精度，调整步长重新积分
        
        更新self.t/self.x/self.h_abs
        """
        t = self.t
        x = self.x        
        h_abs =self.h_abs

        count = 0
        while True:
            
            # 如果单步循环次数过多，则异常
            count +=1            
            if count > 20:
               Exception("单步寻找次数过多！")
            
            # 当前的步长(含+-号)
            h = h_abs * self.direction
            t_new = t + h

            # 判断是否到达预定的时间边界，如果超过边界，则调整步长刚好到边界
            if self.direction * (t_new - self.t_bound) > 0:
                t_new = self.t_bound

            # 保留当前步长与绝对步长
            h = t_new - t
            h_abs = np.abs(h)
                
            # 当前步长h，往前一步
            x_new, error = self.run_one_step(t,x,h)
                        
            scale = max(abs(x)) #自变量分量中的最大值
            tot = max(abs(error)) # 误差分量重的最大值
            tot = tot / scale #相对误差
    
            # 若当前步长积分的误差太大，则减小步长,重新计算
            if tot > self.rtol:
                h_abs = h_abs * 0.5
                continue
            # 若当前步长积分误差太小，则加大步长，
            if tot < self.rtol * 0.01:
                h_abs = h_abs * 2.0
            
            # 误差正常的话，跳出当前步长计算
            break
            
        self.t = t_new
        self.x = x_new
        self.h_abs = h_abs        

###############################################################################
def OdeSolver(fun, t_span, x0, h_abs = 0.1, rtol=1e-14):
    """常微分方程数值积分器

        dx / dt = f(t, x)
        x(t0) = x0

    Parameters
    ----------
    fun : callable
        ODE 右函数 fun(t,x)
    t_span : 2-tuple of floats
        积分起止时间Interval of integration (t0, tf). The solver starts with t=t0 and
        integrates until it reaches t=tf.
    x0 : array_like, shape (n,)
        自变量初值

    Returns
    -------
    Bunch object with the following fields defined:
    t : ndarray, shape (n_points,)
        Time points.
    y : ndarray, shape (n, n_points)
        Solution values at `t`.
    
    nfev : int
        Number of the system rhs evaluations.
    
    message : string
        Verbal description of the termination reason.
    success : bool
        True if the solver reached the interval end or a termination event
        occurred (``status >= 0``).
    """
  
    t0, tf = float(t_span[0]), float(t_span[1])
    
    # 初始化RKF积分器对象
    solver = RKF(fun, t0, x0, tf, h_abs = h_abs, rtol=rtol)
	
    ts = [t0]
    xs = [x0]     
       
    while True:
             
        # 到达时间边界,正常返回
        if solver.t == solver.t_bound:            
            break

        # 往前走一步            
        solver.step_impl()        

        # 保留当前t,x
        ts.append(solver.t)
        xs.append(solver.x)     
  
    ts = np.array(ts)
    xs = np.vstack(xs).T
    
    return OdeResult(t=ts, y=xs, status=1, message="success", success=1 >= 0)
