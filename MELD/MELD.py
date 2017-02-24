# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 15:21:45 2015

@author: zhaoshiwen
@email:  sz63@duke.edu

"""

import numpy as np

class MELD:
    def __init__(self, Y, Yt, k, Phi = np.array([])):
        """
        Yt: the type of y_j
            0: categorical: levels are 0,1,2,...
            1: distribution with mean across the real line
            2: distribution with positive mean
        """
        (p,n) = Y.shape
        self.p = p
        self.n = n
        self.d = np.zeros(p, dtype=np.int8)
        for j in range(p):
            if Yt[j] == 0:
                self.d[j] = np.amax(Y[j,:]) + 1
            else:
                self.d[j] = 1
        self.k = k
        self.Yt = np.copy(Yt)

        self.alpha = np.array([0.1]*k)
        self.alpha0 = sum(self.alpha)
        self.lambda2 = self.alpha/(self.alpha0*(self.alpha0+1)) 
        ## the diagonal matrix for second cross moment
        self.lambda3 = 2*self.alpha/(self.alpha0*(self.alpha0+1)*(self.alpha0+2)) 
        
        self.Phi = np.zeros(p,dtype=object)
        self.X = np.zeros(p,dtype=object)
        self.Yd = np.zeros(p,dtype=object)
        for j in range(p):
            if Yt[j] == 0:
                Y_j = np.zeros((n,self.d[j]))
                for i in range(n):
                    Y_j[i,int(Y[j,i])] = 1
                self.Yd[j] = Y_j
                self.Phi[j] = np.zeros((k,self.d[j]))
                self.X[j] = np.zeros((k,self.d[j]))                      
                if Phi.size != 0:
                    for h in range(k):
                        self.Phi[j][h,:] = Phi[j][h,:]
                        self.X[j][h,:] = np.sqrt(self.Phi[j][h,:])
                else:
                    for h in range(k):
                        self.Phi[j][h,:] = np.random.dirichlet([100.0]*self.d[j])
                        self.X[j][h,:] = np.sqrt(self.Phi[j][h,:])
            else:
                self.Yd[j] = np.copy(Y[j,:])
                self.Phi[j] = np.zeros(k)
                if Phi.size != 0:
                    for h in range(k):
                        self.Phi[j][h] = Phi[j][h]

    def calM1(self): # initialize Phi
        self._M1 = np.zeros(self.p,dtype=object)
        for j in range(self.p):
            if self.Yt[j] == 0:
                M1_j = np.zeros(self.d[j])
                for i in range(self.n):
                    M1_j = M1_j + self.Yd[j][i,:]
                M1_j = M1_j/self.n
                self._M1[j] = M1_j
#                for h in range(self.k):
#                    self.Phi[j][h,:] = M1_j[:]
#                    self.X[j][h,:] = np.sqrt(M1_j[:])
            else:
                self._M1[j] = np.sum(self.Yd[j])/self.n
                for h in range(self.k):
                    self.Phi[j][h] = self._M1[j]

    def calM2(self):
        if not hasattr(self,'_M1'):
            self.calM1()
        self._E2 = np.zeros((self.p,self.p),dtype=object)
        self._M2 = np.zeros((self.p,self.p),dtype=object)
        for j1 in range(self.p):
            for j2 in range(self.p):
                E2_j1j2 = np.dot(self.Yd[j1].transpose(),self.Yd[j2])/self.n
                M2_j1j2 = E2_j1j2 - \
                    self.alpha0/(self.alpha0+1)*self._outer(self._M1[j1],self._M1[j2])
                self._E2[j1,j2] = E2_j1j2
                self._M2[j1,j2] = M2_j1j2

    def calM2_bar(self):
        if not hasattr(self, '_M2'):
            self.calM2()
        self._M2_bar = np.zeros((self.p,self.p),dtype=object)
        self._f2d = 0
        for j in range(self.p):
            for t in range(self.p):
                if t != j:
                    M2_bar_jt = self._M2[j,t] - \
                        np.dot(self.Phi[j].transpose()*self.lambda2,self.Phi[t])
                    self._M2_bar[j,t] = M2_bar_jt
                    self._f2d = self._f2d + self.d[j]*self.d[t]
        self._f2d = self._f2d/2

    def calM3(self):
        if not hasattr(self, '_M2'):
            self.calM2_bar()
        # self._E3 = np.zeros((self._p,self._p,self._p),dtype=object)
        # this E3 stores the first two lines of M3 
        self._M3 = np.zeros((self.p,self.p,self.p),dtype=object)
        for j in range(self.p):
            #print j
            mu_j = self._M1[j]
            for s in range(self.p):
                if s != j:
                    mu_s = self._M1[s]
                    for t in range(self.p):
                        if t != s and t != j:
                            mu_t = self._M1[t]
                            E3_jst = self._calE3(self.Yd[j],\
                                self.Yd[s],self.Yd[t])
                            E3_jst = E3_jst - \
                                self.alpha0/(self.alpha0+2)*(\
                                self._calE3bbm(self.Yd[j],self.Yd[s],mu_t) + \
                                self._calE3bmb(self.Yd[j],mu_s,self.Yd[t]) + \
                                self._calE3mbb(mu_j,self.Yd[s],self.Yd[t]))
                            
                            M3_jst = E3_jst/self.n + \
                                2*self.alpha0**2/(self.alpha0+1)/(self.alpha0+2)*\
                                self._outer(mu_j,mu_s,mu_t)
                            self._M3[j,s,t] = M3_jst

    def calM3_bar(self):
        if not hasattr(self, '_M3'):
            self.calM3()
        self._M3_bar = np.zeros((self.p,self.p,self.p),dtype=object)
        self._f3d = 0
        for j in range(self.p):
            Phi_j = self.Phi[j]
            for s in range(self.p):
                if s!=j:
                    Phi_s = self.Phi[s]
                    for t in range(self.p):
                        if t != s and t != j:
                            Phi_t = self.Phi[t]
                            self._M3_bar[j,s,t] = self._mytensorprod(self.lambda3,\
                                Phi_j,Phi_s,Phi_t)
                            self._M3_bar[j,s,t] = self._M3[j,s,t] - \
                                self._M3_bar[j,s,t]
                            self._f3d = self._f3d + self.d[j]*self.d[s]*self.d[t]
        self._f3d = self._f3d/6
        self._f3d = self._f3d + self._f2d
    
    def initializeWeight_M2(self):
        self._W2 = np.zeros((self.p,self.p),dtype=object)
        for j in range(self.p):
            for t in range(self.p):
                if t != j:
                    self._W2[j,t] = np.ones((self.d[j],self.d[t]))
                    if self.d[j] == 1 and self.d[t] == 1:
                        self._W2[j,t] = 1.0
                    elif self.d[j] == 1 or self.d[t] == 1:
                        self._W2[j,t] = np.hstack(self._W2[j,t])

    
    def initializeWeight_M3(self):
        if not hasattr(self,'_W2'):
            self.initializeWeight_M2()
        self._W3 = np.zeros((self.p,self.p,self.p),dtype=object)
        for j in range(self.p):
            for s in range(self.p):
                if s != j:
                    for t in range(self.p):
                        if t!=s and t!=j:
                            self._W3[j,s,t] = np.ones((self.d[j],self.d[s],self.d[t]))
    
    def updateWeight_M2(self):
        for j in range(self.p):
            for t in range(self.p):
                if t != j:
                    E2jt = (np.dot(self.Phi[j].transpose()*self.alpha,self.Phi[t]) + 
                        self._outer(np.dot(self.Phi[j].transpose(),self.alpha),np.dot(self.Phi[t].transpose(),self.alpha)))
                    E2jt = E2jt**2
                    if self.d[j] == 1 and self.d[t] == 1:
                        self._W2[j,t] = np.sum(self.Phi[j]**2 
                            * self.Phi[t]**2 * self.alpha)
                        self._W2[j,t]= (self._W2[j,t] + 
                            np.dot(self.Phi[j]**2,self.alpha)*np.dot(self.Phi[t]**2,self.alpha))
                        self._W2[j,t] = (self._W2[j,t] - E2jt)/self.alpha0/(self.alpha0+1)
                        self._W2[j,t] = 1.0/self._W2[j,t]
                    elif self.d[j] == 1:
                        self._W2[j,t] = np.zeros(self.d[t])
                        for c_t in range(int(self.d[t])):
                            self._W2[j,t][c_t] = np.sum(self.Phi[j]**2 
                                    * self.Phi[t][:,c_t]**2 * self.alpha)
                            self._W2[j,t][c_t] = (self._W2[j,t][c_t] + 
                                np.dot(self.Phi[j]**2,self.alpha)*np.dot(self.Phi[t][:,c_t]**2,self.alpha))
                        self._W2[j,t] = (self._W2[j,t] - E2jt)/self.alpha0/(self.alpha0+1)
                        self._W2[j,t] = 1.0/self._W2[j,t]
                    elif self.d[t] == 1:
                        self._W2[j,t] = np.zeros(self.d[j])
                        for c_j in range(int(self.d[j])):
                            self._W2[j,t][c_j] = np.sum(self.Phi[j][:,c_j]**2 
                                    * self.Phi[t]**2 * self.alpha)
                            self._W2[j,t][c_j] = (self._W2[j,t][c_j] + 
                                np.dot(self.Phi[j][:,c_j]**2,self.alpha)*np.dot(self.Phi[t]**2,self.alpha))
                        self._W2[j,t] = (self._W2[j,t] - E2jt)/self.alpha0/(self.alpha0+1)
                        self._W2[j,t] = 1.0/self._W2[j,t]
                    else:
                        self._W2[j,t] = np.zeros((self.d[j],self.d[t]))
                        for c_j in range(int(self.d[j])):
                            for c_t in range(int(self.d[t])):
                                self._W2[j,t][c_j,c_t] = np.sum(self.Phi[j][:,c_j]**2 
                                    * self.Phi[t][:,c_t]**2 * self.alpha)
                                self._W2[j,t][c_j,c_t] = (self._W2[j,t][c_j,c_t] + 
                                    np.dot(self.Phi[j][:,c_j]**2,self.alpha)*np.dot(self.Phi[t][:,c_t]**2,self.alpha))
                        self._W2[j,t] = (self._W2[j,t] - E2jt)/self.alpha0/(self.alpha0+1)
                        self._W2[j,t] = 1.0/self._W2[j,t]
                    
    def updateWeight_M3(self):
        E2xx = np.diag(self.alpha) + np.outer(self.alpha,self.alpha)
        E2xx = E2xx/self.alpha0/(self.alpha0+1)
        E3xxx = np.zeros((self.k,self.k,self.k))
        for h1 in range(self.k):
            for h2 in range(self.k):
                for h3 in range(self.k):
                    if h1 != h2 and h1 != h3:
                        E3xxx[h1,h2,h3] = self.alpha[h1]*self.alpha[h2]*self.alpha[h3]
                    elif h1 == h2 and h1 != h3:
                        E3xxx[h1,h2,h3] = (self.alpha[h1]+1)*self.alpha[h1]*self.alpha[h3]
                    elif h1 == h3 and h1 != h2:
                        E3xxx[h1,h2,h3] = (self.alpha[h1]+1)*self.alpha[h1]*self.alpha[h2]
                    elif h2 == h3 and h2 != h1:
                        E3xxx[h1,h2,h3] = (self.alpha[h2]+1)*self.alpha[h2]*self.alpha[h1]
                    else: # all equal
                        E3xxx[h1,h2,h3] = (self.alpha[h1]+2)*(self.alpha[h1]+1)*self.alpha[h1]
        E3xxx = E3xxx/self.alpha0/(self.alpha0+1)/(self.alpha0+2)
        
        for j in range(self.p):
            mu_j = np.dot(self.Phi[j].transpose(),self.alpha)/self.alpha0
            for s in range(self.p):
                if s!= j:
                    mu_s = np.dot(self.Phi[s].transpose(),self.alpha)/self.alpha0
                    for t in range(self.p):
                        if t!=s and t!=j:
                            mu_t = np.dot(self.Phi[t].transpose(),self.alpha)/self.alpha0
                            self._W3[j,s,t] = np.zeros((self.d[j],self.d[s],self.d[t]))
                            E3jst = 2*self.alpha0**2/(self.alpha0+1)/(self.alpha0+2)*\
                                self._outer(mu_j,mu_s,mu_t) - self._mytensorprod(self.lambda3,\
                                self.Phi[j],self.Phi[s],self.Phi[t])
                            E3jst = E3jst**2
                            for cj in range(int(self.d[j])):
                                for cs in range(int(self.d[s])):
                                    for ct in range(int(self.d[t])):
                                        Phi3jst = \
                                            self._outer(self.Phi[j][:,cj]**2, self.Phi[s][:,cs]**2, self.Phi[t][:,ct]**2) + \
                                            self._outer(self.Phi[j][:,cj], self.Phi[s][:,cs]**2, self.Phi[t][:,ct]**2)*2*self.alpha0*mu_j[cj]/(self.alpha0+2) + \
                                            self._outer(self.Phi[j][:,cj]**2, self.Phi[s][:,cs], self.Phi[t][:,ct]**2)*2*self.alpha0*mu_s[cs]/(self.alpha0+2) + \
                                            self._outer(self.Phi[j][:,cj]**2, self.Phi[s][:,cs]**2, self.Phi[t][:,ct])*2*self.alpha0*mu_t[ct]/(self.alpha0+2) + \
                                            self._outer(self.Phi[j][:,cj], self.Phi[s][:,cs], self.Phi[t][:,ct]**2)*self.alpha0**2*mu_j[cj]*mu_s[cs]/(self.alpha0+2)**2 + \
                                            self._outer(self.Phi[j][:,cj], self.Phi[s][:,cs]**2, self.Phi[t][:,ct])*self.alpha0**2*mu_j[cj]*mu_t[ct]/(self.alpha0+2)**2 + \
                                            self._outer(self.Phi[j][:,cj]**2, self.Phi[s][:,cs], self.Phi[t][:,ct])*self.alpha0**2*mu_s[cs]*mu_t[ct]/(self.alpha0+2)**2
                                        Phi3jst = Phi3jst*E3xxx
                                        Phi2jst = \
                                            np.outer(self.Phi[j][:,cj]**2, self.Phi[s][:,cs]**2)*self.alpha0**2*mu_t[ct]**2/(self.alpha0+2)**2 + \
                                            np.outer(self.Phi[s][:,cs]**2, self.Phi[t][:,ct]**2)*self.alpha0**2*mu_j[cj]**2/(self.alpha0+2)**2 + \
                                            np.outer(self.Phi[j][:,cj]**2, self.Phi[t][:,ct]**2)*self.alpha0**2*mu_s[cs]**2/(self.alpha0+2)**2
                                        Phi2jst = Phi2jst*E2xx
                                        self._W3[j,s,t][cj,cs,ct] = np.sum(Phi3jst) + np.sum(Phi2jst) - E3jst[cj,cs,ct]
                                        
                            self._W3[j,s,t] = 1.0/self._W3[j,s,t]
                    


    def estimatePhiGrad_M2(self,S, prt = False, step = 1.0):
        if not hasattr(self,'_M2_bar'):
            self.calM2_bar()
        if not hasattr(self,'_W2'):
            self.initializeWeight_M2()
        
        p = self.p
        d = self.d
        k = self.k
        if step == 1.0:
            beta = 1.0
        else:
            beta = 0.6
                            
        iteration = S
        Q2 = [0]*(iteration)
        PHI = np.zeros(iteration,dtype=object)
        for ii in range(iteration):
            if prt:
                print(ii)
    
            for h in range(k):
                for j in range(p):
                    phi_jh = self.Phi[j][h,:] if self.Yt[j] == 0 else self.Phi[j][h]
                    for t in range(p):
                        if t != j:
                            phi_th = self.Phi[t][h,:] if self.Yt[t] == 0 else self.Phi[t][h]
                            self._M2_bar[j,t] = self._M2_bar[j,t] + \
                                self.lambda2[h]*self._outer(phi_jh,phi_th)  
                
                for j in range(p):
                    a2_jh = np.zeros(d[j])
                    b2_jh = np.zeros(d[j])
                    sub_p = range(p)
                    for t in sub_p:
                        if t != j:
                            phi_th = self.Phi[t][h,:] if self.Yt[t] == 0 else self.Phi[t][h]                                
                            #a2_jh = a2_jh + np.dot(self._M2_bar[j,t], phi_th)
                            a2_jh = a2_jh + np.dot(self._M2_bar[j,t]*self._W2[j,t], phi_th)
                            
                            #b2_jh = b2_jh + np.sum(phi_th**2)
                            b2_jh = b2_jh + np.dot(self._W2[j,t],phi_th**2)

                    a2_jh = -2.0*self.lambda2[h] * a2_jh
                    b2_jh = self.lambda2[h]**2 * b2_jh
                    a_jh = a2_jh
                    b_jh = b2_jh
                    
                    e = step*0.5/b_jh
                    if self.Yt[j] == 0:                     
                        self.Phi[j][h,:] = self.Phi[j][h,:] - e*(a_jh + 2.0*b_jh*self.Phi[j][h,:])
                        #print (a_jh + 2*b_jh*self.Phi[j][h,:])
                        self.Phi[j][h,:] = np.abs(self.Phi[j][h,:])/np.sum(np.abs(self.Phi[j][h,:]))
                    elif self.Yt[j] == 1:
                        self.Phi[j][h] = self.Phi[j][h] - e*(a_jh + 2.0*b_jh*self.Phi[j][h])
                    else:
                        self.Phi[j][h] = self.Phi[j][h] - e*(a_jh + 2.0*b_jh*self.Phi[j][h])
                        self.Phi[j][h] = np.abs(self.Phi[j][h])
                # recover M2_bar 
                for j in range(p):
                    phi_jh = self.Phi[j][h,:] if self.Yt[j] == 0 else self.Phi[j][h]
                    for t in range(p):
                        if t != j:
                            phi_th = self.Phi[t][h,:] if self.Yt[t] == 0 else self.Phi[t][h]
                            self._M2_bar[j,t] = self._M2_bar[j,t] - \
                                self.lambda2[h]*self._outer(phi_jh,phi_th) 

            
            PHI[ii] = np.zeros(p,dtype=object)
            diff = 0
            #print j, id(self.Phi[50][0])
            for j in range(p):
                if self.Yt[j] == 0:
                    Phi_j = np.zeros((k,self.d[j]))
                    Phi_j = np.copy(self.Phi[j][:,:])
                    PHI[ii][j] = Phi_j
                else:
                    Phi_j = np.zeros(k)
                    Phi_j[:] = self.Phi[j][:]
                    PHI[ii][j] = Phi_j
                if ii > 0:
                    diff = diff + np.sum((Phi_j - PHI[ii-1][j])**2)
                for t in range(j+1,p):
                    Q2[ii] = Q2[ii] + np.sum((self._M2_bar[j,t])**2*self._W2[j,t])
            
            if ii > 0:
                if abs(Q2[ii] - Q2[ii-1])/self._f2d < 1e-5:
                    return {'Q2': Q2, 'PHI': PHI, 'iter': ii}

            step = step*beta
        
        return {'Q2': Q2, 'PHI': PHI, 'iter': S}


    def estimatePhiGrad_M2M3(self,S, prt = False, step = 1.0):
        if not hasattr(self,'_M2_bar'):
            self.calM2_bar()
        if not hasattr(self,'_W2'):
            self.initializeWeight_M2()
        if not hasattr(self,'_M3_bar'):
            self.calM3_bar()
        if not hasattr(self,'_W3'):
            self.initializeWeight_M3()
        
        
        p = self.p
        d = self.d
        k = self.k
        
        if step == 1.0:
            beta = 1.0
        else:
            beta = 0.6

        iteration = S
        Q3 = [0]*(iteration)
        PHI = np.zeros(iteration,dtype=object)
        for ii in range(iteration):
            if prt :
                print(ii)
    
            for h in range(k):                
                for j in range(p):
                    phi_jh = self.Phi[j][h,:] if self.Yt[j] == 0 else self.Phi[j][h]
                    for t in range(p):
                        if t != j:
                            phi_th = self.Phi[t][h,:] if self.Yt[t] == 0 else self.Phi[t][h]
                            self._M2_bar[j,t] = self._M2_bar[j,t] + \
                                self.lambda2[h]*self._outer(phi_jh,phi_th); 
                            for s in range(p):
                                if s!=t and s!=j:
                                    phi_sh = self.Phi[s][h,:] if self.Yt[s] == 0 else self.Phi[s][h]
                                    self._M3_bar[j,t,s] = self._M3_bar[j,t,s] + \
                                        self.lambda3[h]*self._outer(phi_jh,phi_th,phi_sh)                
                
                
                for j in range(p):
                    a2_jh = np.zeros(d[j])
                    b2_jh = np.zeros(d[j])
                    a3_jh = np.zeros(d[j])
                    b3_jh = np.zeros(d[j])
                    p_sub1 = range(p)
                    for t in p_sub1:
                        if t != j:
                            phi_th = self.Phi[t][h,:] if self.Yt[t] == 0 else self.Phi[t][h]
                            a2_jh = a2_jh + np.dot(self._M2_bar[j,t]*self._W2[j,t], phi_th)                                    
                            #a2_jh = a2_jh + np.dot(self._M2_bar[j,t], phi_th)
                            #b2_jh = b2_jh + np.sum(phi_th**2)
                            b2_jh = b2_jh + np.dot(self._W2[j,t],phi_th**2)
                    
                            p_sub2 = range(p)
                            for s in p_sub2:
                                if s != t and s != j:
                                    phi_sh = self.Phi[s][h,:] if self.Yt[s] == 0 else self.Phi[s][h]
                                    #a3_jh = a3_jh + self._M3phi1phi2(self._M3_bar[j,t,s],
                                    #                                 phi_th,phi_sh)
                                    a3_jh = a3_jh + self._M3phi1phi2(self._M3_bar[j,t,s]*self._W3[j,t,s],
                                                                     phi_th,phi_sh)
                                    #b3_jh = b3_jh + np.sum(phi_th**2)*\
                                    #    np.sum(phi_sh**2)
                                    b3_jh = b3_jh + self._M3phi1phi2(self._W3[j,t,s],phi_th**2,phi_sh**2)
                    
            
                    a2_jh = -2 * self.lambda2[h] * a2_jh
                    b2_jh = self.lambda2[h]**2 * b2_jh
                    a3_jh = -2* self.lambda3[h] * a3_jh
                    b3_jh = self.lambda3[h]**2 * b3_jh
                    a_jh = a2_jh + a3_jh
                    b_jh = b2_jh + b3_jh
                    
                    a_jh = a_jh
                    b_jh = b_jh
                    
                    e = step*0.5/b_jh
                    if self.Yt[j] == 0:                        
                        self.Phi[j][h,:] = self.Phi[j][h,:] - e*(a_jh + 2*b_jh*self.Phi[j][h,:])
                        self.Phi[j][h,:] = np.abs(self.Phi[j][h,:])/np.sum(np.abs(self.Phi[j][h,:]))
                    elif self.Yt[j] == 1:
                        self.Phi[j][h] = self.Phi[j][h] - e*(a_jh + 2*b_jh*self.Phi[j][h])
                    else:
                        self.Phi[j][h] = self.Phi[j][h] - e*(a_jh + 2*b_jh*self.Phi[j][h])
                        self.Phi[j][h] = np.abs(self.Phi[j][h])
                    
                    # recover M2_bar and M3_bar
                for j in range(p):
                    phi_jh = self.Phi[j][h,:] if self.Yt[j] == 0 else self.Phi[j][h] 
                    for t in range(p):
                        if t != j:
                            phi_th = self.Phi[t][h,:] if self.Yt[t] == 0 else self.Phi[t][h] 
                            self._M2_bar[j,t] = self._M2_bar[j,t] - \
                                self.lambda2[h]*self._outer(phi_jh,phi_th);
                            for s in range(p):
                                if s != t and s!=j:
                                    phi_sh = self.Phi[s][h,:] if self.Yt[s] == 0 else self.Phi[s][h] 
                                    self._M3_bar[j,t,s] = self._M3_bar[j,t,s] - \
                                        self.lambda3[h]*self._outer(phi_jh,phi_th,phi_sh)
    
            PHI[ii] = np.zeros(p,dtype=object)
            for j in range(p):
                if self.Yt[j] == 0:
                    Phi_j = np.zeros((k,self.d[j]))
                    Phi_j = np.copy(self.Phi[j][:,:])
                    PHI[ii][j] = Phi_j
                else:
                    Phi_j = np.zeros(k)
                    Phi_j[:] = self.Phi[j][:]
                    PHI[ii][j] = Phi_j
                for t in range(j+1,p):
                    Q3[ii] = Q3[ii] + np.sum(self._M2_bar[j,t]**2*self._W2[j,t])
                    for s in range(t+1,p):
                        Q3[ii] = Q3[ii] + np.sum(self._M3_bar[j,t,s]**2*self._W3[j,t,s])

            if ii > 1:
                if abs(Q3[ii] - Q3[ii-1])/self._f3d < 1e-5:
                    return {'Q3': Q3, 'PHI': PHI,'iter': ii}
            
            step = step * beta
            
        return {'Q3': Q3, 'PHI': PHI, 'iter': S}


    def _outer(self, v1, v2, v3 = np.array([])):
        if v3.size == 0:
            if np.isscalar(v1) or np.isscalar(v2):
                return v1*v2
            else:
                return np.outer(v1,v2)
        else:
            if np.isscalar(v1):
                return v1*self._outer(v2,v3)
            elif np.isscalar(v2):
                return v2*self._outer(v1,v3)
            elif np.isscalar(v3):
                return v3*self._outer(v1,v2)
            else:
                return np.einsum('i,j,k->ijk',v1,v2,v3)
                
    def _calE3(self,Y1,Y2,Y3):
        if Y1.ndim == 1 and Y2.ndim == 2 and Y3.ndim == 2:
            return np.einsum('i,is,it->st',Y1,Y2,Y3)
        elif Y1.ndim == 2 and Y2.ndim == 1 and Y3.ndim == 2:
            return np.einsum('ij,i,it->jt',Y1,Y2,Y3)
        elif Y1.ndim == 2 and Y2.ndim == 2 and Y3.ndim == 1:
            return np.einsum('ij,is,i->js',Y1,Y2,Y3)
        elif Y1.ndim == 1 and Y2.ndim == 1 and Y3.ndim == 2:
            return np.einsum('i,i,it->t',Y1,Y2,Y3)
        elif Y1.ndim == 1 and Y2.ndim == 2 and Y3.ndim == 1:
            return np.einsum('i,is,i->s',Y1,Y2,Y3)
        elif Y1.ndim == 2 and Y2.ndim == 1 and Y3.ndim == 1:
            return np.einsum('ij,i,i->j',Y1,Y2,Y3)
        elif Y1.ndim == 1 and Y2.ndim == 1 and Y3.ndim == 1:
            return np.sum(Y1*Y2*Y3)
        else:
            return np.einsum('ij,is,it->jst',Y1,Y2,Y3)
    
    def _calE3bbm(self,Y1,Y2,mu3):
        if Y1.ndim == 2 and Y2.ndim == 2 and not np.isscalar(mu3):
            return np.einsum('ij,is,t->jst',Y1,Y2,mu3)
        elif Y1.ndim == 2 and Y2.ndim == 2 and np.isscalar(mu3):
            return mu3*np.einsum('ij,is->js',Y1,Y2)
        elif Y1.ndim == 1 and Y2.ndim == 2 and not np.isscalar(mu3):
            return np.einsum('i,is,t->st',Y1,Y2,mu3)
        elif Y1.ndim == 1 and Y2.ndim == 2 and np.isscalar(mu3):
            return mu3*np.einsum('i,is->s',Y1,Y2)
        elif Y1.ndim == 2 and Y2.ndim == 1 and not np.isscalar(mu3):
            return np.einsum('ij,i,t->jt',Y1,Y2,mu3)
        elif Y1.ndim == 2 and Y2.ndim == 1 and np.isscalar(mu3):
            return mu3* np.einsum('ij,i->j',Y1,Y2)
        else:
            return mu3*np.sum(Y1*Y2)
            
    def _calE3bmb(self,Y1,mu2,Y3):
        if Y1.ndim == 2 and Y3.ndim == 2 and not np.isscalar(mu2):
            return np.einsum('ij,s,it->jst',Y1,mu2,Y3)
        elif Y1.ndim == 2 and Y3.ndim == 2 and np.isscalar(mu2):
            return mu2*np.einsum('ij,is->js',Y1,Y3)
        elif Y1.ndim == 1 and Y3.ndim == 2 and not np.isscalar(mu2):
            return np.einsum('i,s,it->st',Y1,mu2,Y3)
        elif Y1.ndim == 1 and Y3.ndim == 2 and np.isscalar(mu2):
            return mu2*np.einsum('i,is->s',Y1,Y3)
        elif Y1.ndim == 2 and Y3.ndim == 1 and not np.isscalar(mu2):
            return np.einsum('ij,s,i->js',Y1,mu2,Y3)
        elif Y1.ndim == 2 and Y3.ndim == 1 and np.isscalar(mu2):
            return mu2* np.einsum('ij,i->j',Y1,Y3)
        else:
            return mu2*np.sum(Y1*Y3)
    
    def _calE3mbb(self,mu1,Y2,Y3):
        if Y2.ndim == 2 and Y3.ndim == 2 and not np.isscalar(mu1):
            return np.einsum('j,is,it->jst',mu1,Y2,Y3)
        elif Y2.ndim == 2 and Y3.ndim == 2 and np.isscalar(mu1):
            return mu1*np.einsum('ij,is->js',Y2,Y3)
        elif Y2.ndim == 1 and Y3.ndim == 2 and not np.isscalar(mu1):
            return np.einsum('j,i,it->jt',mu1,Y2,Y3)
        elif Y2.ndim == 1 and Y3.ndim == 2 and np.isscalar(mu1):
            return mu1*np.einsum('i,is->s',Y2,Y3)
        elif Y2.ndim == 2 and Y3.ndim == 1 and not np.isscalar(mu1):
            return np.einsum('j,is,i->js',mu1,Y2,Y3)
        elif Y2.ndim == 2 and Y3.ndim == 1 and np.isscalar(mu1):
            return mu1* np.einsum('ij,i->j',Y2,Y3)
        else:
            return mu1*np.sum(Y2*Y3)

    # tensor product
    def _myeinsum(self,Phi1,Phi2,Phi3):
        """
        Phi1: d1 by k
        Phi2: d2 by k
        Phi3: d3 by k
        """
        if Phi1.ndim == 2 and Phi2.ndim == 2 and Phi3.ndim == 2:
            return np.einsum('jh,sh,th->jst',Phi1,Phi2,Phi3)
        elif Phi1.ndim == 1 and Phi2.ndim == 2 and Phi3.ndim == 2:
            return np.einsum('h,sh,th->st',Phi1,Phi2,Phi3)
        elif Phi1.ndim == 2 and Phi2.ndim == 1 and Phi3.ndim == 2:
            return np.einsum('jh,h,th->jt',Phi1,Phi2,Phi3)
        elif Phi1.ndim == 2 and Phi2.ndim == 2 and Phi3.ndim == 1:
            return np.einsum('jh,sh,h->js',Phi1,Phi2,Phi3)
        elif Phi1.ndim == 1 and Phi2.ndim == 1 and Phi3.ndim == 2:
            return np.einsum('h,h,th->t',Phi1,Phi2,Phi3)
        elif Phi1.ndim == 1 and Phi2.ndim == 2 and Phi3.ndim == 1:
            return np.einsum('h,sh,h->s',Phi1,Phi2,Phi3)
        elif Phi1.ndim == 2 and Phi2.ndim == 1 and Phi3.ndim == 1:
            return np.einsum('jh,h,h->j',Phi1,Phi2,Phi3)
        elif Phi1.ndim == 1 and Phi2.ndim == 1 and Phi3.ndim == 1:
            return sum(Phi1*Phi2*Phi3)
    
    def _mytensorprod(self,Core,Phi1,Phi2,Phi3):
        """
        Phi1: k by d1
        Phi2: k by d2
        Phi3: k by d3
        """
        if Core.ndim == 1:
            return self._myeinsum(Phi1.transpose()*Core,Phi2.transpose(),Phi3.transpose())
        else:
            print("Error: Core tensor is assumed to be diagonal")
            
    def _M3phi1phi2(self,M3,phi_sh,phi_th):
        """
        calculate M3_jst times_2 phi_sh times_3 phi_th
        """
        if np.isscalar(M3):
            return M3*phi_sh*phi_th
        elif M3.ndim == 1 and np.isscalar(phi_sh) and np.isscalar(phi_th): 
            # M3 is a scalar or vector; the return value is a scalar or vector
            return M3*phi_sh*phi_th
        elif M3.ndim == 1 and phi_sh.ndim == 1 and np.isscalar(phi_th):
            # M3 is a vector
            return np.sum(M3*phi_sh)*phi_th
        elif M3.ndim == 1 and np.isscalar(phi_sh) and phi_th.ndim == 1:
            # M3 is a vector
            return np.sum(M3*phi_th)*phi_sh
        elif M3.ndim == 2 and phi_sh.ndim == 1 and phi_th.ndim == 1:
            return np.einsum('st,s,t',M3,phi_sh,phi_th)
        elif M3.ndim == 2 and phi_sh.ndim == 1 and np.isscalar(phi_th):
            return np.einsum('js,s->j',M3,phi_sh)*phi_th
        elif M3.ndim == 2 and phi_th.ndim == 1 and np.isscalar(phi_sh):
            return np.einsum('jt,t->j',M3,phi_th)*phi_sh
        elif M3.ndim == 3:
            return np.einsum('jst,s,t',M3,phi_sh,phi_th)
