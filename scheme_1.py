#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 17:00:34 2017

@author: lorenzolmp

In this script the functions needed to evolve Euler's equation are defined
"""

import numpy as np
import scipy.sparse as sp
cs = 1
G = 10.0
tol = 1E-4

def sigmoid(x):
    return np.divide(1,1+np.exp(-x))

""" initialize_vec combines the density and current into one single vector as the equations are coupled. 
One ghost cell is added to both and left sides of the domain. This cell will make calculations easier."""
def initialize_vec(vec_rho, vec_q):
    vec_rho = np.reshape(vec_rho, [-1, 1]);vec_q = np.reshape(vec_q, [-1, 1]);N = np.shape(vec_rho)[0]
    vec_rho = np.vstack([vec_rho[0], vec_rho, vec_rho[N-1]])
    vec_q = np.vstack([-vec_q[0], vec_q, -vec_q[N-1]])
    return np.reshape(np.hstack((vec_rho, vec_q)), [-1,1])

""" decompose vec decouples the density and current"""
def decompose_vec(vec):
    vec = np.reshape(vec, [-1,2]); M = np.shape(vec)[0]
    return np.reshape(vec[1:M-1,0], [-1,1]), np.reshape(vec[1:M-1,1], [-1,1])

""" In func, the vector function is evaluated on the euler's equations: f:[rho,q] = [q, q^2/rho + cs^2*rho].
We enforce the physical condition that where rho=0 (void) also the current is zero."""
def func(vec):
    rho, q = decompose_vec(vec)
    q[abs(rho)<=tol] = 0
    f2 = np.divide(np.power(q,2),rho, where=rho>0) + cs**2*rho
    return initialize_vec(q, f2)
    
""" The minmod slope limiter checks that the linear reconstruction inside each cell preserves the total TVD property.
We use raise and lower sparse matrices to keep working with the whole vector of the solution. 
This is because, e.g. we want to subtract the u_min of cell i with u_plus of cell i+1. """
def minmod_gen(vec):
    N = np.shape(vec)[0]
    S1 = sp.spdiags(np.ones(N), -2, N,N);S2 = sp.spdiags(np.ones(N), 2, N,N)
    a = vec - S1.dot(vec); b = S2.dot(vec)-vec;
    du = np.sign(a)*np.minimum(abs(a), abs(b))*(np.sign(a)==np.sign(b))
    du[0] = du[2];du[1] = -du[3]; du[N-2] = du[N-4]; du[N-1] = -du[N-3]
    du = np.reshape(du, [-1,1])
    return du

""" The muscl function is what actual reconstruct the solution in the cell. 
u_plus is the right side of the interface (left side of the cell) """
def muscl_s(vec):
    du = minmod_gen(vec)
    u_plus = vec - 0.5*du
    u_min = vec + 0.5*du
    return u_plus, u_min

""" F returns the numerical flux as defined in the text. 
The closed boundary conditions are enforced"""
def F(vec1, vec2, damp0, damp1, flux):
    N = np.shape(vec1)[0]
    if flux == 'LF':
        vec = 0.5*(func(vec2) + func(vec1)) - 0.5*(vec2-vec1)*np.maximum(damp0, damp1)
    vec[0] = vec[2];vec[1] = -vec[3]; vec[N-2] = vec[N-4]; vec[N-1] = -vec[N-3]
    return vec


""" DAMP returns an upper bound for the largest eigenvalue of the flux jacobian over all the mesh """
def DAMP(vec):
    N = np.shape(vec)[0]
    vec = np.reshape(vec, [-1,2])
    d_nonzero = np.reshape(vec[:,0][np.nonzero(abs(vec[:,0]-tol))], [-1,1]); 
    d = np.abs(np.divide(np.reshape(vec[0:np.shape(d_nonzero)[0],1], [-1,1]), d_nonzero)) + cs
    d = np.reshape(d, [-1,1])
    return np.vstack((np.reshape(np.hstack([d,d]), [-1,1]), np.zeros([N-2*len(d),1])))

""" L is the linear operator as defined for the Runge-Kutta iterative process"""
def L(vec,t,mode,flux):
    N = np.shape(vec)[0]; S1 = sp.spdiags(np.ones(N), 2, N,N); S2 = sp.spdiags(np.ones(N), -2, N,N)
    u_plus, u_min = muscl_s(vec); 
    deltax = 4/(N-4)
    vec1 = - (F(u_min, S1.dot(u_plus), DAMP(vec), S1.dot(DAMP(vec)),flux) - F(S2.dot(u_min), u_plus, S2.dot(DAMP(vec)), DAMP(vec),flux))/deltax
    vec1 += finitex_source(t,(N-4)/2, vec, mode)
    return vec1

""" SSPRK3 is the iterative runge kutta strong stability preserving"""
def SSPRK3(vec,k,t,mode,flux):
    u1 = vec + k*L(vec,t,mode,flux)
    u2 = 3*vec/4 + (u1 + k*L(u1,t+k,mode,flux))/4
    u3 = vec/3 + 2*(u2 + k*L(u2,t+0.5*k,mode,flux))/3
    return u3

#Just some source
def source1(x,t, vec, mode):
    x = np.reshape(x, [-1, 1])
    rho, q = decompose_vec(vec);
    return np.zeros(np.shape(rho))

"""Definition of the contribution given by the gravitational potential: refer to the report  """
def source2(x,t, vec, mode):
    rho, q = decompose_vec(vec); N = len(rho); deltax = 2/N
    x = np.reshape(x, [-1, 1]); 
    phix = 4*np.pi*G*np.cumsum(rho, axis=0)*deltax - 2*np.pi*G*np.sum(rho,axis=0)*deltax
    gpot = np.multiply(phix, rho)
    gpot = 0.5*(gpot-np.flipud(gpot))

    return -gpot

#Definition of initial conditions
def q0(x,t, mode):
    x = np.reshape(x, [-1, 1]); N = np.shape(x)[0];
    if mode=='source':
#        return np.vstack((np.zeros([int((N+1)/2), 1]), np.zeros([int((N-1)/2), 1])))
        return np.zeros([N,1])

def rho0(x,t,mode,deltaL):
    x = np.reshape(x, [-1, 1]);N = np.shape(x)[0];
    if mode=='source':
        y = (1-0.5*deltaL)*0.95
        s =  np.piecewise(x, [ x < 1-0.5*deltaL, (x >= 1-0.5*deltaL)*(x < 1+0.5*deltaL), x >= 1+0.5*deltaL], 
                              [sigmoid(350*(x[x<1-0.5*deltaL]-y)), 1, sigmoid(350*(2-y-x[x>=1+0.5*deltaL]))])
        s = np.reshape(s, [-1,1])
        s = 0.5*(s+np.flipud(s))
        return s
        
        
def finitex_source(t,N, vec, mode):
    N = int(N); deltax = 2/N
    x = np.linspace(0.5*deltax, 2-0.5*deltax, N) #that's where the gridpoints are
    rho_source = source1(x,t, vec, mode)
    q_source = source2(x,t, vec, mode)
    return initialize_vec(rho_source, q_source)

def finitex_ic(N,mode,deltaL):
    N = int(N);deltax = 2/N
    x = np.linspace(0.5*deltax, 2-0.5*deltax, N)
    return initialize_vec(rho0(x,0,mode,deltaL), q0(x,0,mode))
    