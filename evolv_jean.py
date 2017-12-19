#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 20:56:26 2017

@author: lorenzolmp

In this script, the time evolution is defined. This code can be run straight-away.
"""

import numpy as np
import matplotlib.pyplot as plt
#import matplotlib.animation as animation
from pylab import *
import scheme_1 as scheme
from scipy.interpolate import interp1d
import timeit

CFL = 0.5
xmax = 2
#tmax = 0.02


def main_evolver(Nx,flux,plot_mode,mode,case,deltaL,N_iter, N_list):
    deltax = xmax/Nx

    u0 = scheme.finitex_ic(Nx,mode,deltaL)
    RHO0, Q0 = scheme.decompose_vec(u0)
    effective_size = sum(RHO0)*deltax

    #################
        #PLOT DEF
    if plot_mode == 'ALL':
        x = np.linspace(0.5*deltax, 2-0.5*deltax, Nx)
        fig, (ax1, ax2) = plt.subplots(1,2, sharey = False)    
        ion()
        ax1.set_title('q(x,t)')
        ax2.set_title('rho(x,t)')
        ax2.set_ylim([0, int(max(RHO0)) + 1])
        ax1.set_ylim([-1.1, 1.1])
        line, = ax1.plot(x, Q0, color='k');
        line2, = ax2.plot(x, RHO0, color='k');
    #################  
    T = 0
    i = 0
    N_list = np.array(N_list)
    heights = []
    profiles = []
    while i <= N_iter:
        k = CFL*deltax/max(scheme.DAMP(u0))
        u1 = scheme.SSPRK3(u0,k,T,mode,flux)
        if np.sum(i == N_list)==1:
            RHO1, Q1 = scheme.decompose_vec(u1)
            heights.append(max(RHO1))
            profiles.append(RHO1)
        if plot_mode == 'ALL':
            RHO1, Q1 = scheme.decompose_vec(u1)
            line.set_ydata(Q1);
            line2.set_ydata(RHO1);
            draw();pause(0.0002)
        u0 = u1
        T = T + k
        i +=1
    if plot_mode == 'LAST':
        x = np.linspace(deltax/2,2-deltax/2,Nx)
        fig, (ax1, ax2) = plt.subplots(1,2, sharey=False)
        ax1.set_title('q(x,t)')
        ax2.set_title('rho(x,t)')
        ax1.set_ylim([-1.1, 1.1])
        RHO1, Q1 = scheme.decompose_vec(u1)
        ax2.set_ylim([0, 2])
        line, = ax1.plot(x, Q1, color='k');
        line2, = ax2.plot(x, RHO1, color='k');
        plt.show()
    return Q1, RHO1, effective_size, heights,profiles


def instabil(N_iter_max, N_list, deltasize):
    heights = np.zeros([len(N_list),len(deltasize)]) #size in the columns
    eff_size = np.zeros([1, len(deltasize)])
    for jj in range(len(deltasize)):
        _, RHO1, effective_size,heights[:,jj] = main_evolver(300,'LF',plot_mode='LAST',mode='source', 
                                                    case='minmod',deltaL=deltasize[jj], 
                                                    N_iter=N_iter_max, N_list=N_list)
        eff_size[0,jj] = effective_size;
    np.savetxt('max_height.txt',heights)
    np.savetxt('effect_size.txt',eff_size)
    plt.close('all')
    plt.figure()
    for ss in range(len(N_list)):
        plt.plot(eff_size.T, heights[ss,:], '^', label='N iter = %.2e'%N_list[ss])
    
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim([0.09,1])
    plt.axhline(y=1,xmin=0.09,xmax=1,linestyle='-',color='r',linewidth=1)
    plt.grid(which='both',linestyle='--', linewidth=0.5)
    plt.title('Jeans Instability')
    plt.xlabel(r'effective size $\delta L$')
    plt.ylabel(r'$\rho^{\infty}$')
    plt.legend()
    plt.savefig('jeans_inst_log.pdf',dpi=200)
    plt.show()
               
""" plot mode can be 'LAST' for only final snapshot or 'ALL' for all the evolution"""
""" deltaL can be changed in the range 0.1- up to 1/1.5 """
""" the number of points in the mesh can be chosen at will: 300 are sufficient for a good resolution"""
start_time = timeit.default_timer()
Q1, RHO1, effective_size,heights,profiles = main_evolver(300,'LF',plot_mode='ALL',mode='source', case='minmod',deltaL=0.1, N_iter=10000,N_list = [2000])
#instabil(70000, [5000, 7500, 15000, 20000, 30000, 45000, 70000], [0.02, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
elapsed = timeit.default_timer() - start_time; print('Elapsed time: ', elapsed)