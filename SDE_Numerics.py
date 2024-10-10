#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 22 14:35:34 2022

@author: maximebouchereau
"""

# Code for numerical simulations of stochastic differential equations via Forward Euler Method - Example of Black & Scholes model

# Modules importation

import numpy as np
import statistics
import math as mt
import matplotlib.pyplot as plt
import sys

# Selection of parameters

T = 5           # Duration of the simulation
K = 100         # Number of simulations
h = 0.05        # Time step
mu = 0.2        # Interest rate
sigma = 0.2     # Volatility
y0 = 1          # Initial datum

# Simulation of SDE's

class SDE_Numerics:

    def BM(Tf=T):
        """
        Computes a trajectory given by a brownian motion, by using independants increments of step Tf/10000

        Parameters
        ----------
        Tf : TYPE, optional: Float
            DESCRIPTION. The default is T. Duration of simulation of the bvrownian motion over the interval [0,Tf]

        Returns an array of shape (10001,) containing a trajectory of a brownian motion at time 0, Tf/10000, ...
        -------
        None.

        """
        
        
        B = np.zeros(10001,)
        for j in range(10000):
            B[j+1] = B[j] + np.random.normal(loc=0,scale=np.sqrt(Tf/10000))
        return B
    
    def Exact(Tf=T,ht=h,x=y0):
        
        """
        Computes an exact solution of the stochastic differential equation
        dX_t = mu*X_t*dt + sigma*X_t*dB_t via Forward Euler method.

        Parameters
        ----------
        Tf : TYPE, optional: Float
            DESCRIPTION. The default is T. Duration of the simulation
        ht : TYPE, optional: Float
            DESCRIPTION. The default is h. Time step
        x : TYPE, optional: Float
            DESCRIPTION. The default is y0. Initial condition

        Returns an array containing the approximated trajectory of the solution of the SDE
        -------
        None.

        """
        
        B = SDE_Numerics.BM(Tf)
        TT = np.arange(0,Tf+ht,ht)
        Z = np.zeros(len(TT),)
        
        for j in range(len(TT)):
            Z[j] = x*np.exp((mu-sigma**2/2)*(j*ht) + sigma*B[int(10000*(j)*ht/Tf)])
        
        return Z        

    def Euler(Tf=T,ht=h,x=y0):
        """
        Computes an approximation of the solution of the stochastic differential equation
        dX_t = mu*X_t*dt + sigma*X_t*dB_t via Forward Euler scheme.

        Parameters
        ----------
        Tf : TYPE, optional: Float
            DESCRIPTION. The default is T. Duration of the simulation
        ht : TYPE, optional: Float
            DESCRIPTION. The default is h. Time step
        x : TYPE, optional: Float
            DESCRIPTION. The default is y0. Initial condition

        Returns an array containing the trajectory of the solution of the SDE
        -------
        None.

        """
        
        B = SDE_Numerics.BM(Tf)
        TT = np.arange(0,Tf+ht,ht)
        Z = np.zeros(len(TT),)
        Z[0] = x
        
        for j in range(len(TT)-1):
            Z[j+1] = Z[j] + ht*mu*Z[j] + sigma*Z[j]*(B[int(10000*(j+1)*ht/Tf)] - B[int(10000*(j)*ht/Tf)])
        
        return Z
    
    def Milstein(Tf=T,ht=h,x=y0):
        """
        Computes an approximation of the solution of the stochastic differential equation
        dX_t = mu*X_t*dt + sigma*X_t*dB_t via Milstein scheme.

        Parameters
        ----------
        Tf : TYPE, optional: Float
            DESCRIPTION. The default is T. Duration of the simulation
        ht : TYPE, optional: Float
            DESCRIPTION. The default is h. Time step
        x : TYPE, optional: Float
            DESCRIPTION. The default is y0. Initial condition

        Returns an array containing the trajectory of the solution of the SDE
        -------
        None.

        """
        
        B = SDE_Numerics.BM(Tf)
        TT = np.arange(0,Tf+ht,ht)
        Z = np.zeros(len(TT),)
        Z[0] = x
        
        for j in range(len(TT)-1):
            Z[j+1] = Z[j] + ht*mu*Z[j] + sigma*Z[j]*(B[int(10000*(j+1)*ht/Tf)] - B[int(10000*(j)*ht/Tf)]) + (1/2)*sigma**2*Z[j]*(B[int(10000*(j+1)*ht/Tf)] - B[int(10000*(j)*ht/Tf)]) + (1/2)*sigma**2*Z[j]*((B[int(10000*(j+1)*ht/Tf)] - B[int(10000*(j)*ht/Tf)])**2-ht)
        
        return Z
            
    
    def Run(Ns=K,Tf=T,ht=h,x=y0,scheme="Euler",save=False,name_run="Run_SDE"):
        """
        
        Plot several simulations of the stochastic differential equation dX_t = mu*X_t*dt + sigma*X_t*dB_t via
        selected scheme.

        Parameters
        ----------
        Ns : TYPE, optional: Int. Number of simulations
            DESCRIPTION. The default is K.
        Tf : TYPE, optional: Float
            DESCRIPTION. The default is T. Duration of the simulation
        ht : TYPE, optional: Float
            DESCRIPTION. The default is h. Time step
        x : TYPE, optional: Float
            DESCRIPTION. The default is y0. Initial condition
        scheme : TYPE, Character string
                 DESCRIPTION. The default is "Euler". The numerical sheme which is selected, choice between "Euler" and "Milstein"
        save : TYPE, optional: Boolean
               DESCRIPTION. The default is False. Saves the graph or not
        name_run : TYPE, optional: Character string
               DESCRIPTION. The default is "Convergence_SDE". Name of the graph saved (useful if save = True)
        
        Plots several trajectories of the solution of the SDE
        -------
        None.

        """
        
        List_sol = []
        TT = np.arange(0,Tf+ht,ht)
        
        print("Computation of approximated solutions...")
        for k in range(K):
            kk = k + 1
            sys.stdout.write("\r%d   "% kk+"/ "+str(K))
            if scheme == "Euler":
                List_sol.append(SDE_Numerics.Euler(Tf,ht,x))
            if scheme == "Milstein":
                List_sol.append(SDE_Numerics.Milstein(Tf,ht,x))
        
        plt.figure(0)
        plt.title("Plot of approximated solutions of SDE $dX_t = \mu X_tdt + \sigma X_tdB_t$")
        plt.xlabel("$t$")
        plt.ylabel("$X_t$")
        plt.grid()
        for k in range(K):
            plt.plot(TT,List_sol[k])
            
        if save == True:
            plt.savefig(name_run+".pdf")
        else:     
            plt .show()
        
        pass

class Conv(SDE_Numerics):
    def Err(Ns=K,Tf=T,ht=h,x=y0,scheme = "Euler"):
        """
        Computes the L2(P)-error between approximated and exact solution of the stochastic differential equation
        dX_t = mu*X_t*dt + sigma*X_t*dB_t. The numerical methods employed is Forward Euler and Milstein schemes.

        Parameters
        ----------
        Ns : TYPE, optional: Int. Number of simulations
            DESCRIPTION. The default is K.
        Tf : TYPE, optional: Float
            DESCRIPTION. The default is T. Duration of the simulation
        ht : TYPE, optional: Float
            DESCRIPTION. The default is h. Time step
        x : TYPE, optional: Float
            DESCRIPTION. The default is y0. Initial condition
        scheme : TYPE, Character string
                 DESCRIPTION. The default is "Euler". The numerical sheme which is selected, choice between "Euler" and "Milstein"

        Returns the square root of mean squarred error (MSE) between the K exact solutions and K correponding
        approximated solutions
        -------
        None.

        """
        
        
        TT = np.arange(0,Tf+ht,ht)
        err = np.zeros(len(TT),)
        
        for k in range(K):
            B = SDE_Numerics.BM(Tf)
            
            Z_ex = np.zeros(len(TT),)
            Z_app = np.zeros(len(TT),)
            Z_app[0] = x
            Z_ex[0] = x
            
            for j in range(len(TT)-1):
                Z_ex[j+1] = x*np.exp((mu-sigma**2/2)*((j+1)*ht) + sigma*B[int(10000*(j+1)*ht/T)])
                if scheme == "Euler":
                    Z_app[j+1] = Z_app[j] + ht*mu*Z_app[j] + sigma*Z_app[j]*(B[int(10000*(j+1)*ht/Tf)] - B[int(10000*(j)*ht/Tf)])
                if scheme == "Milstein":
                    Z_app[j+1] = Z_app[j] + ht*mu*Z_app[j] + sigma*Z_app[j]*(B[int(10000*(j+1)*ht/Tf)] - B[int(10000*(j)*ht/Tf)]) + (1/2)*sigma**2*Z_app[j]*((B[int(10000*(j+1)*ht/Tf)] - B[int(10000*(j)*ht/Tf)])**2-ht)
            
            err = err + (Z_app-Z_ex)**2
        err = err/K
        err = np.linalg.norm(err,np.inf)
        return err**0.5
    
    def Curve_Error(Ns=K,Tf=T,h_t=h,x=y0,save=False,name_curve="Convergence_SDE"):
        """
        Plot the curve of convergence with the Forward Euler method: errors are plot on a logarithmic scale for
        time steps ht/2**j for j=0,...,8

        Parameters
        ----------
        Ns : TYPE, optional: Int. Number of simulations
            DESCRIPTION. The default is K.
        Tf : TYPE, optional: Float
            DESCRIPTION. The default is T. Duration of the simulation
        h_t : TYPE, optional: Float
            DESCRIPTION. The default is h. Time step
        x : TYPE, optional: Float
            DESCRIPTION. The default is y0. Initial condition
        save : TYPE, optional: Boolean
               DESCRIPTION. The default is False. Saves the graph or not
        name_curve : TYPE, optional: Character string
               DESCRIPTION. The default is "Convergence_SDE". Name of the graph saved (useful if save = True)

        Returns
        -------
        None.

        """
        
        H = np.array([h_t/2**j for j in range(9)])
        Err_Euler = np.zeros(len(H),)
        Err_Milstein = np.zeros(len(H),)
        
        print("Error computation...")
        for j in range(len(H)):
            sys.stdout.write("\r%d   "% j+"/ "+str(8))
            Err_Euler[j] = Conv.Err(Ns=K,Tf=T,ht=h_t/2**j,x=y0,scheme="Euler")
            Err_Milstein[j] = Conv.Err(Ns=K,Tf=T,ht=h_t/2**j,x=y0,scheme="Milstein")
        
        plt.figure(0)
        plt.xscale('log')
        plt.yscale('log')
        plt.scatter(H,Err_Euler,marker="s",color="red",label="Euler")
        plt.scatter(H,Err_Milstein,marker="s",color="green",label="Milstein")
        plt.plot(H,(Err_Euler[0]/H[0]**0.5)*H**0.5,"--k",label="$h \mapsto h^{-1/2}$")
        plt.plot(H,(Err_Milstein[0]/H[0])*H,"--",color="silver",label="$h \mapsto h^{-1}$")
        plt.xlabel("h")
        plt.ylabel("$L^2(IP)$-error")
        plt.title("Convergence for the SDE $dX_t = \mu X_tdt + \sigma X_tdB_t$")
        plt.grid()
        plt.legend()
        
        if save == True:
            plt.savefig(name_curve+".pdf")
        else:
            plt.show()
        pass
            









