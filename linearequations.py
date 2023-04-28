#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 16:29:27 2023

@author: vedantvairagi
"""
import numpy as np

def hval(r,α,b,T,h,E1):
    
    return -r*h - α*b*T - α*E1

def Tval(b,R,γ,T,h,en,E1,E2):
    
    return R*T + γ*h - en*(h + b*T)**3 + γ*E1 + E2

def Selfexcite(µo,µann,τ,t):
    
    return µo*(1+µann*np.cos((2*np.pi*t/τ)-(5*np.pi/6)))

def noisywind(fann,fran,τ,t,W,τcorr,delt):
    
    return fann*np.cos(2*np.pi*t/τ) + fran*W*τcorr/delt
    