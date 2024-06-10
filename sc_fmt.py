# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 19:48:17 2024

@author: HP
"""
import numpy as np

#Convertir un número a formato científico

#Sea un número s=x*10^-n
#Tomando logaritmo en base 10
#log10(s)=log10(x)-n

#Dado que 1<x<10, log10(x) estara entre 0 y 1

def exponent(s):
    y=round(np.log10(s)-0.5)
    return y

def int_part(s):
    exp=exponent(s)
    y=np.log10(s)-exp
    y=10**y
    return y

def sc_fmt(s,round=2):
    x=int_part(s)
    n=exponent(s)
    p=int(round)
    if n!=0:
        if n > 0:
            str_fmt='{:.{}f} e+{}'.format(x,p,n)
        else:
            str_fmt='{:.{}f} e{}'.format(x,p,n)
    else:
        str_fmt='{:.{}f}'.format(x,p)
    return str_fmt


