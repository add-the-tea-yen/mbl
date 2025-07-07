#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  7 11:59:06 2025

@author: adityan
"""

L = 10
print(2**L)
nev = int(min((2**L/10),1000))
print(nev)
ncv = int(2 * nev)
print(ncv)
k = int(0.95 * (2**(L+1))/ncv)
print(k)
