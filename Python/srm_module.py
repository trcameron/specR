# SRM: Spectral Rankability Measure
#
# This module contains the functions necessary for measuring the rankability of a dataset.
# In particular, the dataset should be captured as a directed graph with weights between 
# zero and one.
# Given the corresponding adjacency matrix, a rankability measure is returned based on the 
# spectral-degree characterization of the graph Laplacian of a complete dominance graph.
#
# Author: Thomas R. Cameron
# Date: 7/16/2019
import numpy as np
    
###############################################
###             Hausdorff                   ###
###############################################
#   Hausdorff distance between sets e and s.
###############################################
def Hausdorff(e,s):
    # spectral variation
    def _sv(e,s):
        return max([min([abs(e[i]-s[j]) for j in range(len(s))]) for i in range(len(e))])
    # Hausdorff distance
    return max(_sv(e,s),_sv(s,e))

###############################################
###             specR                       ###
###############################################
#   Computes Spectral Rankability Measure.
###############################################
def specR(a):
    # given graph Laplacian
    n = len(a)
    x = np.array([np.sum(a[i,:]) for i in range(n)])
    d = np.diag(x)
    l = d - a;
    # perfect dominance graph spectrum and out-degree
    s = np.array([n-k for k in range(1,n+1)])
    # eigenvalues of given graph Laplacian
    e = np.linalg.eigvals(l)
    # rankability measure
    return 1. - ((Hausdorff(e,s)+Hausdorff(x,s))/(2*(n-1)))