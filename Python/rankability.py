# Rankability Module
#
# This module contains the functions necessary for measuring the rankability of a dataset.
# In particular, the dataset should be captured as a directed graph with weights between 
# zero and one.
# Given the corresponding adjacency matrix, a rankability measure is returned based on the 
# spectral-degree characterization of the graph Laplacian of a complete dominance graph.
#
# Author: Thomas R. Cameron
# Date: 11/1/2019
import numpy as np
import itertools
from math import factorial
from scipy.stats import spearmanr
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
#   Computes Spectral-Degree Rankability Measure.
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
###############################################
###             edgeR                       ###
###############################################
#   Computes edge Rankability Measure using brute force approach.
###############################################
def edgeR(a):
    # size
    n = len(a)
    # complete dominance
    domMat = np.triu(1.0*np.ones((n,n)),1)
    # fitness list
    fitness = []
    # brute force (consider all permutations)
    for i in itertools.permutations(range(n)):
        b = a[i,:]
        b = b[:,i]
        # number of edge changes (k) for given permutation
        fitness.append(np.sum(np.abs(domMat - b)))
    # minimum number of edge chagnes
    k = np.amin(fitness)
    # number of permutations that gave this k
    p = np.sum(np.abs(fitness-k)<np.finfo(float).eps)
    # rankability measure
    return 1.0 - 2.0*k*p/(n*(n-1)*factorial(n))
###############################################
###             main                        ###
###############################################
#   main method tests SIMOD 1 examples
###############################################
def main():
    adj = [np.array([[0.,1,1,1,1,1],[0,0.,1,1,1,1],[0,0,0.,1,1,1],[0,0,0,0.,1,1],[0,0,0,0,0.,1],[0,0,0,0,0,0.]]),
            np.array([[0.,1,1,1,1,1],[0,0.,0,1,1,1],[1,0,0.,1,1,1],[0,0,0,0.,1,1],[0,0,0,0,0.,1],[0,0,0,0,0,0.]]),
            np.array([[0.,1,1,0,0,1],[0,0.,0,1,1,0],[0,0,0.,0,0,0],[1,1,0,0.,0,1],[1,1,0,0,0.,0],[1,1,1,0,1,0.]]),
            np.array([[0.,1,1,1,0,0],[0,0.,1,0,0,0],[0,0,0.,0,0,0],[0,0,0,0.,1,1],[0,0,0,0,0.,1],[0,0,0,0,0,0.]]),
            np.array([[0.,1,1,0,0,1],[0,0.,0,1,1,0],[0,0,0.,0,0,0],[1,0,0,0.,0,1],[1,1,0,0,0.,0],[0,1,1,0,1,0.]]),
            np.array([[0.,1,0,0,0,0],[0,0.,1,0,0,0],[0,0,0.,1,0,0],[0,0,0,0.,1,0],[0,0,0,0,0.,1],[1,0,0,0,0,0.]]),
            np.array([[0.,1,1,1,1,1],[1,0.,1,1,1,1],[1,1,0.,1,1,1],[1,1,1,0.,1,1],[1,1,1,1,0.,1],[1,1,1,1,1,0.]]),
            np.zeros((6,6))
            ]
    er = []
    sr = []
    for k in range(len(adj)):
        er.append(edgeR(adj[k]))
        sr.append(specR(adj[k]))
    corr,pval = spearmanr(er,sr)
    print('Anderson et al. Digraph Examples: ')
    print('edgeR = [%.4f' % er[0], end='')
    for k in range(len(er)):
        print(', %.4f' % er[k], end='')
    print(']')
    print('specR = [%.4f' % sr[0], end='')
    for k in range(len(sr)):
        print(', %.4f' % sr[k], end='')
    print(']')
    print('edgeR and specR corr = %.4f' % corr)
    print('edgeR and specR pval = %.4f' % pval)
if __name__ == '__main__':
    main()