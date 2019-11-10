# SQField-Rank-EloCorr: SinquefieldCup Rankability and Elo Correlation
#
# Author: Thomas R. Cameron
# Date: 11/1/2019
from rankability import specR
from copy import deepcopy
import numpy as np
from scipy.stats import pearsonr, spearmanr, kendalltau

# Elo constant
K = 40.
X = 400.

###########################################################
#                       sqfieldData                       #
###########################################################
#   Reads SinquefieldCup data from a certain year. Returns 
#   Elo Correlation and rankability for that year. The opt 
#   variable determines how the Elo Correlation is measured 
#   with either kendalltau (KT), spearmanr (SR), or pearsonr (PR).
###########################################################
def sqfieldData(year,opt):
    # open file
    f = open('../DataFiles/SinquefieldCup/SinquefieldCup'+str(year)+'.csv')
    # read all lines
    lineList = f.readlines()
    # grab first line and split
    row = lineList.pop(0)
    row = row.split(",")
    # store numPlayers and numRounds
    numPlayers = eval(row[0])
    numRounds = eval(row[1])
    # create matches, adj, and r lists
    matches = [np.zeros((numPlayers,numPlayers)) for k in range(numRounds)]
    adj = [np.zeros((numPlayers,numPlayers)) for k in range(numRounds)]
    elo_rating = [np.zeros(numPlayers) for k in range(numRounds)]
    elo_corr = []
    rankability = []
    # populate matches, adj, elo_rating, elo_corr, and rankability
    for k in range(numRounds):
        # deepcopy previous rounds matches and elo_rating
        matches[k] = deepcopy(matches[k-1])
        elo_rating[k] = deepcopy(elo_rating[k-1])
        # play out number of matches in kth round
        for l in range(numPlayers//2):
            row = lineList.pop(0)
            row = row.split(",")
            row = [eval(row[0]),eval(row[1]),eval(row[2])]
            i = row[0]-1
            j = row[2]-1
            if(row[1]>0):
                # player i beat player j
                matches[k][i,j] = matches[k][i,j] + 1
                # update player i Elo rating
                d = elo_rating[k][i] - elo_rating[k][j]
                u = 1./(1.+10.**(-d/X))
                elo_rating[k][i] = elo_rating[k][i] + K*(1.-u)
                # update player j Elo rating
                elo_rating[k][j] = elo_rating[k][j] + K*(u-1.)
            elif(row[1]<0):
                # player j beat player i
                matches[k][j,i] = matches[k][j,i] + 1
                # update player j Elo rating
                d = elo_rating[k][j] - elo_rating[k][i]
                u = 1./(1.+10.**(-d/X))
                elo_rating[k][j] = elo_rating[k][j] + K*(1.-u)
                # update player i Elo rating
                elo_rating[k][i] = elo_rating[k][i] + K*(u-1.)
            else:
                # draw
                matches[k][i,j] = matches[k][i,j] + 0.5
                matches[k][j,i] = matches[k][j,i] + 0.5
                # update player i Elo rating
                d = elo_rating[k][i] - elo_rating[k][j]
                u = 1./(1.+10.**(-d/X))
                elo_rating[k][i] = elo_rating[k][i] + K*(0.5-u)
                # update player j Elo rating
                elo_rating[k][j] = elo_rating[k][j] + K*(u-0.5)
        # update adjacency matrix
        for i in range(numPlayers):
            for j in range(i+1,numPlayers):
                total = matches[k][i,j] + matches[k][j,i]
                if(total!=0):
                    adj[k][i,j] = matches[k][i,j]/total
                    adj[k][j,i] = matches[k][j,i]/total
        # Rankability
        rankability.append(specR(adj[k]))
        # Elo Correlation
        if(k>=1 and opt=="SR"):
            corr,pval = spearmanr(elo_rating[k],elo_rating[k-1])
            elo_corr.append(corr)
        elif(k>=1 and opt=="KT"):
            corr,pval = kendalltau(elo_rating[k],elo_rating[k-1])
            elo_corr.append(corr)
        elif(k>=1 and opt=="PR"):
            corr,pval = pearsonr(elo_rating[k],elo_rating[k-1])
            elo_corr.append(corr)
    # return variables
    return elo_corr, rankability
###########################################################
#                       main                              #
###########################################################
def main():
    # open files
    f1 = open('../DataFiles/PythonResults/SQField-Rank-EloCorr-Rounds.csv','w+')
    f2 = open('../DataFiles/PythonResults/SQField-Rank-EloCorr-Summary.csv','w+')
    # round by round analysis and summary
    f1.write('Year, Round, Rankability, EloCorr \n')
    f2.write('Year, Rankability, EloCorr \n')
    x = []; y = []
    for year in range(2013,2020):
        elo_corr,rankability = sqfieldData(year,"SR")
        f1.write('%d,,,\n' % year)
        f2.write('%d' % year)
        for k in range(len(rankability)):
            if(k>=1):
                f1.write(',%d,%.4f,%.4f\n' % (k+1,rankability[k],elo_corr[k-1]))
            else:
                f1.write(',%d,%.4f,%.4f\n' % (k+1,rankability[k],0))
        x.append(rankability[-1])
        y.append(np.average(elo_corr,weights=[k for k in range(len(elo_corr))]))
        f2.write(',%.4f,%.4f\n' % (x[-1],y[-1]))
    # correlation between year summary data
    corr,pval = spearmanr(x,y)
    print('\tspecR and EloCorr corr = %.4f' % corr)
    print('\tspecR and EloCorr pval = %.4f' % pval)
    # close files
    f1.close()
    f2.close()

if __name__ == '__main__':
    main()