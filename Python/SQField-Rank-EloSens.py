# SQField-Rank-EloSens: SinquefieldCup Rankability and Elo Sensitivity
#
# Author: Thomas R. Cameron
# Date: 7/16/2019
from srm_module import specR
from copy import deepcopy
import numpy as np

# Elo constant
K = 40.
X = 400.

###########################################################
#                       EloSensitivity                    #
###########################################################
#   Computes sensitivity of Elo Ratings r1 and r2.
###########################################################
def EloSensitivity(r1,r2):
    # indices that would sort array in ascending order
    ind1 = np.argsort(r1)
    ind2 = np.argsort(r2)
    # store number of changes in ranking
    numChanges = 0
    for k in range(len(ind1)):
        if(ind1[k]!=ind2[k]):
            numChanges = numChanges + 1
    # return ratio of number of changes over number of players
    return float(numChanges)/float(len(ind1))

###########################################################
#                       ReadData                          #
###########################################################
#   Reads SinquefieldCup data from a certain year. Returns 
#   adjacency matrix, Elo ratings, and Elo sensitivity for 
#   eeach round, as well as the number of players and total 
#   number of rounds.
###########################################################
def ReadData(year):
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
    # create games, adj, and r lists
    games = [np.zeros((numPlayers,numPlayers)) for k in range(numRounds)]
    adj = [np.zeros((numPlayers,numPlayers)) for k in range(numRounds)]
    elo_rating = [np.zeros(numPlayers) for k in range(numRounds)]
    elo_sens = [0 for k in range(numRounds)]
    # populate games, adj, and r lists
    for k in range(numRounds):
        # copy previous rounds game info
        games[k] = deepcopy(games[k-1])
        # copy previous rounds elo_rating info
        elo_rating[k] = deepcopy(elo_rating[k-1])
        # update game info
        for l in range(numPlayers//2):
            row = lineList.pop(0)
            row = row.split(",")
            row = [eval(row[m]) for m in range(len(row))]
            i = row[0]-1
            j = row[2]-1
            # copy of Elo rating from previous round
            x = deepcopy(elo_rating[k-1])
            if(row[1]>0):
                # player i beat player j
                games[k][i,j] = games[k][i,j] + 1
                # update player i Elo rating
                d = elo_rating[k][i] - elo_rating[k][j]
                u = 1./(1.+10.**(-d/X))
                elo_rating[k][i] = elo_rating[k][i] + K*(1.-u)
                x[i] = x[i] + K*(1.-u)
                # update player j Elo rating
                elo_rating[k][j] = elo_rating[k][j] + K*(u-1.)
                x[j] = x[j] + K*(u-1.)
            elif(row[1]<0):
                # player j beat player i
                games[k][j,i] = games[k][j,i] + 1
                # update player j Elo rating
                d = elo_rating[k][j] - elo_rating[k][i]
                u = 1./(1.+10.**(-d/X))
                elo_rating[k][j] = elo_rating[k][j] + K*(1.-u)
                x[j] = x[j] + K*(1.-u)
                # update player i Elo rating
                elo_rating[k][i] = elo_rating[k][i] + K*(u-1.)
                x[i] = x[i] + K*(u-1.)
            else:
                # draw
                games[k][i,j] = games[k][i,j] + 0.5
                games[k][j,i] = games[k][j,i] + 0.5
                # update player i Elo rating
                d = elo_rating[k][i] - elo_rating[k][j]
                u = 1./(1.+10.**(-d/X))
                elo_rating[k][i] = elo_rating[k][i] + K*(0.5-u)
                x[i] = x[i] + K*(0.5-u)
                # update player j Elo rating
                elo_rating[k][j] = elo_rating[k][j] + K*(u-0.5)
                x[j] = x[j] + K*(u-0.5)
            # record predictability
            if(EloSensitivity(x,elo_rating[k-1]) <= np.finfo(float).eps):
                elo_sens[k] = elo_sens[k] + 1
        # update adjacency matrix
        for i in range(numPlayers):
            for j in range(numPlayers):
                total = games[k][i,j] + games[k][j,i]
                if(total>1):
                    # player i and j played twice
                    adj[k][i,j] = games[k][i,j]/2
                elif(total>0):
                    # player i and j played once
                    adj[k][i,j] = games[k][i,j]
                # otherwise, player i and j have not yet played
    # return
    return adj, elo_rating, elo_sens, numPlayers, numRounds
    
###########################################################
#                       Main                              #
###########################################################
def Main():
    # print statement
    print('Running SQField-Rank-EloSens ...')
    # open files
    f1 = open('../DataFiles/PythonResults/SQField-Rank-EloSens-Rounds.csv','w+')
    f2 = open('../DataFiles/PythonResults/SQField-Rank-EloSens-Summary.csv','w+')
    # round by round analysis and summary
    f1.write('Year, Round, Rank, EloSens \n')
    f2.write('Year, Rank, EloSens \n')
    x = []; y = []
    for year in range(2013,2019):
        adj, elo_rating, elo_sens, numPlayers, numRounds = ReadData(year)
        f1.write('%d,,,\n' % year)
        f2.write('%d' % year)
        for k in range(numRounds):
            f1.write(',%d,%.4f,%.4f\n' % (k+1,specR(adj[k]),elo_sens[k]/(numPlayers/2.)))
        x.append(specR(adj[-1]))
        y.append(sum(elo_sens[1:numRounds])/((numRounds-1)*(numPlayers/2.)))
        f2.write(',%.4f,%.4f\n' % (x[-1],y[-1]))
    # correlation between year summary data
    print('\tspecR and EloSens corr = %.4f' % np.corrcoef(x,y)[0,1])
    # close files
    f1.close()
    f2.close()
            

# call Main       
Main()