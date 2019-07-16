# CFB-Rank-EloPred: College Football Rankability and Elo Predictability

#
# Author: Thomas R. Cameron
# Date: 7/16/2019
from srm_module import specR
from copy import deepcopy
import numpy as np

# Elo constants
K = 32.
H = 15.
X = 1000.

###########################################################
#                       ReadData                          #
###########################################################
#   Reads College Football data from a certain year and
#   and conference. Returns adjacency matrix, Elo ratings,
#   and forward predictability of the Elo ratings for each
#   round, as well as the round structure. 
#   Note that the rounds are determined by the number of 
#   games necessary for each team to partake in another 
#   conference game. This process is continued until
#   each team plays every other team in the conference
#   exactly once.
###########################################################
def ReadData(conf,year):
    # open file
    f = open('../DataFiles/CFB/'+str(conf)+'/'+str(year)+'games.txt')
    # read all lines
    lineList = f.readlines()
    # store team, score, and date info
    numGames = len(lineList)
    teami = []; scorei = []; teamj = []; scorej = []
    for k in range(numGames):
        row = lineList.pop(0)
        row = row.split(",")
        teami.append(eval(row[2]))
        scorei.append(eval(row[4]))
        teamj.append(eval(row[5]))
        scorej.append(eval(row[7]))
    # determine round structure
    numTeams = max(max(teami),max(teamj))
    games_played = [0 for k in range(numTeams)]
    e_row = 0; s_row = 0; k = 0
    count = 1; rnd = []
    while(count<=(numTeams-1)):
        i = teami[k] - 1
        j = teamj[k] - 1
        games_played[i] = games_played[i] + 1
        games_played[j] = games_played[j] + 1
        if(min(games_played)==count):
            e_row = k
            rnd.append([s_row,e_row])
            s_row = e_row + 1
            count = count + 1
        k = k + 1
    # initalize games, adjacency, elo_rating, and forw_pred arrays
    numRounds = len(rnd)
    games = [np.zeros((numTeams,numTeams)) for k in range(numRounds)]
    adj = [np.zeros((numTeams,numTeams)) for k in range(numRounds)]
    elo_rating = [np.zeros(numTeams) for k in range(numRounds)]
    forw_pred = [0 for k in range(numRounds)]
    # store game, adjacency, elo_rating, and forw_pred info round by round
    for k in range(numRounds):
        s_row = rnd[k][0]
        e_row = rnd[k][1]
        # copy over games, win-losses, and Elo rating from (k-1) round
        games[k] = deepcopy(games[k-1])
        adj[k] = deepcopy(adj[k-1])
        elo_rating[k] = deepcopy(elo_rating[k-1])
        # record games and win-losses for kth round
        # also, update Elo rating and forw_pred
        for l in range(s_row,e_row+1):
            # team i and team j
            i = teami[l] - 1
            j = teamj[l] - 1
            # game record
            games[k][i,j] = games[k][i,j] + 1
            games[k][j,i] = games[k][j,i] + 1
            # win-loss record
            if(scorei[l]>scorej[l]):
                # team i > team j
                adj[k][i,j] = adj[k][i,j] + 1
                # forw_pred
                if(elo_rating[k][i]>(elo_rating[k][j]+H)):
                    forw_pred[k] = forw_pred[k] + 1
                # Elo rating
                d = elo_rating[k][i] - elo_rating[k][j]
                u = 1./(1.+10.**(-d/X))
                elo_rating[k][i] = elo_rating[k][i] + K*(1.-u)
                elo_rating[k][j] = elo_rating[k][j] + K*(u-1.)
            else:
                # team j > team i
                adj[k][j,i] = adj[k][j,i] + 1
                # forw_pred
                if(elo_rating[k][i]<(elo_rating[k][j]+H)):
                    forw_pred[k] = forw_pred[k] + 1
                # Elo rating
                d = elo_rating[k][j] - elo_rating[k][j]
                u = 1./(1.+10.**(-d/X))
                elo_rating[k][j] = elo_rating[k][j] + K*(1.-u)
                elo_rating[k][i] = elo_rating[k][i] + K*(u-1.)
    # normalize adjacency array round by round
    for k in range(numRounds):
        for i in range(numTeams):
            for j in range(numTeams):
                if(games[k][i,j]!=0):
                    adj[k][i,j] = adj[k][i,j]/games[k][i,j]
    # close file
    f.close()
    # return
    return adj, elo_rating, forw_pred, rnd
    
###########################################################
#                       BackwardPredictability            #
###########################################################
#   Use final Elo rating to predict entire season. Return
#   the ratio of the number of correct predictions over the
#   total number of games. 
###########################################################
def BackPred(conf,year,elo_rating):
    # open file
    f = open('../DataFiles/CFB/'+str(conf)+'/'+str(year)+'games.txt')
    # read all lines
    lineList = f.readlines()
    # compute back_pred
    numGames = len(lineList)
    back_pred = 0
    for k in range(numGames):
        row = lineList.pop(0)
        row = row.split(",")
        teami = eval(row[2]) - 1
        scorei = eval(row[4])
        teamj = eval(row[5]) - 1
        scorej = eval(row[7])
        if(scorei>scorej):
            if(elo_rating[teami]>(elo_rating[teamj]+H)):
                back_pred = back_pred + 1
        else:
            if(elo_rating[teami]<(elo_rating[teamj]+H)):
                back_pred = back_pred + 1
    # close file
    f.close()
    # return
    return float(back_pred)/float(numGames)
    
#####################################################
#                 Main                              #
#####################################################
def Main():
    # print statement
    print('Running CFB-Rank-EloPred ...')
    # open files
    f1 = open('../DataFiles/PythonResults/CFB-Rank-EloPred-Rounds.csv','w+')
    f2 = open('../DataFiles/PythonResults/CFB-Rank-EloPred-Summary.csv','w+')
    # rankability and predictability
    rank = []
    forwPred = []
    backPred = []
    
    # Atlantic Coast
    print('Atlantic Coast: ')
    x = []; y = []; z = []
    f1.write('Atlantic Coast,Year,Round,Rank,Forw-Pred\n')
    f2.write('Atlantic Coast,Year,Rank,Forw-Pred,Back-Pred\n')
    for year in range(1995,2004):
        f1.write(',%d,,,\n' % year)
        f2.write(',%d' % year)
        adj, elo_rating, forw_pred, rnd = ReadData('Atlantic Coast',year)
        numRounds = len(rnd)
        for k in range(numRounds):
            f1.write(',,%d,%.4f,%.4f\n' % (k+1,specR(adj[k]),forw_pred[k]/float(rnd[k][1]-rnd[k][0]+1)))
        # year summary
        x.append(specR(adj[-1]))
        y.append(sum(forw_pred[1:numRounds])/float(rnd[-1][1]-rnd[1][0]+1))
        z.append(BackPred('Atlantic Coast',year,elo_rating[-1]))
        f2.write(',%.4f,%.4f,%.4f\n' % (x[-1],y[-1],z[-1]))
    # correlation between year summary data
    print('\tspecR and Elo ForwPred corr = %.4f' % np.corrcoef(x,y)[0,1])
    print('\tspecR and Elo BackPred corr = %.4f' % np.corrcoef(x,z)[0,1])
    # update rank, forwPred, and backPred
    rank = rank + x
    forwPred = forwPred + y
    backPred = backPred + z
    
    # Big East
    print('Big East: ')
    x = []; y = []; z = []
    f1.write('Big East,Year,Round,Rank,Forw-Pred\n')
    f2.write('Big East,Year,Rank,Forw-Pred,Back-Pred\n')
    for year in range(1995,2013):
        f1.write(',%d,,,\n' % year)
        f2.write(',%d' % year)
        adj, elo_rating, forw_pred, rnd = ReadData('Big East',year)
        numRounds = len(rnd)
        for k in range(numRounds):
            f1.write(',,%d,%.4f,%.4f\n' % (k+1,specR(adj[k]),forw_pred[k]/float(rnd[k][1]-rnd[k][0]+1)))
        # year summary
        x.append(specR(adj[-1]))
        y.append(sum(forw_pred[1:numRounds])/float(rnd[-1][1]-rnd[1][0]+1))
        z.append(BackPred('Big East',year,elo_rating[-1]))
        f2.write(',%.4f,%.4f,%.4f\n' % (x[-1],y[-1],z[-1]))
    # correlation between year summary data
    print('\tspecR and Elo ForwPred corr = %.4f' % np.corrcoef(x,y)[0,1])
    print('\tspecR and Elo BackPred corr = %.4f' % np.corrcoef(x,z)[0,1])
    # update rank, forwPred, and backPred
    rank = rank + x
    forwPred = forwPred + y
    backPred = backPred + z
    
    # Mountain West
    print('Mountain West: ')
    x = []; y = []; z = []
    f1.write('Mountain West,Year,Round,Rank,Forw-Pred\n')
    f2.write('Mountain West,Year,Rank,Forw-Pred,Back-Pred\n')
    for year in range(1999,2012):
        f1.write(',%d,,,\n' % year)
        f2.write(',%d' % year)
        adj, elo_rating, forw_pred, rnd = ReadData('Mountain West',year)
        numRounds = len(rnd)
        for k in range(numRounds):
            f1.write(',,%d,%.4f,%.4f\n' % (k+1,specR(adj[k]),forw_pred[k]/float(rnd[k][1]-rnd[k][0]+1)))
        # year summary
        x.append(specR(adj[-1]))
        y.append(sum(forw_pred[1:numRounds])/float(rnd[-1][1]-rnd[1][0]+1))
        z.append(BackPred('Mountain West',year,elo_rating[-1]))
        f2.write(',%.4f,%.4f,%.4f\n' % (x[-1],y[-1],z[-1]))
    # correlation between year summary data
    print('\tspecR and Elo ForwPred corr = %.4f' % np.corrcoef(x,y)[0,1])
    print('\tspecR and Elo BackPred corr = %.4f' % np.corrcoef(x,z)[0,1])
    # update rank, forwPred, and backPred
    rank = rank + x
    forwPred = forwPred + y
    backPred = backPred + z
    
    # close files
    f1.close()
    f2.close()
    
    # complete correlation summary
    print('Complete Summary: ')
    print('\tspecR and Elo ForwPred corr = %.4f' % np.corrcoef(rank,forwPred)[0,1])
    print('\tspecR and Elo BackPred corr = %.4f' % np.corrcoef(rank,backPred)[0,1])
    

# call Main
Main()