# CFB-Rank-EloCorr: CFB Rankability and Elo Correlation
#
# Author: Thomas R. Cameron
# Date: 11/1/2019
from rankability import SDR
from copy import deepcopy
import numpy as np
from scipy.stats import pearsonr, spearmanr, kendalltau

# Elo constants
K = 32.
H = 2.
X = 1000.

###########################################################
#                       cfbData                           #
###########################################################
#   Reads CFB data from a certain year and conference. 
#   Returns Elo Correlation and rankability for that year 
#   and conference.
#   The opt variable determines how the Elo Correlation
#   is measured with either kendalltau (KT), spearmanr (SR),
#   or pearsonr (PR).
###########################################################
def cfbData(conf,year,opt):
    # open file
    f = open('../DataFiles/CFB/'+str(conf)+'/'+str(year)+'games.txt')
    # read all lines
    lineList = f.readlines()
    # date, team, and score info
    numGames = len(lineList)
    date=[]; teami = []; scorei = []; teamj = []; scorej = []
    for k in range(numGames):
        row = lineList.pop(0)
        row = row.split(",")
        date.append(eval(row[0]))
        teami.append(eval(row[2]))
        scorei.append(eval(row[4]))
        teamj.append(eval(row[5]))
        scorej.append(eval(row[7]))
    # populate matches, adj, elo_rating, elo_corr, and rankability
    numTeams = max(max(teami),max(teamj))
    matches = [np.zeros((numTeams,numTeams))]
    elo_rating = [np.zeros(numTeams)]
    adj = [np.zeros((numTeams,numTeams))]
    rankability = []
    elo_corr = []
    for k in range(numGames):
        i = teami[k] - 1
        j = teamj[k] - 1
        if(scorei[k]>scorej[k]):
            # team i beat team j
            matches[-1][i,j] = matches[-1][i,j] + 1
            # update team i Elo rating
            d = elo_rating[-1][i] - elo_rating[-1][j]
            u = 1./(1.+10.**(-d/X))
            elo_rating[-1][i] = elo_rating[-1][i] + K*(1.-u)
            # update player j Elo rating
            elo_rating[-1][j] = elo_rating[-1][j] + K*(u-1.)
        elif(scorei[k]<scorej[k]):
            # team j beat team i
            matches[-1][j,i] = matches[-1][j,i] + 1
            # update team j Elo rating
            d = elo_rating[-1][j] - elo_rating[-1][i]
            u = 1./(1.+10.**(-d/X))
            elo_rating[-1][j] = elo_rating[-1][j] + K*(1.-u)
            # update team i Elo rating
            elo_rating[-1][i] = elo_rating[-1][i] + K*(u-1.)
        else:
            # team i and team j tied
            matches[-1][i,j] = matches[-1][i,j] + 0.5
            matches[-1][j,i] = matches[-1][j,i] + 0.5
            # update team i Elo rating
            d = elo_rating[-1][i] - elo_rating[-1][j]
            u = 1./(1.+10.**(-d/X))
            elo_rating[-1][i] = elo_rating[-1][i] + K*(0.5-u)
            # update team j Elo rating
            elo_rating[-1][j] = elo_rating[-1][j] + K*(u-0.5)
        # next round
        if(k<(numGames-1) and (date[k+1]-date[k]+1)>4):
            # update adjacency matrix
            for i in range(numTeams):
                for j in range(i+1,numTeams):
                    total = matches[-1][i,j] + matches[-1][j,i]
                    if(total>0):
                        adj[-1][i,j] = matches[-1][i,j]/total
                        adj[-1][j,i] = matches[-1][j,i]/total
            # Rankability
            rankability.append(SDR(adj[-1]))
            # Elo Correlation
            if(len(elo_rating)>1 and opt=="SR"):
                corr,pval = spearmanr(elo_rating[-1],elo_rating[-2])
                elo_corr.append(corr)
            elif(len(elo_rating)>1 and opt=="KT"):
                corr,pval = kendalltau(elo_rating[-1],elo_rating[-2])
                elo_corr.append(corr)
            elif(len(elo_rating)>1 and opt=="PR"):
                corr,pval = pearsonr(elo_rating[-1],elo_rating[-2])
                elo_corr.append(corr)
            # add next rounds matches, elo_rating, and adjacency storage
            matches.append(deepcopy(matches[-1]))
            elo_rating.append(deepcopy(elo_rating[-1]))
            adj.append(np.zeros((numTeams,numTeams)))
        # last round
        if(k==(numGames-1)):
            # update adjacency matrix
            for i in range(numTeams):
                for j in range(i+1,numTeams):
                    total = matches[-1][i,j] + matches[-1][j,i]
                    if(total>0):
                        adj[-1][i,j] = matches[-1][i,j]/total
                        adj[-1][j,i] = matches[-1][j,i]/total
            # Rankability
            rankability.append(SDR(adj[-1]))
            # Elo Correlation
            if(len(elo_rating)>1 and opt=="SR"):
                corr,pval = spearmanr(elo_rating[-1],elo_rating[-2])
                elo_corr.append(corr)
            elif(len(elo_rating)>1 and opt=="KT"):
                corr,pval = kendalltau(elo_rating[-1],elo_rating[-2])
                elo_corr.append(corr)
            elif(len(elo_rating)>1 and opt=="PR"):
                corr,pval = pearsonr(elo_rating[-1],elo_rating[-2])
                elo_corr.append(corr)
    # return
    return elo_corr, rankability, elo_rating[-1]
###########################################################
#                    Elo Predictability                   #
###########################################################
#   Use final Elo rating to predict entire season. Return
#   the ratio of the number of correct predictions over the
#   total number of games. 
###########################################################
def eloPred(conf,year,elo_rating):
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
        homei = eval(row[3])
        scorei = eval(row[4])
        teamj = eval(row[5]) - 1
        homej = eval(row[6])
        scorej = eval(row[7])
        if(scorei>scorej):
            # team i won at home
            if(homei==1 and elo_rating[teami]>(elo_rating[teamj]-H)):
                back_pred = back_pred + 1
            # team i won on the road
            elif(homei==-1 and elo_rating[teami]>(elo_rating[teamj]+H)):
                back_pred = back_pred + 1
        else:
            # team j won at home
            if(homej==1 and elo_rating[teamj]>(elo_rating[teami]-H)):
                back_pred = back_pred + 1
            # team j won on the road
            elif(homej==-1 and elo_rating[teamj]>(elo_rating[teami]+H)):
                back_pred = back_pred + 1
    # close file
    f.close()
    # return
    return float(back_pred)/float(numGames)
###########################################################
#                       main                              #
###########################################################
def main():
    # open files
    f1 = open('../DataFiles/PythonResults/CFB-Rank-EloCorr-Rounds.csv','w+')
    f2 = open('../DataFiles/PythonResults/CFB-Rank-EloCorr-Summary.csv','w+')
    # Atlantic Coast round by round analysis and summary
    f1.write('Atlantic Coast, Year, Round, Rankability, EloCorr\n')
    f2.write('Atlantic Coast, Year, Rankability, EloCorr, EloPred\n')
    x = []; y = []; z = []
    for year in range(1995,2004):
        elo_corr,rankability,elo_rating = cfbData('Atlantic Coast',year,"SR")
        f1.write(',%d,,,\n' % year)
        f2.write(',%d,,\n' % year)
        for k in range(len(rankability)):
            if(k>=1):
                f1.write(',,%d,%.4f,%.4f\n' % (k+1,rankability[k],elo_corr[k-1]))
            else:
                f1.write(',,%d,%.4f,%.4f\n' % (k+1,rankability[k],0))
        x.append(rankability[-1])
        y.append(np.average(elo_corr,weights=[k for k in range(len(elo_corr))]))
        z.append(eloPred('Atlantic Coast',year,elo_rating))
        f2.write(',,%.4f,%.4f,%.4f\n' % (x[-1],y[-1],z[-1]))
    # correlation between year summary data
    print('Atlantic Coast: ')
    corr,pval = spearmanr(x,y)
    print('\tSDR and EloCorr corr = %.4f' % corr)
    print('\tSDR and EloCorr pval = %.4f' % pval)
    corr,pval = spearmanr(x,z)
    print('\tSDR and EloPred corr = %.4f' % corr)
    print('\tSDR and EloPred pval = %.4f' % pval)
    # Big East round by round analysis and summary
    f1.write('Big East, Year, Round, Rankability, EloCorr\n')
    f2.write('Big East, Year, Rankability, EloCorr, EloPred\n')
    x = []; y = []; z = []
    for year in range(1995,2013):
        elo_corr,rankability,elo_rating = cfbData('Big East',year,"SR")
        f1.write(',%d,,,\n' % year)
        f2.write(',%d,,\n' % year)
        for k in range(len(rankability)):
            if(k>=1):
                f1.write(',,%d,%.4f,%.4f\n' % (k+1,rankability[k],elo_corr[k-1]))
            else:
                f1.write(',,%d,%.4f,%.4f\n' % (k+1,rankability[k],0))
        x.append(rankability[-1])
        y.append(np.average(elo_corr,weights=[k for k in range(len(elo_corr))]))
        z.append(eloPred('Big East',year,elo_rating))
        f2.write(',,%.4f,%.4f,%.4f\n' % (x[-1],y[-1],z[-1]))
    # correlation between year summary data
    print('Big East: ')
    corr,pval = spearmanr(x,y)
    print('\tSDR and EloCorr corr = %.4f' % corr)
    print('\tSDR and EloCorr pval = %.4f' % pval)
    corr,pval = spearmanr(x,z)
    print('\tSDR and EloPred corr = %.4f' % corr)
    print('\tSDR and EloPred pval = %.4f' % pval)
    # Mountain West round by round analysis and summary
    f1.write('Mountain West, Year, Round, Rankability, EloCorr\n')
    f2.write('Mountain West, Year, Rankability, EloCorr, EloPred\n')
    x = []; y = []; z = []
    for year in range(1999,2012):
        elo_corr,rankability,elo_rating = cfbData('Mountain West',year,"SR")
        f1.write(',%d,,,\n' % year)
        f2.write(',%d,,\n' % year)
        for k in range(len(rankability)):
            if(k>=1):
                f1.write(',,%d,%.4f,%.4f\n' % (k+1,rankability[k],elo_corr[k-1]))
            else:
                f1.write(',,%d,%.4f,%.4f\n' % (k+1,rankability[k],0))
        x.append(rankability[-1])
        y.append(np.average(elo_corr,weights=[k for k in range(len(elo_corr))]))
        z.append(eloPred('Mountain West',year,elo_rating))
        f2.write(',,%.4f,%.4f,%.4f\n' % (x[-1],y[-1],z[-1]))
    # correlation between year summary data
    print('Mountain West: ')
    corr,pval = spearmanr(x,y)
    print('\tSDR and EloCorr corr = %.4f' % corr)
    print('\tSDR and EloCorr pval = %.4f' % pval)
    corr,pval = spearmanr(x,z)
    print('\tSDR and EloPred corr = %.4f' % corr)
    print('\tSDR and EloPred pval = %.4f' % pval)
    # close files
    f1.close()
    f2.close()

if __name__ == '__main__':
    main()