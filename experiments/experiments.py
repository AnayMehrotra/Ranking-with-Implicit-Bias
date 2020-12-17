from scipy.stats import truncnorm,uniform,powerlaw, lognorm, johnsonsu
import numpy as np
import pickle
import itertools
import time

import matplotlib.pyplot as plt

##################################################
## Parameters
##################################################
# Test
beta = 0.5
alpha = 0.0 # strength of fairness constraints. 0.0 Min -- 1.0 Max
k = 100 # length of ranking
dataset_name = "unknown"

##################################################
## Utility
##################################################

## Generate random people for the experiment
## value[l][i]: i-th largest item of type l
## shades it by a factor of beta
def biased_regresser(size, ty,beta=1.0):
    #
    value = []
    #
    if dataset_name=="SS":
        ## Best fit for citation dataset
        value=lognorm.rvs(1.604389429520587, 48.91174576443938, 77.36426476362374, size=size)
    elif dataset_name=="JEE":
        # Best fit for JEE scoress
        if ty==0: value=johnsonsu.rvs(-1.3358254338685507, 1.228621987785165, -16.10471198333935, 25.658144591068066, size=size) ## Men
        if ty==1: value=johnsonsu.rvs(-1.1504808824385124, 1.3649066883190795, -12.879957294149737, 27.482272133428403, size=size) ## Women
    else:
        print("Unknown dataset_name=%x", dataset_name)
        exit();
    #
    if ty == 1:value*=(beta+1e-4);
    #
    return [ {'val':val, 'real_type':ty} for val in value]

## Constraint at position pos for type ty
def get_consU(pos, type):
    #
    if type==0:
        return pos-np.floor(pos * alpha)
    else:
        return 10000000

##################################################
## Ranking Algorithms
##################################################
def fair_ranking_algorithm(p, k, items):
    fair_ranking = [] ## fair_ranking[i]: {'id': ..., 'pseudo_type':..., 'pos': i, 'real_type':...}
    cnt = [0 for l in range(p)] # cnt how many of each type were taken
    for j in range(k): # for each position
        best_score, best_type = -1, -1; # current best candidate
        #
        for l in range(p): # loop over properties
            #
            if cnt[l] < get_consU(j+1, l) and cnt[l] < len(items[l])  and best_score < items[l][cnt[l]]['val']:
                best_score = items[l][cnt[l]]['val']
                best_type = l
                real_type = items[l][cnt[l]]['real_type']
        #
        if best_score == -1:
            print("INFEASIBLE!") # no feasible fair ranking
            return []
        else:
            fair_ranking.append({'id': cnt[best_type], 'pseudo_type':best_type, 'pos':j, 'real_type': real_type, 'val':best_score})
            cnt[best_type] += 1
    #
    return fair_ranking

def sorting_algorithm(p, k, items):
    sort_ranking = [] ## fair_ranking[i]: {'id': ..., 'pseudo_type':..., 'pos': i, 'real_type':...}
    cnt = [0 for l in range(p)] #cnt how many of each type were taken
    #
    for j in range(k): # for each position
        best_score, best_type = -1, -1; # current best candidate
        #
        for l in range(p):  # loop over properties
            if l == 1 and cnt[l] < len(items[l]) and best_score < items[l][cnt[l]]['val']/(beta+1e-4) * 1.0:
                best_score = items[l][cnt[l]]['val']
                best_type = l
                real_type = items[l][cnt[l]]['real_type']
            elif cnt[l] < len(items[l]) and best_score < items[l][cnt[l]]['val']:
                best_score = items[l][cnt[l]]['val']
                best_type = l
                real_type = items[l][cnt[l]]['real_type']
        #
        if best_score == -1:
            print("INFEASIBLE!") # no feasible ranking
            return []
        else:
            sort_ranking.append({'id': cnt[best_type], 'pseudo_type':best_type, 'pos':j, 'real_type': real_type,  'val':best_score})
            cnt[best_type] += 1
    #
    return sort_ranking

##################################################
## Functions to run experiments
##################################################

trials = 5000
def run_test(p,k,given_ranking_algorithm,beta=1.0,ratio_of_group=1.0):
    global alpha
    #
    result = [] # Stores the results of all runs
    size=[1.0,ratio_of_group];
    #
    print("alpha x beta:", alpha, beta, ratio_of_group);
    #
    a,b=0,0
    #
    # run many iterations of the ranking, to get expectation over the rankings
    for i in range(trials):
        value = [biased_regresser(int(1000*size[type]),type,beta) for type in range(p)] # Run the regresser
        #
        items = [ [] for i in range(p)] # Stores all information about the items
        #
        # Generate
        # - pseudo labels for items based on the classifier
        # - store values of the items
        for l in range(p):
            for t in range(int(1000*size[l])):
                items[l].append({'val':value[l][t]['val'], 'pseudo_type':1, 'real_type':l})
        #
        for l in range(p): items[l].sort(key=lambda i: i['val'], reverse=True)
        # run the fair ranking algorithm
        fair_ranking = given_ranking_algorithm(p,k,items);
        #
        # no feasible ranking found
        if len(fair_ranking)==0: continue;
        #
        result_ = [ ] # store results
        for j, item in enumerate(fair_ranking): result_.append( {'type':item['real_type'], 'utility': item['val']} )
        #
        result.append(result_)
    #
    ## cnt[i]['type']: type of the i-th item
    ## cnt[i]['utility']: percieved utility of the i-th item
    return result,a,b

def run_global(exp_num, dataset="JEE"):
    global alpha ## fairness constrain
    global k ## number of positions in ranking
    global beta ## biased utility
    global dataset_name ## name of dataset JEE or SS
    #
    dataset_name = dataset
    #
    k=100
    algorithm = fair_ranking_algorithm;
    #
    x=[i/20 for i in range(0,11,1)] ## range of alpha
    y=[0.25,0.5,1.0] ## range of beta
    z_list=[[0.25],[0.33],[0.5]] ## range of ratio_of_group
    #
    if exp_num==1: ## Constrained ranking algorithm with bias
        for z in z_list:
            result = {}
            for b,ratio_of_group in itertools.product(y,z):
                for a in x:
                    alpha = a # varying constraints
                    beta = b # varying bias
                    result[(a,b,ratio_of_group)] = {'result': run_test(2,k,algorithm,b,ratio_of_group)[0], 'trials':trials, 'k':k}
                file = open(dataset+str(trials)+"-alg="+str(algorithm.__name__)+"_r="+str(int(z[0]*100))+".pickle", "wb")
                pickle.dump(result,file)
                file.close()
    elif exp_num==2: ## Optimal ranking algorithm
        x=[0] ## range of alpha
        y=[1] ## range of beta
        for z in z_list:
            result = {}
            for b,ratio_of_group in itertools.product(y,z):
                for a in x:
                    alpha = a # varying constraints
                    beta = b # varying bias
                    result[(a,b,ratio_of_group)] = {'result': run_test(2,k,algorithm,b,ratio_of_group)[0], 'trials':trials, 'k':k}
                file = open(dataset+str(trials)+"-no_implicit_bias_r="+str(int(z[0]*100))+".pickle", "wb")
                pickle.dump(result,file)
                file.close()
    elif exp_num==3: ## Optimal unconstrained ranking
        x=[0] ## range of alpha
        for z in z_list:
            result = {}
            for b,ratio_of_group in itertools.product(y,z):
                for a in x:
                    alpha = a # varying constraints
                    beta = b # varying bias
                    result[(a,b,ratio_of_group)] = {'result': run_test(2,k,algorithm,b,ratio_of_group)[0], 'trials':trials, 'k':k}
            file = open(dataset+str(trials)+"-no_contraints_r="+str(int(z[0]*100))+".pickle", "wb")
            pickle.dump(result,file)
            file.close()

##################################################
## Get characteristics of ranking
##################################################

USE_DCG = False

def get_utility(ranking,beta):
    w=0
    cnt=0
    for j in range(len(ranking)):
        cnt+=ranking[j]['type']
        if USE_DCG:
            if ranking[j]['type']==1: w+=ranking[j]['utility']/(beta+1e-4) * 1 / np.log(j+1); ## get latent utility with no positional discount
            else: w+=ranking[j]['utility'] * 1 / np.log(j+1) ## no positional discount
        else:
            if ranking[j]['type']==1: w+=ranking[j]['utility']/(beta+1e-4) * 1; ## get latent utility with no positional discount
            else: w+=ranking[j]['utility'] * 1 ## no positional discount
    w=w/len(ranking);
    return w, cnt

def get_utility_from_result(result, b):
    utility_ = [  get_utility(result_,b)[0] for result_ in result]
    return np.mean(utility_)

##################################################
## Generate plot for experiment
##################################################
def plot_1_paper(folder=".", dataset="JEE"):
    import matplotlib.pyplot as plt;
    #
    global alpha
    global beta
    global dataset_name ## name of dataset JEE or SS
    dataset_name = dataset
    #
    alpha = 1.0
    #
    x=[i/20 for i in range(0,11,1)] ## range of alpha
    y=[0.25,0.5,1.0] ## range of beta
    z_list=[[0.25],[0.33],[0.5]] ## range of ratio_of_group
    #
    for z in z_list:
        ## get results
        print("Loading constrained")
        file=open(dataset+str(trials)+'-alg=fair_ranking_algorithm_r='+str(int(z[0]*100))+'.pickle', "rb")
        result_const = pickle.load(file)
        file.close()
        #
        print("Loading optimal")
        file=open(dataset+str(trials)+'-no_implicit_bias_r='+str(int(z[0]*100))+'.pickle', "rb")
        result_nobias = pickle.load(file)
        file.close()
        #
        print("Loading no constraints")
        file=open(dataset+str(trials)+'-no_contraints_r='+str(int(z[0]*100))+'.pickle', "rb")
        result_noconst = pickle.load(file)
        file.close()
        #
        print("Done loading")
        #
        for b,r in itertools.product(y,z):
            utility = []
            #
            print(result_nobias.keys())
            print(result_noconst.keys())
            #
            ra=r;
            if list(result_nobias.keys())[0][2] == 0.333: ra=0.333;
            rb=r;
            if list(result_noconst.keys())[0][2] == 0.333: rb=0.333;
            #
            ## latent utility
            latent_utility = [get_utility_from_result(result_nobias[(0,1,ra)]['result'], 1)] * len(x)
            latent_utility=np.array(latent_utility)
            #
            ## biased utility
            biased_utility = [get_utility_from_result(result_noconst[(0,b,rb)]['result'], b)] * len(x)
            biased_utility=np.array(biased_utility)
            #
            for a in x:
                print(a, r, ": ", b)
                alpha = a
                beta = b
                #
                rc=r;
                if list(result_const.keys())[0][2] == 0.333: rc=0.333;
                #
                result_cur = result_const[(a,b,rc)]['result']
                utility.append( get_utility_from_result(result_cur, b) )
            #
            utility=np.array(utility)
            #
            plt.figure(figsize=(9,6))
            #
            k=1;
            x_strech=[i/15-0.1 for i in range(0,11,1)] ## range of alpha
            plt.plot(x,utility/k, color='b', label='Cons', linewidth=4)
            plt.plot(x_strech,latent_utility/k, '--', color='r', label='Opt', linewidth=4)
            plt.plot(x_strech,biased_utility/k, '--', color='g', label='Uncons', linewidth=4)
            # legend = plt.legend(loc='best', shadow=False, fontsize=30)
            plt.xlabel('$\\alpha$',fontsize=25)
            plt.ylabel('Average Latent Utility', fontsize=25)
            # plt.ylim(1.8*trials/5,2.5*trials/5)
            plt.xlim(0,0.5)
            plt.locator_params(axis='y', nbins=4)
            plt.locator_params(axis='x', nbins=5)
            plt.tick_params(axis='both', which='major', labelsize=16)
            plt.xticks(list(plt.xticks()[0]) + [0.5])
            # plt.title('$\\beta=$'+str(b)+', $m_b/m_a$='+str(r), fontsize=15)
            strtmp = folder+'/'+dataset+str(trials)+'beta='+str(int(100*b))+"ratio"+str(int(r*100))
            plt.savefig(strtmp+'.eps', format='eps', dpi=150)
            plt.savefig(strtmp+'.png', format='png', dpi=150)
        #
        del(result_const)
        del(result_nobias)
        del(result_noconst)


USE_DCG = False
folder = './semantic-scholar-figs/no-dcg'

for i in range(3): run_global(i+1, dataset="SS")
plot_1_paper(folder, dataset="SS")

USE_DCG = True
folder = './semantic-scholar-figs/dcg'

for i in range(3): run_global(i+1, dataset="SS")
plot_1_paper(folder, dataset="SS")



USE_DCG = False
folder = './jee-figs/no-dcg'

for i in range(3): run_global(i+1, dataset="JEE")
plot_1_paper(folder, dataset="JEE")

USE_DCG = True
folder = './jee-figs/dcg'

for i in range(3): run_global(i+1, dataset="JEE")
plot_1_paper(folder, dataset="JEE")
