import datetime, json, powerlaw, pickle, copy, warnings
import numpy as np
import pandas as pd
import random
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels as sm
import scipy.stats as st
from scipy.stats import norm, truncnorm,uniform,powerlaw, lognorm, johnsonsu

home_folder = './jee-pickles/'
out_folder = './jee-figs/'

def initialize():
    """
        Not in use.
        This function is used to create picke objects from datasets.
        We already provide the pickle files.
    """
    ## To create initial record
    from meza import io
    record = io.read(home_folder+'jee2009.mdb')
    result=[]
    for i in range(384977): result.append(next(record))
    pickle.dump(result, open(home_folder+"jeeresult.pickle", "wb"))

    mark=[]
    for r in result: mark.append(int(r['mark']))
    mark.sort(key=lambda i:-int(i))
    pickle.dump(mark, open(home_folder+"jeemark.pickle", "wb"))


mark=pickle.load(open(home_folder+"jeemark.pickle", "rb"))
result=pickle.load(open(home_folder+"jeeresult.pickle", "rb"))

result.sort(key=lambda i:-int(i['mark']))

## Gather female results
fem=[]
mal=[]
for i in range(len(result)):
    if result[i]['GENDER']=='F':fem.append(result[i])
    if result[i]['GENDER']=='M':mal.append(result[i])


#########################
## Subsample males
#########################
random.shuffle(mal)
mal_sub=mal[:int(len(fem))]

def get_sub_samples_plot(mal_sub, fem_sub,thresh=1000, plot=0, save=0):
    fem_mark=[]
    mal_mark=[]
    for i in range(len(fem_sub)):
        if int(fem_sub[i]['mark']) <= thresh:
            fem_mark.append(int(fem_sub[i]['mark']))
    for i in range(len(mal_sub)):
        if int(mal_sub[i]['mark']) <= thresh:
            mal_mark.append(int(mal_sub[i]['mark']))
    mal_mark = np.array(mal_mark)
    fem_mark = np.array(fem_mark)
    print("Male mean is ", np.mean(mal_mark), ". There length is", len(mal_mark))
    print("Female mean is ", np.mean(fem_mark), ". There length is", len(fem_mark))
    if plot==1:
        plt.figure(figsize=(9,5))
        sns.distplot(mal_mark, hist = False, kde = True, kde_kws = {'shade': True, 'linewidth': 3}, label = "Male Candidates (Subsampled)")
        sns.distplot(fem_mark, hist = False, kde = True, kde_kws = {'shade': True, 'linewidth': 3}, label = "Female Candidates")
        legend = plt.legend(loc='best', shadow=False, fontsize=20)
        plt.ylim(0,0.019)
        plt.tick_params(axis='both', which='major', labelsize=15)
        plt.tick_params(axis='both', which='minor', labelsize=8)
        plt.xlabel("Total Score", fontsize=18)
        if save==1:
            plt.savefig(out_folder+'subsampling_plot.eps', format='eps', dpi=150, transparent=True)
            plt.savefig(out_folder+'subsampling_plot.png', format='png', dpi=150)
    plt.show()
    return np.mean(fem_mark), np.mean(mal_mark)

fem_mark, mal_mark = get_sub_samples_plot(mal_sub,fem)

fem_mean=[]
mal_mean=[]
for i in range(100):
    random.shuffle(mal)
    mal_sub=mal[:int(len(fem)/100)]
    random.shuffle(fem)
    fem_sub=fem[:int(len(fem)/100)]
    x, y = get_sub_samples_plot(mal_sub,fem_sub,10000,0,0)
    fem_mean.append(x)
    mal_mean.append(y)

print("Male mean of means over 1000 trials is ", np.mean(mal_mean), ", sd=", np.std(mal_mean))
print("Female mean of means over 1000 trials is ", np.mean(fem_mean), ", sd=", np.std(fem_mean))


########################################
## Get Female scores
########################################

def get_female_scores():
    fem_mark=[]
    mal_mark=[]
    for i in range(len(result)):
        if result[i]['GENDER']=='F':fem_mark.append(int(result[i]['mark']))
        if result[i]['GENDER']=='M':mal_mark.append(int(result[i]['mark']))
    mal_mark = np.array(mal_mark)
    fem_mark = np.array(fem_mark)
    plt.figure(figsize=(9,5))
    sns.kdeplot(mal_mark, shade=True, linewidth=3, label = "Male Candidates")
    sns.kdeplot(fem_mark, shade=True, linewidth=3, label = "Female Candidates")
    legend = plt.legend(loc='best', shadow=False, fontsize=20)
    plt.ylim(0,0.019)
    plt.ticklabel_format(style='sci', axis='y', scilimits=(-1,0))
    plt.tick_params(axis='both', which='major', labelsize=15)
    plt.tick_params(axis='both', which='minor', labelsize=8)
    plt.ylabel("Probability Density Function", fontsize=18)
    plt.xlabel("Total Score", fontsize=18)
    plt.savefig(out_folder+'jee_distribution_of_men_vs_women.eps', format='eps', dpi=150, transparent=True)
    plt.savefig(out_folder+'jee_distribution_of_men_vs_women.png', format='png', dpi=150)
    plt.show()
    return fem_mark, mal_mark

fem_mark, mal_mark = get_female_scores()

## Get TV distances
fem_hist=np.histogram(fem_mark, bins=[i for i in range(-105,481)])[0]/len(fem_mark)
mal_hist=np.histogram(mal_mark, bins=[i for i in range(-105,481)])[0]/len(mal_mark)
dist_tv = 0.5 * np.sum(np.abs(np.array(fem_hist) - np.array(mal_hist)))

print("TV distance  between the distributions is", dist_tv)

# Multiplicative correction
beta=np.mean(fem_mark+105)/np.mean(mal_mark+105)
print("beta is", beta)
print("beta is", 1/beta)
fem_mark_corrected = copy.deepcopy((fem_mark+105)/beta)-105

########################################
## Fit distribution for men and women
########################################
def find_best_fitting_distribution():
    def best_fit_distribution(data, bins=200, ax=None):
        """Model data by finding best fit distribution to data"""
        # Get histogram of original data
        y, x = np.histogram(data, bins=bins, density=True)
        x = (x + np.roll(x, -1))[:-1] / 2.0
        # Distributions to check
        DISTRIBUTIONS = [
            st.alpha,st.anglit,st.arcsine,st.beta,st.betaprime,st.bradford,st.burr,st.cauchy,st.chi,st.chi2,st.cosine,
            st.dgamma,st.dweibull,st.erlang,st.expon,st.exponnorm,st.exponweib,st.exponpow,st.f,st.fatiguelife,st.fisk,
            st.foldcauchy,st.foldnorm,st.frechet_r,st.frechet_l,st.genlogistic,st.genpareto,st.gennorm,st.genexpon,
            st.genextreme,st.gausshyper,st.gamma,st.gengamma,st.genhalflogistic,st.gilbrat,st.gompertz,st.gumbel_r,
            st.gumbel_l,st.halfcauchy,st.halflogistic,st.halfnorm,st.halfgennorm,st.hypsecant,st.invgamma,st.invgauss,
            st.invweibull,st.johnsonsb,st.johnsonsu,st.ksone,st.kstwobign,st.laplace,st.levy,st.levy_l,st.levy_stable,
            st.logistic,st.loggamma,st.loglaplace,st.lognorm,st.lomax,st.maxwell,st.mielke,st.nakagami,st.ncx2,st.ncf,
            st.nct,st.norm,st.pareto,st.pearson3,st.powerlaw,st.powerlognorm,st.powernorm,st.rdist,st.reciprocal,
            st.rayleigh,st.rice,st.recipinvgauss,st.semicircular,st.t,st.triang,st.truncexpon,st.truncnorm,st.tukeylambda,
            st.uniform,st.vonmises,st.vonmises_line,st.wald,st.weibull_min,st.weibull_max,st.wrapcauchy]
        # Best holders
        best_distribution = st.norm
        best_params = (0.0, 1.0)
        best_sse = np.inf
        # Estimate distribution parameters from data
        for distribution in DISTRIBUTIONS:
            # Try to fit the distribution
            print(distribution.name)
            if distribution in [st.levy,st.levy_l,st.levy_stable,st.ncf,st.nct,st.tukeylambda]: ## these distribution case some infinite loops
                continue
            try:
                # Ignore warnings from data that can't be fit
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore')
                    # fit dist to data
                    params = distribution.fit(data)
                    # Separate parts of parameters
                    arg = params[:-2]
                    loc = params[-2]
                    scale = params[-1]
                    # Calculate fitted PDF and error with fit in distribution
                    pdf = distribution.pdf(x, loc=loc, scale=scale, *arg)
                    sse = np.sum(np.power(y - pdf, 2.0))
                    # if axis pass in add to plot
                    try:
                        if ax:
                            pd.Series(pdf, x).plot(ax=ax)
                    except Exception:
                        pass
                    print("Params: ", params)
                    print("SSE: ", sse)
                    # identify if this distribution is better
                    if best_sse > sse > 0:
                        best_distribution = distribution
                        best_params = params
                        best_sse = sse
            except Exception:
                pass
        return (best_distribution.name, best_params)
    #######################################
    # Load data for Women
    #######################################
    data = pd.Series(fem_mark)
    # Plot for comparison
    plt.figure(figsize=(12,8))
    ax = data.plot(kind='hist', bins=50, density=True, alpha=0.5, color='b')
    # Save plot limits
    dataYLim = ax.get_ylim()
    best_fit_name, best_fit_params = best_fit_distribution(data, 200, ax)
    print("Best fit for women: ", best_fit_name, best_fit_params)
    #######################################
    # Load data for Women corrected
    #######################################
    data = pd.Series(fem_mark)
    # Plot for comparison
    plt.figure(figsize=(12,8))
    ax = data.plot(kind='hist', bins=50, density=True, alpha=0.5, color='b')
    # Save plot limits
    dataYLim = ax.get_ylim()
    best_fit_name, best_fit_params = best_fit_distribution(data, 200, ax)
    print("Best fit for women: ", best_fit_name, best_fit_params)
    #######################################
    # Load data for Men
    #######################################
    data = pd.Series(mal_mark)
    # Plot for comparison
    plt.figure(figsize=(12,8))
    ax = data.plot(kind='hist', bins=50, density=True, alpha=0.5, color='b')
    # Save plot limits
    dataYLim = ax.get_ylim()
    best_fit_name, best_fit_params = best_fit_distribution(data, 200, ax)
    print("Best fit for men: ", best_fit_name, best_fit_params)


# Uncomment to fit the distributions again.
# We already added the fitted parameters in the code below.
# find_best_fitting_distribution()
