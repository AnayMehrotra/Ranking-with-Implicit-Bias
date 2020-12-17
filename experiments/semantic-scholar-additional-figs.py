import datetime, json, powerlaw, pickle
import numpy as np
import matplotlib.pyplot as plt
import unidecode
import seaborn as sns

#######################################
## Read the dataset and convert it to
## more managable python objects
#######################################
def initialize():
    """
        Not in use.
        This function is used to create picke objects from datasets.
        We already provide the pickle files.
    """
    author = {} # author[a]['citations'] is the number of citations of author a.
    paper = {} # paper[p]['authors'] the list of authors of paper p.

    cnt_no_yr =0 ## number of papers without an year

    for index in range(0,47):
        if index>9: f=open('s2-corpus-'+str(index),"r")
        else: f=open('s2-corpus-0'+str(index),"r")
        print("File no:", index)
        tmp = f.readline()
        i=0
        while len(tmp)>0:
            i+=1
            mp = json.loads(tmp)
            if mp['id'] in paper: print("error!")
            if 'year' not in mp.keys():
                cnt_no_yr +=1 ## do not take papers without a year
                tmp = f.readline()
                continue
            for a in mp['authors']:
                if len(a['ids'])==0: continue
                id = a['ids'][0]
                if id not in author:
                    author[id]={'citations':0}
                    author[id]['first_publish'] = int(mp['year'])
                    author[id]['last_publish'] = int(mp['year'])
                    author[id]['name']=a['name']
                author[id]['citations']+=len(mp['outCitations'])
                author[id]['first_publish'] = min(author[id]['first_publish'], int(mp['year']))
                author[id]['last_publish'] = max(author[id]['last_publish'], int(mp['year']))
            tmp = f.readline()
            if i%100000==1:print(i)
        f.close()

    print(cnt_no_yr) ## 874619
    print(len(author)) ## 17790560

    #######################################
    ## Pickle objects
    #######################################

    pickle.dump(author, open("semantic-scholar-pickles/author_ss_dict_id_and_citation-count"+str(datetime.datetime.today())+".pickle", "wb"))
    pickle.dump(citation, open("semantic-scholar-pickles/citation_count_ss_sorted"+str(datetime.datetime.today())+".pickle", "wb"))
    del(author) ## this is a large object only load it if needed

out_folder = './semantic-scholar-figs/'

#######################################
## Load pickled objects
#######################################
citation=pickle.load(open("semantic-scholar-pickles/citation_count_ss_sorted2019-07-16_19_30_59_180426.pickle", "rb"))
author=pickle.load(open("semantic-scholar-pickles/author_ss_dict_id_and_citation-count2019-07-2403-07-59-289763.pickle", "rb"))
print("done: loading")

#######################################
## Read the name database and conver to
## more managable python objects
#######################################

name={}
for index in range(1880,2019):
    f=open('./names/yob'+str(index)+".txt","r")
    if index%20==0:print("File no:", index)
    tmp = f.readline()
    i=0
    while len(tmp)>0:
        i+=1
        n, gender, cnt = tmp.split(',')
        n=n.lower()
        if n not in name: name[n]={'m':0, 'f':0}
        if gender=='F':name[n]['f']+=int(cnt)
        elif gender=='M':name[n]['m']+=int(cnt)
        else: print("Error:", tmp)
        tmp = f.readline()
    f.close()

for n in name:
    if name[n]['m'] > name[n]['f']:
        name[n]['p'] = name[n]['m']/(name[n]['m']+name[n]['f'])
        name[n]['gender']='m'
    elif name[n]['m'] <= name[n]['f']:
        name[n]['p'] = name[n]['f']/(name[n]['f']+name[n]['m'])
        name[n]['gender']='f'

print("Number of names", len(name)) ## 98400

#######################################
## Plot tradeoff when setting different
## thresholds for names
#######################################
def plot_tradeoff_for_names(author, name):
    num_m,num_f,num_u, num_no =[],[],[], []
    thresh = [0.5,0.6,0.7,0.8,0.9,0.93,0.95,0.97,0.98,0.99]
    for x in thresh:
        print("Threshold:", x)
        male, female, unknown, no_first_name = [], [], [], []
        for key in author.keys():
            a=author[key]
            a_first_name=a['name'].split(' ')[0].split('.')[0].split('-')[0].lower()
            a_first_name=unidecode.unidecode(a_first_name)
            if len(a_first_name) <= 2:
                no_first_name.append(a)
            elif a_first_name in name:
                if name[a_first_name]['p'] > x and name[a_first_name]['gender']=='f':
                    female.append(a)
                elif name[a_first_name]['p'] > x and name[a_first_name]['gender']=='m':
                    male.append(a)
                else: unknown.append(a)
            else: unknown.append(a)
        print("Length male, female, unknown, no_first_name:", len(male), len(female), len(unknown), len(no_first_name))
        print("Total sum:", len(male)+len(female)+len(unknown)+len(no_first_name), "vs unique authors", len(author))
        num_m.append(len(male))
        num_f.append(len(female))
        num_u.append(len(unknown))
        num_no.append(len(no_first_name))
    #
    #
    plt.figure(figsize=(9,5))
    plt.plot(thresh, num_m,'--', label="Men",  linewidth=4)
    plt.plot(thresh, num_f, label="Women",  linewidth=4)
    plt.plot(thresh, num_u, ':', label="Unknown first name",  linewidth=4)
    legend = plt.legend(loc='best', shadow=False, fontsize=20)
    plt.ylabel("Number of Candidates Admitted", fontsize=16)
    plt.xlabel("Fraction of Women of Total Admitted Candidates", fontsize=16)
    plt.ylim(0,16000)
    plt.ticklabel_format(style='sci', axis='y', scilimits=(-1,3))
    plt.ylabel("Number of authors",fontsize=20)
    plt.xlabel("Threshold",fontsize=20)
    plt.ylim(0,0.5*len(author))
    plt.savefig(out_folder+'count_vs_thresh_tradeoff2.eps', format='eps', dpi=500)
    # plt.show()
    return num_m, num_f, num_u, num_no
num_m, num_f, num_u, num_no = plot_tradeoff_for_names(author, name)
print("done: plot_tradeoff_for_names")

#################################################
## Find male and female authors given a threshold
#################################################
def partition_authors_by_gender(thresh, author, name):
    male, female, unknown, no_first_name = [], [], [], []
    for key in author.keys():
        a=author[key]
        a_first_name=a['name'].split(' ')[0].split('.')[0].split('-')[0].lower() ## get first name
        a_first_name=unidecode.unidecode(a_first_name) ## convert non ascii characters
        if len(a_first_name) <= 2:  ## remove initials
            no_first_name.append(a)
        elif a_first_name in name:
            if name[a_first_name]['p'] > thresh and name[a_first_name]['gender']=='f':
                female.append(a)
            elif name[a_first_name]['p'] > thresh and name[a_first_name]['gender']=='m':
                male.append(a)
            else: unknown.append(a)
        else: unknown.append(a)
    print("Length male, female, unknown, no_first_name:", len(male), len(female), len(unknown), len(no_first_name))
    print("Total sum:", len(male)+len(female)+len(unknown)+len(no_first_name), "vs unique authors", len(author))
    return male, female, unknown, no_first_name
male, female, unknown, no_first_name = partition_authors_by_gender(0.9, author, name)
print("done: partition_authors_by_gender")

#######################################
## Plot number of authors by year
#######################################
def plot_number_of_authors_by_first_publication_yr(male, female):
    min_yr = 2000
    for m in male: min_yr = min(m['first_publish'], min_yr)
    for f in female: min_yr = min(f['first_publish'], min_yr)
    print("First citation in the dataset at", min_yr)
    yrs = [1936,1940,1950,1960,1970,1980,1990,2000,2010,2020]
    male_yr, female_yr = [], []
    for yr in yrs:
        print(yr)
        male_cit_r=[]
        female_cit_r=[]
        for m in male:
            if m['first_publish'] <= yr: male_cit_r.append(m['citations'])
        for f in female:
            if f['first_publish'] <= yr: female_cit_r.append(f['citations'])
        male_yr.append(len(male_cit_r))
        female_yr.append(len(female_cit_r))
    plt.plot(yrs, male_yr, '--', label="Male", linewidth=4)
    plt.plot(yrs, female_yr, label="Female", linewidth=4)
    legend = plt.legend(loc='best', shadow=False, fontsize=20)
    plt.ticklabel_format(style='sci', axis='y', scilimits=(-1,3))
    plt.xlabel("Year", fontsize=20)
    plt.ylabel("Number of authors", fontsize=20)
    plt.savefig(out_folder+'num_by_first_pub_yr.eps', format='eps', dpi=150)
    plt.savefig(out_folder+'num_by_first_pub_yr.png', format='png', dpi=150)
    # plt.show()
    return male_yr, female_yr
male_yr, female_yr = plot_number_of_authors_by_first_publication_yr(male, female)
print("done: plot_number_of_authors_by_first_publication_yr")

#################################################
## Find unweighted citations of authors
#################################################
def distribution_of_citations(male, female, after_yr=1990, plot=0):
    male_cit_u=[]
    female_cit_u=[]
    for m in male:
        if m['first_publish'] >= after_yr: male_cit_u.append(m['citations'] )
    for f in female:
        if f['first_publish'] >= after_yr: female_cit_u.append(f['citations'] )
    male_cit_u = np.array(male_cit_u)
    female_cit_u = np.array(female_cit_u)
    ## make a histogram of distribution
    if plot:
        plt.figure(figsize=(8.2,5))
        powerlaw.plot_pdf(male_cit_u[male_cit_u>0], label="Male", color='b', linewidth=4)
        powerlaw.plot_pdf(female_cit_u[female_cit_u>0], label="Female", color='orange', linewidth=4)
        plt.xscale('linear')
        plt.ticklabel_format(style='sci', axis='x', scilimits=(-1,3))
        legend = plt.legend(loc='best', shadow=False, fontsize=20)
        after_yr=1980
        plt.xlabel("Total Citations (Papers after "+str(after_yr)+")",fontsize=20)
        plt.ylabel("Probability Mass Function", fontsize=20)
        plt.savefig(out_folder+'citation_unweighted_'+str(after_yr)+'.eps', format='eps', dpi=500)
        plt.savefig(out_folder+'citation_unweighted_'+str(after_yr)+'.png', format='png', dpi=150)
        # plt.show()
    return male_cit_u, female_cit_u

male_cit_u, female_cit_u = distribution_of_citations(male, female, after_yr=1980, plot=1)
print("done: distribution_of_citations")
