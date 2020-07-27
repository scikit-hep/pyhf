import scipy.stats
import numpy as np
import matplotlib.pyplot as plt


def get_interval(mu,sigma,level,teststat):
    muhat_over_sigma = np.asarray([-1000,-900] + np.linspace(-50,50,5000).tolist() + [900,1000])
    muhat = muhat_over_sigma*sigma
        
    delta_mu = muhat-mu
    delta_mu_over_sigma = delta_mu/sigma
    q = teststat(mu,muhat,sigma)

    a = delta_mu[np.argmax(q<level)]
    b = delta_mu[::-1][np.argmax(q[::-1]<level)]

    d = scipy.stats.norm(loc = 0, scale = sigma)
    size = d.cdf(a) + 1-d.cdf(b)
    return delta_mu_over_sigma,q,a/sigma,b/sigma,size

def scan_tests_for_size(mu,sigma,teststat):
    scan_results = []
    for level in np.linspace(25,1,101).tolist() + np.logspace(0,-5,51).tolist():
        result = x,q,a,b,size = get_interval(mu,sigma,level,teststat)
        result = list(result)
        scan_results.append([level,a,b,size])
    return np.asarray(scan_results)

def find_level_for_size(mu,sigma,target_size,teststat):
    a = scan_tests_for_size(mu,sigma,teststat)
    result = level,a,b,size = a[np.argmax(a[:,-1]>target_size)]
    return level

def jointplot(mu,sigma,teststat,tcut = 4):
    f,axarr = plt.subplots(2,2)

    muhat = np.random.normal(mu,scale = sigma,size = 1000)
    
    muhat_over_sigma_scan = np.linspace(-5,5)
    tmulin = teststat(mu,muhat_over_sigma_scan*sigma,sigma = sigma)
    
    tmu = teststat(mu,muhat,sigma = sigma)

    ax = axarr[1,1]
    ax.set_xlim(-5.5,5.5)

    bins = np.linspace(-10,10,41)
    ax.hist(muhat/sigma, density = False,bins = bins)
    ax.hist(muhat[tmu>tcut]/sigma, density = False,bins = bins, facecolor = 'orange')
    ax.set_xlabel('μ̂/σ')
    ax.set_ylim(0,200)
    
    
    ax = axarr[0,1]
    ax.set_ylim(-1,20)
    ax.set_xlim(-5.5,5.5)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.scatter(muhat/sigma,tmu)
    ax.scatter(muhat[tmu>tcut]/sigma,tmu[tmu>tcut], c = 'orange')
    ax.plot(muhat_over_sigma_scan,tmulin)

    ax = axarr[0,0]
    bins = np.linspace(0,20,21)
    ax.hist(tmu, orientation='horizontal', bins = bins, density = False)
    ax.hist(tmu[tmu>tcut], orientation='horizontal', bins = bins, density = False, facecolor = 'orange')
    ax.set_ylim(-1,20)
    ax.set_ylabel('t(μ̂,σ)')
    ax.set_xscale('log')
    ax.set_xlim(100000,1e-1)
    axarr[1,0].axis('off')
    f.set_size_inches(6,6)
    
def plot_teststat(min_mu,sigma,teststat):
    f,axarr = plt.subplots(2,1)
    muhat = np.linspace(-50,50,10000)

    mus = np.linspace(min_mu,5*sigma,11)
    ax = axarr[0]
    for mu in mus:
        q = teststat(mu,muhat,sigma)
        ax.plot(muhat/sigma,q)
        ax.set_xlim(-5,5)
        ax.set_ylim(-1,25)
        ax.set_xlabel('μ̂/σ')
    ax.set_ylabel('test statistic')
        
    ax = axarr[1]
    for mu in mus:
        q = teststat(mu,muhat,sigma)
        ax.plot((muhat-mu)/sigma,q)
        ax.set_xlim(-5,5)
        ax.set_ylim(-1,25)
        ax.set_xlabel('(μ̂-µ)/σ')
    ax.set_ylabel('test statistic')
    return f
    
def plot_oneinterval(ax,mu,sigma,size,teststat):
    level = find_level_for_size(mu,sigma,size, teststat)

    x,q,a,b,size = get_interval(mu,sigma,level,teststat)
    ax.plot(x,q)
    ax.vlines([a,b],0,25)
    ax.hlines([level],-20,20)
    ax.set_ylim(0,16)
    ax.set_xlim(-5,5)
    ax.set_xlabel('(μ̂-µ)/σ')
    ax.set_ylabel('test statistic')

def plot_neyman_construction(ax,min_mu,max_mu,hypos,scans,delta = True):
    if not delta:
        ax.plot(hypos+scans[:,1],hypos)
        ax.plot(hypos+scans[:,2],hypos)
        ax.hlines(hypos,hypos+scans[:,1],hypos+scans[:,2], colors = 'k', alpha = 0.2)
    else:
        ax.plot(scans[:,1],hypos)
        ax.plot(scans[:,2],hypos)
        ax.hlines(hypos,scans[:,1],scans[:,2], colors = 'k', alpha = 0.2)
    
    ax.set_xlim(-5,5)
    ax.set_ylim(min_mu,max_mu)
    colors = ['r','b','k','g','y']
    if not delta:
        ax.vlines([-2,-1,0,1,2],-5,5, colors = colors)
        ax.set_xlabel('μ̂/σ')
    else:
        deltas = np.linspace(-5,5)
        for n,c in zip([-2,-1,0,1,2],colors):
            ax.plot(deltas,n-deltas,c = c)
        ax.set_xlabel('(μ̂-µ)/σ')
    ax.set_ylabel('μ/σ')
        

def plot_cuts(ax,hypos_over_sigma,atcut,sigma,teststat):
    ax.plot(hypos_over_sigma,atcut[:,0])
    ax.vlines(hypos_over_sigma,0,atcut[:,0], colors = 'k', alpha = 0.2)
    ax.set_ylim(0,4)
    
    colors = ['r','b','k','g','y']
    for n,c in zip([-2,-1,0,1,2],colors):
        ax.plot(hypos_over_sigma,teststat(hypos_over_sigma*sigma,n*sigma,sigma = sigma),c=c) 
    ax.set_xlabel('μ/σ')
    ax.set_ylabel('test statistic')

def plot_pvalue(ax,hypos_over_sigma,scans,sigma,teststat,cuts):
    colors = ['r','b','k','g','y']
    for c,n in zip(colors,[-2,-1,0,1,2]):
        ax.plot(hypos_over_sigma,[np.interp(teststat(h,n,sigma=1),s[:-1,0][::-1],s[:-1,-1][::-1]) for s,h in zip(scans,hypos_over_sigma)],c=c)
        
    cuts = [
        np.interp(c,s[:-1,0][::-1],s[:-1,-1][::-1]) for c,s,h in
        zip(cuts,scans,hypos_over_sigma)
    ]
    ax.plot(hypos_over_sigma,cuts,c='steelblue')
    ax.vlines(hypos_over_sigma,cuts,1,alpha = 0.2)
    ax.set_ylim(0,1)
    ax.set_xlabel('μ/σ')    
    ax.set_ylabel('p-value')
    