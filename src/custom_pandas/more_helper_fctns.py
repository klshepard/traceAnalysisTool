# %load /proj/jakobuchheim/Repositories/Schnellstapel/src/custom_pandas/more_helper_fctns.py
# %load /proj/jakobuchheim/Repositories/Schnellstapel/src/custom_pandas/more_helper_fctns.py

def read_pckl_data(readFile):
    with open(readFile, "rb") as f:
        seg = pickle.load(f)
    if ((hasattr(seg,'hmmTrace')) & (hasattr(seg,'idealTrace'))):
        return seg.IaRAW, seg.Ia, seg.samplerate, seg.exp.metaData["SETUP_ADCSAMPLERATE"], seg.hmmTrace, seg.idealTrace
    elif hasattr(seg,'hmmTrace'):
        return seg.IaRAW, seg.Ia, seg.samplerate, seg.exp.metaData["SETUP_ADCSAMPLERATE"], seg.hmmTrace, None
    elif hasattr(seg,'idealTrace'):
        return seg.IaRAW, seg.Ia, seg.samplerate, seg.exp.metaData["SETUP_ADCSAMPLERATE"], None, seg.idealTrace
    elif not hasattr(seg,'IaRAW'):
        # k=np.zeros((len(seg[0])))
        # k[:]=seg[0]
        return None,seg[0] ,seg[1], None, None, None
    else:
        return seg.IaRAW, seg.Ia, seg.samplerate, seg.exp.metaData["SETUP_ADCSAMPLERATE"], None, None

def mergeEvents(df, firstEventIDX, secondEventIDX):
    ''' merges events on data frame. secondEventIDX is added to firstEventIDX. NOTE that STD is not calculated correctly.'''
    from scipy.stats import t
    df.loc[firstEventIDX,'EVENTDEPTH'] = (((df.loc[firstEventIDX,'EVENTDEPTH'] * df.loc[firstEventIDX,'EVENTDWELLTIME']) + 
                                             (df.loc[secondEventIDX,'EVENTDEPTH'] * df.loc[secondEventIDX,'EVENTDWELLTIME'])) / 
                                             (df.loc[firstEventIDX,'EVENTDWELLTIME'] + df.loc[secondEventIDX,'EVENTDWELLTIME']))
    df.loc[firstEventIDX,'EVENTSTD'] = (((df.loc[firstEventIDX,'EVENTSTD'] * df.loc[firstEventIDX,'EVENTDWELLTIME']) + 
                                             (df.loc[secondEventIDX,'EVENTSTD'] * df.loc[secondEventIDX,'EVENTDWELLTIME'])) / 
                                             (df.loc[firstEventIDX,'EVENTDWELLTIME'] + df.loc[secondEventIDX,'EVENTDWELLTIME']))
    df.loc[firstEventIDX,'EVENTDWELLTIME'] = (df.loc[secondEventIDX,'EVENTSTOPIDX'] + df.loc[firstEventIDX,'EVENTSTARTIDX']) / df.loc[secondEventIDX,'SAMPLERATE'] * 1e6 
    dof = np.ceil(df.loc[firstEventIDX,'EVENTDWELLTIME'] * 1e-6 * df.loc[firstEventIDX,'FILTERCUTOFF']                    )
    df.loc[firstEventIDX,'EVENTTVALUE'] = df.loc[firstEventIDX,'EVENTDEPTH'] / df.loc[firstEventIDX,'EVENTSTD'] * np.sqrt(dof)
    df.loc[firstEventIDX,'EVENTPVALUE'] = (t.sf(abs(df.loc[firstEventIDX,'EVENTTVALUE']), dof)) * 2.0
    df.loc[firstEventIDX,'EVENTSTOPIDX'] = df.loc[secondEventIDX,'EVENTSTOPIDX']
    print('merged Events {} and {} on VAG {}'.format(df.loc[firstEventIDX,'Unnamed: 0'], df.loc[secondEventIDX,'Unnamed: 0'],df.loc[secondEventIDX,'VAG']))
    df = df.drop([secondEventIDX])
    df.loc[df['EXPERIMENTNAME'] == df.loc[firstEventIDX,'EXPERIMENTNAME'],'NUMBEROFEVENTS'] =-1 
    return df.reset_index(drop = True)

def cleanupEvents(df,minSeparation = 20):
    '''
    Cleanup of events for noisy event trace in superlattice pindown data.
    Iterates over entire dataframe and identifies subsequent down events which are less than minSeparation apart.
    If event is found which is closer than it merges the event to one long one.
    '''
    for i in range(len(df.index)-1,1, -1):
        firstEventIDX = i-1
        secondEventIDX = i
        if ((df.loc[firstEventIDX,'EXPERIMENTNAME'] == df.loc[secondEventIDX,'EXPERIMENTNAME'])
            & (df.loc[secondEventIDX,'EVENTSTARTIDX'] < minSeparation + df.loc[firstEventIDX,'EVENTSTOPIDX'])
            & (df.loc[firstEventIDX,'EVENTTYPE'] == 'down') & (df.loc[secondEventIDX,'EVENTTYPE'] == 'down')
           ):
            df = mergeEvents(df, firstEventIDX, secondEventIDX)
    return df
def move_legend(ax, new_loc, **kws):
    old_legend = ax.legend_
    handles = old_legend.legendHandles
    labels = [t.get_text() for t in old_legend.get_texts()]
    title = old_legend.get_title().get_text()
    ax.legend(handles, labels, loc=new_loc, title=title, **kws)

def doubleExpLog(param,x):
    f = singleExpLog(param[[0,2]],x) + singleExpLog(param[[1,3]],x)
    return f

def singleExpLog(param,x):
    A = param[1]
    tau = param[0]
    return _singleExpLog(x,A,tau)

def dwell_fit_priors(dt, bins):
    nDt = len(dt)
    meanDt = np.nanmedian(dt)
    stdDt = np.nanstd(dt)
    res = scipy.optimize.minimize(prior_err,[nDt*0.4],args=(nDt, meanDt, (bins)))
    return meanDt, stdDt, res.x[0]

def prior_err(param, nDt, k, bins):
    err = np.abs(nDt - scipy.integrate.trapz(singleExpLog([k,param],bins)))
    return err

def _singleExpLog(x,A,tau): 
    return A * np.exp(np.log(x) - np.log(tau) - np.exp(np.log(x) - np.log(tau)))

def _logNormal(x, A, mu, sig):
    return (A / sig * (np.divide(np.exp( - np.square(np.log(x) - mu) / (2.0 * sig**2)),
                        x * np.sqrt(2.0 * np.pi))))

def _inverseGauss(x, A, mu, lam):
    '''https://en.wikipedia.org/wiki/Inverse_Gaussian_distribution'''
    return (
            A 
            * np.sqrt(lam / 2.0 / np.pi)
            * np.divide(np.exp( - np.divide(lam * np.square(x - mu),(2.0 * mu**2 * x))),
                         np.power(x, 3.0 / 2.0)) 
            
    )

def inverseGaussMaxPosition(mu, lam):
    '''https://www.wolframalpha.com/input/?i=d%2Fdx+B*%28sqrt%28A%2Fx%5E3%29+e%5E%28-%28A+%28x+-+%CE%BC%29%5E2%29%2F%282+%CE%BC%5E2+x%29%29%29%2Fsqrt%282+%CF%80%29+%3D0+solve+for+x'''
    return (np.sqrt(4 * lam**2 * mu**2 + 9 * mu**4) - 3 * mu**2) / 2.0 / lam

def inverseGaussSigma(mu, lam):
    return np.sqrt(mu**3 / lam)

def inverseGaussPriors(mean, sigma, counts):
    '''iterative to good priors'''
    deltaMu = 1.0
    mu = mean
    oldMu = 0
    lam = 4.0 * mean
    counter = 0
    while (np.abs(oldMu - mu) > 1e-9) & (counter < 1e5):
        oldMu = mu
        mu = np.sqrt(lam) * mean / np.sqrt(lam - 3.0 * mean)
        tmp = mu**3 / sigma**2
        if tmp > 3.0 * mean:
            lam = tmp
        counter+=1
    B = counts / _inverseGauss(mean, 1.0, mu, lam)
    return B, mu, lam


def logNormalMaxPosition(mu,sigma):
    '''https://www.wolframalpha.com/input/?i=log+normal+distribution'''
    return np.exp(mu - sigma**2)

def logNormalSigma(mu,sigma):
    '''https://www.wolframalpha.com/input/?i=log+normal+distribution'''
    return np.sqrt((np.exp(sigma**2) -1) * np.exp(2 * mu + sigma**2))

def fitDwellTimes(dt, LL = None, UL = None, numberOfBins = 80, fitType = 'exp', prefix = '', printOut = True, noLogNPlot = False, noCI = False, yscale = 1.0,**kwargs):
    '''
    fit dell time distributions in log space using lmfit library. Uses log space bins sizes!
    
    parameter:
    dt np.array float
        dwell times
        
    LL np.float
        lower limit for dwell times to fit
        
    UL np.float
        upper limit for dwell times to fit
        
    numberOfBins np.int
        number of bins for histogramm 
        
    fitType str
        different fit option
        default: will return single log-normal disttibution of dwell times.
        other options will give additional fits
        'exp' for exponentially distributed dwell times
        'lognormalexp' for log-normal and exponentially distributed dwell times https://en.wikipedia.org/wiki/Log-normal_distribution and https://www.sciencedirect.com/science/article/pii/S0006349587832988?via%3Dihub
        'inverseGaussian' for 2 inverse Gaussian distributions: https://en.wikipedia.org/wiki/Inverse_Gaussian_distribution
        
    returns best_fit model parameter
    '''
    if LL is None:
        LL = np.nanmin(dt) * 0.5
    if UL is None:
        UL = np.nanmax(dt) * 1.5
    count, binse = np.histogram(dt,bins=np.logspace(np.log(LL), np.log(UL), numberOfBins, base=math.e))#,density=True)
    bins = np.mean(np.vstack([binse[0:-1],binse[1:]]), axis=0)
    x = np.exp(np.arange(np.log(LL), np.log(UL), np.abs(np.log(LL) - np.log(UL))/4.0/numberOfBins))
    
    meanDt, stdDt, countsDt = dwell_fit_priors(dt, bins)
    countsDt = np.max(count)
    if printOut:
        print('\nFitting dwell histogram with mean dwell time {:.3e}s total counts {:.1f}\n'.format(meanDt,np.sum(count)))
        print('\nFitting dwell histogram with mean dwell time {:.3e}s normalized counts {:.3e}\n'.format(meanDt,np.sum(count / yscale)))
#### exponential fit for paperreview    
#     modLogNormal = lmfit.models.LognormalModel(prefix = prefix)
#     parsLogNormal = modLogNormal.guess(count, x = bins)
#     parsLogNormal[prefix + 'center'].set(value = np.log(meanDt), min = np.abs(np.log(LL)), max = np.abs(np.log(UL)))
#     parsLogNormal[prefix + 'sigma'].set(value = 1, min = 0, max = 1*1e3)
        
#     outLogNormal = modLogNormal.fit(count, parsLogNormal, x = bins)
#     if printOut:
#         print(outLogNormal.fit_report(min_correl=0.25))
    
    modLogNormal = lmfit.models.Model(_singleExpLog, nan_policy="omit", prefix = prefix)
    parsLogNormal = modLogNormal.make_params(A=countsDt, tau = meanDt)
    parsLogNormal[prefix + "A"].set(value = countsDt, min = 0, max = countsDt * 10.0 + 1)
    parsLogNormal[prefix + "tau"].set(value = meanDt*10, min = LL, max = UL)
# END NEW
    outLogNormal = modLogNormal.fit(count, parsLogNormal, x = bins)
    if printOut:
        print(outLogNormal.fit_report(min_correl=0.25))
    
    
    print('/n/t/t{}/n'.format(outLogNormal.residual[0]))
    
    
    pfix1 = ''
    if fitType == 'exp':
        mod = lmfit.models.Model(_singleExpLog, nan_policy="omit", prefix = prefix)
        pars = mod.make_params(A=countsDt, tau = meanDt)
        pars[prefix + "A"].set(value = countsDt, min = 0, max = countsDt * 10.0 + 1)
        pars[prefix + "tau"].set(value = meanDt*10, min = LL, max = UL)
        out = mod.fit(count, pars, x = bins)
    elif fitType == 'lognormalexp':
        pfix1 = prefix + 'M1_'
        mod1 = lmfit.models.LognormalModel(nan_policy="omit", prefix = pfix1)
        pars = mod1.guess(count, x = bins)
        pars[pfix1 + 'center'].set(value = outLogNormal.best_values['center'], min = -np.abs(outLogNormal.best_values['center']*1e2), max = np.abs(outLogNormal.best_values['center']*1e2))
        pars[pfix1 + 'sigma'].set(value = outLogNormal.best_values['sigma'], min = 0, max = outLogNormal.best_values['sigma']*1e3)
        pfix2 = prefix + 'M2_'
        mod2 = lmfit.models.Model(_singleExpLog, nan_policy="omit", prefix = pfix2)
        pars.update(mod2.make_params(A=countsDt, tau = meanDt))
        pars[pfix2 + "A"].set(value = countsDt * 1e-1, min = 0, max = countsDt * 10.0 + 1)
        pars[pfix2 + "tau"].set(value = meanDt * 1e2, min = meanDt * 1e-8, max = meanDt * 1e4)
        mod = mod1 + mod2
        out = mod.fit(count, pars, x = bins)    
    elif fitType == 'expexp':
        pfix1 = prefix + 'M1_'
        mod1 = lmfit.models.Model(_singleExpLog, nan_policy="omit", prefix = pfix1)
        pars = mod1.make_params(A=countsDt, tau = meanDt)
        pars[pfix1 + 'A'].set(value = countsDt * 0.1, min = 0, max = countsDt * 10.0 + 1)
        pars[pfix1 + 'tau'].set(value = np.exp(0.5 * np.log(meanDt) + 0.5 * np.log(LL)), min = LL, max = UL)
        pfix2 = prefix + 'M2_'
        mod2 = lmfit.models.Model(_singleExpLog, nan_policy="omit", prefix = pfix2)
        pars.update(mod2.make_params(A=countsDt, tau = meanDt))
        pars[pfix2 + "A"].set(value = countsDt * 0.5, min = 0, max = countsDt * 10.0 + 1)
        pars.add(name = pfix2 + 'tauSplit', value = np.exp(0.5 * np.log(meanDt) + 0.5 * np.log(UL)), min = meanDt, max = UL-LL, vary=True)
        pars[pfix2 + "tau"].set(expr = pfix2 + 'tauSplit+' + pfix1 + 'tau')
        mod = mod1 + mod2
        out = mod.fit(count, pars, x = bins)
    elif fitType == 'normal':
        mod = lmfit.models.normalModel(prefix = prefix)
        pars = mod.guess(count, x = bins)
        out = mod.fit(count, pars, x = bins)
    elif fitType == 'inverseGaussian':
    # inverse gaussian dist:
        pfix1 = prefix + 'M1_'
        mod = lmfit.models.Model(_inverseGauss, nan_policy="omit", prefix = pfix1)
        APrior, muPrior, lamPrior = inverseGaussPriors(meanDt, stdDt, countsDt/2.0)
        pars = mod.make_params(A = APrior, mu = APrior, lam = lamPrior)
        pars[pfix1 + "A"].set(value = APrior, min = 0, max = APrior * 1e3 + 1)
        pars[pfix1 + "mu"].set(value = muPrior, min = muPrior * 1e-7, max = muPrior * 1e3)
        pars[pfix1 + "lam"].set(value = lamPrior, min = lamPrior * 1e-7, max = lamPrior * 1e3)    
        pfix2 = prefix +'M2_'
        mod2 = lmfit.models.Model(_inverseGauss, nan_policy="omit", prefix = pfix2)
        APrior, muPrior, lamPrior = inverseGaussPriors(meanDt * 1e-2, stdDt * 1e-1, countsDt/2.0)
        pars.update(mod2.make_params(A = APrior, mu = muPrior, lam = lamPrior))
        pars[pfix2 + "A"].set(value = APrior, min = 0, max = APrior * 1e3 + 1)
        pars[pfix2 + "mu"].set(value = muPrior, min = muPrior * 1e-7, max = muPrior * 1e3)
        pars[pfix2 + "lam"].set(value = lamPrior, min = lamPrior * 1e-7, max = lamPrior * 1e3)   
        mod = mod + mod2
        out = mod.fit(count, pars, x = bins)
    
#### calculate F statistics for paperreview
    RSSexp = np.sum(outLogNormal.residual**2)
    RSSexpexp = np.sum(out.residual**2)
    FStatistics = (RSSexp - RSSexpexp) / RSSexpexp / (((np.sum(count) - outLogNormal.nvarys) - (np.sum(count) - out.nvarys)) / (np.sum(count) - out.nvarys))
    PValue = 1 - stats.f.cdf(FStatistics, ((np.sum(count) - outLogNormal.nvarys) - (np.sum(count) - out.nvarys)), (np.sum(count) - out.nvarys))
    FCritical = stats.f.ppf(1-0.05, ((np.sum(count) - outLogNormal.nvarys) - (np.sum(count) - out.nvarys)), (np.sum(count) - out.nvarys))
    print('\n F stats expexp vs. exp dist fit: {:.4e} and the corresponing PValue: {:.4e}'.format(FStatistics, PValue))
# END NEW

    if 'out' in locals():
        if out.errorbars and not noCI:
            ci = out.conf_interval(sigmas=[1])
            print(out.ci_report())
            print(out.ci_out)
        if not (pfix1 == ''):
            comps = out.eval_components(x = bins)
            out.best_values[pfix1 + 'counts'] = np.trapz(comps[pfix1])
            out.best_values[pfix2 + 'counts'] = np.trapz(comps[pfix2])
            #### calculate F statistics for paperreview
            out.best_values[pfix1 + 'Fstatistics'] = FStatistics
            out.best_values[pfix1 + 'RSSexp'] = RSSexp
            out.best_values[pfix1 + 'RSSexpexp'] = RSSexpexp
            out.best_values[pfix1 + 'DOFexp'] = (np.sum(count) - outLogNormal.nvarys)
            out.best_values[pfix1 + 'modelDOFexpexp'] = out.nfree
            out.best_values[pfix1 + 'modelNvarysexpexp'] = out.nvarys
            out.best_values[pfix1 + 'counts'] = np.sum(count)
            out.best_values[pfix1 + 'Pvalue'] = PValue
            out.best_values[pfix1 + 'DOF'] = (np.sum(count) - out.nvarys)
            out.best_values[pfix1 + 'Fcritical'] = FCritical
            out.best_values[pfix1 + 'CountsPerS'] = np.sum(count / yscale)
            # END NEW
        else:
            out.best_values[prefix + 'counts'] = np.trapz(out.best_fit)
            #### calculate F statistics for paperreview
            out.best_values[prefix + 'Fstatistics'] = FStatistics
            out.best_values[pfix1 + 'RSSexp'] = RSSexp
            out.best_values[pfix1 + 'RSSexpexp'] = RSSexpexp
            out.best_values[pfix1 + 'DOFexp'] = (np.sum(count) - outLogNormal.nfree)
            out.best_values[pfix1 + 'modelDOFexpexp'] = out.nfree
            out.best_values[pfix1 + 'modelNvarysexpexp'] = out.nvarys
            out.best_values[pfix1 + 'counts'] = np.sum(count)
            out.best_values[prefix + 'Pvalue'] = PValue
            out.best_values[prefix + 'DOF'] = (np.sum(count) - out.nfree)
            out.best_values[pfix1 + 'Fcritical'] = FCritical
            out.best_values[prefix + 'CountsPerS'] = np.sum(count / yscale)
            # END NEW
        
        for name, value in out.params.items():
            if 'tau' in name:
                out.best_values[name + '_stderr'] = value.stderr
                if out.errorbars and not noCI:
                    try:
                        out.best_values[name + '_ciLOW'] = out.ci_out[name][0][1]
                        out.best_values[name + '_ciHIGH'] = out.ci_out[name][2][1]
                    except:
                        out.best_values[name + '_ciLOW'] = np.nan
                        out.best_values[name + '_ciHIGH'] = np.nan
                else:
                    out.best_values[name + '_ciLOW'] = np.nan
                    out.best_values[name + '_ciHIGH'] = np.nan
                
    if printOut:   
        from datetime import datetime
        t = sns.color_palette("tab10",3)
        fig = plt.figure()
        if not noLogNPlot:
            t = sns.color_palette("tab10",4)
#### exponential fit for paperreview 
#             ax = sns.lineplot(x = bins, y = outLogNormal.best_fit, linestyle = '-', label = 'log normal', color = t[3], zorder = 2)
            ax = sns.lineplot(x = bins, y = outLogNormal.best_fit / yscale, linestyle = '-', label = 'single exp', color = t[3], zorder = 2)
# END NEW   
        if 'out' in locals():
            print(out.fit_report(min_correl=0.25))
    #         print(out.ci_report())
            if noLogNPlot:
                ax = sns.lineplot(x = bins, y = out.best_fit / yscale, linestyle = '-', label = fitType, color = t[1], zorder = 3, alpha = 0.6)
            else:
                plt.plot(bins, out.best_fit / yscale, '-',label=fitType, color = t[1])
            if not (pfix1 == ''):
                plt.plot(bins, comps[pfix1] / yscale, linestyle = (0,(1,2)), label=fitType + '_comp_1',color=t[1], zorder = 5, alpha = 1.0)
                plt.plot(bins, comps[pfix2] / yscale, linestyle = (0,(5,7)), label=fitType + '_comp_2',color=t[1], zorder = 4, alpha = 0.8)   
        ax.bar(bins, count / yscale, width = np.diff(binse), alpha = 0.3, color = t[0], zorder = 1)
        ax.set_xlim([LL,UL])
        ax.set_xscale("log")
        plt.xlabel('dwell time [s]')
        plt.ylabel('counts normalized')
        plt.legend()
        move_legend(ax,"center left", bbox_to_anchor=(1.1,0.5), frameon=False)
#         plt.savefig('figure-2_panel-b-{:}_v0.eps'.format(datetime.now()), dpi = 600, bbox_inches = "tight")
        plt.show()
    
# #### exponential fit for paperreview 
# #         print('\nlog normal dist fit: \t\t tau_s = {:.3e}s, sigma = {:.3e}, counts_s {:.1f}'.format(
# #                     logNormalMaxPosition(outLogNormal.best_values[prefix +'center'], outLogNormal.best_values[prefix +'sigma']),
# #                     logNormalSigma(outLogNormal.best_values[prefix +'center'], outLogNormal.best_values[prefix +'sigma']),
# #                     np.trapz(outLogNormal.best_fit)))
        
#         print('\nexp dist fit: \t \t tau__exp = {:.3e}s, counts__exp {:.1f}'.format(
#                     outLogNormal.best_values[prefix +'tau'],
#                     outLogNormal.best_values[prefix + 'A']))
#         RSSexp = np.sum(outLogNormal.residual**2)
#         RSSexpexp = np.sum(out.residual**2)
#         FStatistics = (RSSexp - RSSexpexp) / RSSexpexp / (((np.sum(count) - out.nfree) - (np.sum(count) - outLogNormal.nfree)) / (np.sum(count) - out.nfree))
#         PValue = 1 - stats.f.cdf(FStatistics, ((np.sum(count) - out.nfree) - (np.sum(count) - outLogNormal.nfree)), (np.sum(count) - out.nfree))
# #         FStatistics = (RSSexp - RSSexpexp) / RSSexpexp / ((outLogNormal.nfree - out.nfree) / out.nfree)
# #         FStatistics = (outLogNormal.chisqr - out.chisqr) / out.chisqr / ((outLogNormal.nfree - out.nfree) / out.nfree)
#         print('\n F stats expexp vs. exp dist fit: {:.4e} and the corresponing PValue: {:.4e}'.format(FStatistics, PValue))
# # END NEW

        if 'out' in locals():    
            if fitType == 'inverseGaussian':
                print('\ninverseGaussian dist fit dist1: \t tau_1 = {:.3e}s, sigma_1 = {:.3e}, counts_1 {:.1f}'.format(
                    inverseGaussMaxPosition(out.best_values[pfix1 + 'mu'], out.best_values[pfix1 + 'lam']),
                    inverseGaussSigma(out.best_values[pfix1 + 'mu'], out.best_values[pfix1 + 'lam']), out.best_values[pfix1 + 'counts']))
                print('inverseGaussian dist fit dist2: \t tau_1 = {:.3e}s, sigma_2 = {:.3e}, counts_2 {:.1f}'.format(
                    inverseGaussMaxPosition(out.best_values[pfix2 + 'mu'], out.best_values[pfix2 + 'lam']),
                    inverseGaussSigma(out.best_values[pfix2 + 'mu'], out.best_values[pfix2 + 'lam']),out.best_values[pfix1 + 'counts']))
            elif fitType == 'lognormalexp':
                print('\nlog-normal with Exp dist fit: \t tau_log = {:.3e}s, sigma_log = {:.3e}, counts_log {:.1f}'.format(
                    logNormalMaxPosition(out.best_values[pfix1 + 'center'], out.best_values[pfix1 + 'sigma']),
                    logNormalSigma(out.best_values[pfix1 + 'center'], out.best_values[pfix1 + 'sigma']),
                    out.best_values[pfix1 + 'counts']))
                print('\t\t\t \t tau__exp = {:.3e}s, counts__exp {:.1f}'.format(
                    out.best_values[pfix2 + 'tau'],
                    out.best_values[pfix2 + 'counts']))
            elif fitType == 'exp':
                print('\nexp dist fit: \t \t tau__exp = {:.3e}s, counts__exp {:.1f}'.format(
                    out.best_values[prefix +'tau'],
                    out.best_values[prefix + 'counts']))
            elif fitType == 'expexp':
                print('\nexp dist fit dist 1: \t \t tau__exp = {:.3e}s, counts__exp {:.1f}'.format(
                    out.best_values[pfix1 + 'tau'],
                    out.best_values[pfix1 + 'counts']))
                print('\nexp dist fit dist 2: \t \t tau__exp = {:.3e}s, counts__exp {:.1f}'.format(
                    out.best_values[pfix2 + 'tau'],
                    out.best_values[pfix2 + 'counts']))
    
    if 'out' in locals():           
        return out.best_values, outLogNormal.best_values, fig
    else:
        return outLogNormal.best_values, {'other' : None}

def exponentialDecay(x, A, exp1_decay, x0, offsetP):
    return A * np.exp(-((x - x0) / exp1_decay)) + offsetP

def lmFIT_exponentialDecay(x, y, guess, startIdx = 20, backCut = 0):
    """
    implementation of exponential distribution fitting function with lmfit module

    Parameters
    ----------
    x : np.array
        min to max occuring dwell time bins in [s]
    y : np.array
        number of events with at specific dwell time bin

    Returns
    -------
    out : lmfit object

    """    
    mod = Model(exponentialDecay, nan_policy="raise")
    pars = mod.make_params(offsetP=np.mean(y), A=np.mean(y[startIdx:-backCut]),exp1_decay =guess,x0=x[np.argmax(np.abs(y))])
#     if down:
    pars["offsetP"].set(value = np.mean(y[startIdx:-backCut]), 
                        min = np.min(y[startIdx:-backCut]) - 2.0*np.abs(np.mean(y[startIdx:-backCut])), 
                        max = np.max(y[startIdx:-backCut]) + 2.0*np.abs(np.mean(y[startIdx:-backCut])))

    pars["A"].set(value = y[startIdx], 
                  min = np.min(y[startIdx:-backCut]) - 2.0*np.abs(np.mean(y[startIdx:-backCut])), 
                  max = np.max(y[startIdx:-backCut]) + 2.0*np.abs(np.mean(y[startIdx:-backCut])))

    pars["exp1_decay"].set(value=guess, min = 0.05 * guess, max = 100 * guess)

    pars["x0"].set(value=x[np.argmax(np.abs(y))], min = x[1], max = x[-1] / 2)
#     else:

    out = mod.fit(y[startIdx:-backCut], pars, x=x[startIdx:-backCut])
    print(out.best_values)
    plt.plot(x[startIdx:-backCut], out.best_fit, '-',linewidth = 0.7,alpha = 0.7)  
    return out

def read_event_data(files_for_frame):
    DFlist=[]
    for fileN in files_for_frame:
        DFlist.append(pd.read_csv(fileN))
        try:
            mult = 1.0
            print(fileN.stem)
            group = re.search("(\d*)([n|p])(M)([A-z]*)(-*|_)(T|BS)([-]*|_)", str(fileN.stem))
            if 'p' in group[2]:
                mult = 1e-3
            speciesType = group[4]
            print(group)
            conc = np.round(np.float32(group[1]) * mult,decimals=1)
        except Exception as e:         
            conc = None
            speciesType = None
        try:
            group = re.search("(_)(-*\d*)(mV_events)", str(fileN.stem))
            VAC = np.int16(group[2])
        except Exception as e:         
            VAC = 0
        DFlist[-1]['TYPE']=speciesType
        DFlist[-1]['CONC']=conc
        DFlist[-1]['VAC']=VAC
        DFlist[-1]['exp']=str(fileN.stem)
        DFlist[-1]['UP'] = 1
        DFlist[-1]['STATEDWELLTIME']=DFlist[-1]['EVENTDWELLTIME']
        tmpDF = DFlist[-1][['EVENTTRIGGERVALUE','EVENTTYPE','EVENTSTARTIDX','EVENTSTOPIDX','EVENTDEPTH','VAC','exp','CONC']]
#         tmpDF.rename(columns={'EVENTSTARTIDX': 'EVENTSTOPIDX', "B": "c"},inplace = True)
        tmpDF['STATEDWELLTIME'] = (DFlist[-1]['EVENTSTOPIDX'] - DFlist[-1].shift(1)['EVENTSTARTIDX'])*1e6/DFlist[-1].loc[0,'SAMPLERATE']
        tmpDF['UP'] = 0
        tmpDF['EVENTTYPE'] = 'BASELINE'
#         display(tmpDF.info())
        x = len(DFlist[-1])
        DFlist[-1] = DFlist[-1].append(tmpDF, ignore_index=True)
    hmmeventsDF = pd.concat(DFlist, ignore_index = True)
    return hmmeventsDF
 
def get_event_data(files_for_frame):
    hmmeventsDF = read_event_data(files_for_frame)

    grouper=hmmeventsDF[hmmeventsDF['UP'] == 1].groupby([(hmmeventsDF['UP'] != 1).cumsum(),'exp'])
    dt=grouper.agg({'STATEDWELLTIME' : ['sum'],'VAC' : ['mean']})
    dt['STATEDWELLTIME']=dt['STATEDWELLTIME']/1e6
    dt=dt.droplevel(1, axis=1) 
    dt =dt[dt['STATEDWELLTIME']!=0]
    dt.reset_index(inplace=True, level = ['exp'])
    
    #down
    grouper=hmmeventsDF[hmmeventsDF['UP'] == 0].groupby([(hmmeventsDF['UP'] != 0).cumsum(),'exp'])
    dtdown=grouper.agg({'STATEDWELLTIME' : ['sum'], 'VAC' : ['mean']})
    dtdown['STATEDWELLTIME']=dtdown['STATEDWELLTIME']/1e6
    dtdown=dtdown.droplevel(1, axis=1) 
    dtdown =dtdown[dtdown['STATEDWELLTIME']!=0]
    dtdown.reset_index(inplace=True, level = ['exp'])

    display(HTML('UP state total time:'))
    display(dt['STATEDWELLTIME'].sum())
    display(HTML('DOWN state total time:'))
    display(dtdown['STATEDWELLTIME'].sum())
    display(HTML('total time:'))
    display(dtdown['STATEDWELLTIME'].sum() + dt['STATEDWELLTIME'].sum())
    
    return dt, dtdown

def calc_PUP_PDOWN(dt, dtdown):
    df = pd.DataFrame(columns=['T_UP'])
    df = dt.groupby(['VAC','exp']).agg({'STATEDWELLTIME' : ['sum']})
    df.droplevel(1, axis=1)
    df.columns = ['T_UP']
    df['T_DOWN'] = dtdown.groupby(['VAC','exp']).agg({'STATEDWELLTIME' : ['sum']})
    df.fillna(0.0, inplace = True)
    df['T_TOTAL'] = df['T_UP'] + df['T_DOWN']
    df['P_UP'] = df['T_UP'] / df['T_TOTAL']
    df['P_DOWN'] = df['T_DOWN'] / df['T_TOTAL']
    df.reset_index(inplace=True, level = ['exp'])
    return df

def cm2inch(*tupl):
    inch = 2.54
    if isinstance(tupl[0], tuple):
        return tuple(i/inch for i in tupl[0])
    else:
        return tuple(i/inch for i in tupl)
