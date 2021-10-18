'''
A wrapup and extention for the linear_model library of sklearn
The functionality includes:
- automatic selection of optimal regularising parameter Alpha for 
  Ridge and Lasso optimization schemes
- computes scores for test and train datasets

Created on 22.3.2020

@author: sofievm
'''
import matplotlib
from _ast import Or
matplotlib.use('Agg')
import copy, os, sys, io
import bisect  # random
import numpy as np
import datetime as dt
from zipfile import ZipFile
from sklearn import linear_model
import sklearn
from scipy import optimize
from toolbox import supplementary as spp , MyTimeVars
from matplotlib import pyplot as plt

minimum_regularization_alpha = 1e-5

# Quality flags
#
flg_unknown = 'unknown'
flg_OK = 'OK'
flg_bad = 'input integrity or other fatal error'
flg_fewOutVarData = 'too few outvars data: %i'
flg_noOutVarData = 'no outvar data'
flg_oldOutVarData = 'too old outvar data, hole %g days'
flg_noPredictors = 'no valid predictors'
flg_fewPredictrs4days = 'too few predictrs: %i'
flg_NaNinPredictors = 'NaN in transformed forecasted predictors'
#
# Species that we can analyse
#
log_predictors = 'ald bir gra mug rag oli'.split()
lin_predictors = 'SO2 NO2 O3 PM2_5 t2m q2m bir_max gra_max ald_max mug_max oli_max rag_max SO2_max NO2_max O3_max PM2_5_max t2m_max q2m_max'.split() # w10 w10_max'.split()


################################################################

def split_times(times, sample_weights, training_fraction,  # [0..1] 
                subset_selection, randomGen, log, obsdata=None):  # native_rnd, daily_rnd, daily_first_test, daily_last_test
        #
        # There are several ways to split the dataset: split can be with native resolution, 
        # daily or station-wise (not yet implemented). It can be random or earlier/later times 
        # can be forced to be trainign  Accidentally, one of the subsets can be emoty. 
        # Let's try 10 times... 
        #
        nTimes = len(times)
        ifSelectionDone = False
        #
        nUsefulTimes = np.sum(sample_weights[np.isfinite(sample_weights)] > 0.0)
        # If sample_weight filters too few items, make sure of non-empty training
        if nUsefulTimes < 2:
            log.log('Too few useful times: %g, due to sample_weights' % nUsefulTimes)
            log.log(np.sum(sample_weights))
            raise ValueError
        elif nUsefulTimes == 2:
            log.log('SPLITTING WARNING. Only two useful times in sample_weights')
            train_indices = [np.argmax(sample_weights)]
            ifSelectionDone = True
        else:
            # In case of sufficient selection possibilities, proceed usual way
            ifSelectionDone = False
            for iTry in range(10):
                if subset_selection == 'native_random':
                    # Hourly, daily, ... whatever time step
                    train_indices = sorted(randomGen.choice(range(nTimes), 
                                                            max(1, int(nTimes * training_fraction))))
                elif subset_selection.startswith('daily'):
                    # If native is hourly we might wish days
                    days_lst = list( (dt.datetime(t.year,t.month,t.day) 
                                      for t in (np.array(times) - spp.one_minute)) )
                    days_set = np.array(sorted(list(set(days_lst))))
                    nDays = days_set.shape[0]    # hours have been dropped. days are repetitive
                    if nDays < 2 : 
                        log.log('Too short time period for day-level time split: %g days' % nDays)
                        raise ValueError
                    # shall we pick random days, the first or the last ones?  
                    if subset_selection == 'daily_random':
                        train_days_idx = sorted(randomGen.sample(range(nDays), 
                                                                 max(1, int(nDays * training_fraction))))
                    elif subset_selection == 'daily_first_test':
                        train_days_idx = range(min(nDays-1, 
                                                   np.ceiling(nDays * (1.- training_fraction)), nDays))
                    elif subset_selection == 'daily_last_test':
                        train_days_idx = range(max(1, int(nDays * training_fraction)))
                    else:
                        log.log('Unknown daily training-test split:' + subset_selection)
                        raise ValueError
                    # ensure existence of at least one training and test day
                    if ((len(train_days_idx) - nDays) * len(train_days_idx) == 0 and 
                        (1.- training_fraction) * training_fraction > 0): 
                        log.log('Zero lengths')
                        continue    # failed selection
                    # Turn training days to indices of training native times
                    days_train = days_set[train_days_idx]  # the subset of training days, sorted
                    TrainingSubset = np.searchsorted(days_train, days_lst) < len(days_train)
                    train_indices = np.array(range(nTimes))[TrainingSubset]
                else:
                    log.log('Unknown non-daily training-test split:' + subset_selection)
                    raise ValueError
                #
                # Test is the leftover
                test_indices = sorted(list(set(range(nTimes)) - set(train_indices)))
                # Managed?
                # sample_weights must be reasonable (they are probably nan for obsdata nan)
                # 
                if np.sum(sample_weights[test_indices][np.isfinite(sample_weights[test_indices])] > 0) == 0:
                    log.log('Zero test indices N_finite_weights = %g' % 
                            np.sum(np.isfinite(sample_weights[test_indices]))) 
                    continue
                if np.sum(sample_weights[train_indices][np.isfinite(sample_weights[train_indices])] > 0) == 0: 
                    log.log('Zero train indices. N_finite_weights = %g' %
                            np.isfinite(sample_weights[train_indices])) 
                    continue
                
                if obsdata is None:
                    ifSelectionDone = not ((len(train_indices) - nTimes) * len(train_indices) == 0 or 
                                           (1.- training_fraction) * training_fraction == 0 or
                                           np.sum(sample_weights[train_indices]
                                                  [np.isfinite(sample_weights[train_indices])]) < 1e-5 or 
                                           np.sum(sample_weights[test_indices]
                                                  [np.isfinite(sample_weights[test_indices])]) < 1e-5)
                else:
                    ifSelectionDone = np.all(np.nanstd(sample_weights[train_indices] * 
                                                       obsdata[train_indices])  > 
                                             1e-6 * np.nanmean(sample_weights[train_indices] * 
                                                               obsdata[train_indices]))
                if ifSelectionDone: break # good selection, stop trying
                #
                # There are subset_selection types, which are not really random. Nothing to try then 
                if subset_selection == 'daily_last_test' or subset_selection == 'daily_first_test': break
                    
        # Success?
        if not ifSelectionDone:
            if subset_selection == 'daily_last_test' or subset_selection == 'daily_first_test': 
                log.log('Failed training / test split for ' + subset_selection)
                log.log('nTimes = %g, training fraction = %g' % (nTimes, training_fraction))
                log.log('Training indices:' + ' '.join(list( (str(v) for v in train_indices) )))
                if obsdata is None:
                    log.log('None obsdata, sample_weights_test:' + 
                            ' '.join(list(('%g' % w for w in sample_weights[test_indices]))))
                    log.log('Zero-len train-test:' + 
                            str(((len(train_indices) - nTimes) * len(train_indices) == 0)))
                    log.log('Sum of train weights %g' % np.nansum(sample_weights[train_indices]))
                    log.log('Sum of test sample_weights_test = %g' % 
                            np.nansum(sample_weights[test_indices]))
                else:
                    log.log('obsdata available')
                    log.log('std(obs*weight), training %g = ' % 
                            np.nanstd(sample_weights[train_indices] * obsdata[train_indices]))
                    log.log('Mean of obs*weight, training %g' % 
                            np.nanmean(sample_weights[train_indices] * obsdata[train_indices]))
                    log.log('std(obs*weight), test %g = ' % 
                            np.nanstd(sample_weights[test_indices] * obsdata[test_indices]))
                    log.log('Mean of obs*weight, test %g' % 
                            np.nanmean(sample_weights[test_indices] * obsdata[test_indices]))
                return((None, None))
            else:
                log.log('Failed 10 tries of picking the training indices')
                log.log('nTimes = %g, training fraction = %g' % (nTimes, training_fraction))
                log.log('Training indices:' + ' '.join(list( (str(v) for v in train_indices) )))
                raise ValueError

        return(train_indices, test_indices)
        

####################################################################################

def ortogonalise_TSM(poolMdl, idxSort, kernel, tc_indices_mdl, log):
    #
    # Creates an orthogonal set of "models" of the same shape as the given pool
    # ATTENTION.
    # Note the order of dimensions variable-time-station, as in tsMatrices
    # Procedure: 
    # 1. Make linear regression of unity and zero-mean best model to He second-best model
    #    store residuals of regression as the added-value of this model
    # 2..N. Each next model is regressed with processed models leaving residual as added value
    # 3. The collection of residuals is stored as the new uncorrelated set of predictors.
    #
    poolMdlOrt= np.ones(shape=poolMdl.shape, dtype=np.float32) * np.nan    # will be output
    if tc_indices_mdl is None: tc_indices = range(poolMdl.shape[0])
    else: tc_indices = tc_indices_mdl
    #
    # Store the first model: zero-mean best model
    # The corresponding fitting model is just the intercept
    clf = [poolMdl[idxSort[0],:,:].mean()]
    poolMdlOrt[idxSort[0],:,:] = poolMdl[idxSort[0],:,:] - clf[0]
    #
    # Every next model is turned into its residuals from regression with all previous models
    for iOrder in range(1, len(idxSort)):
        clf.append(linear_model.LinearRegression())
        x = np.array(list((poolMdl[idxSort[iOr],tc_indices,:].ravel() 
                           for iOr in range(iOrder))))
        x[np.abs(x) < 1.e-15] = 0.0    # avoid underflow for x**2 in clf.fit
        xFull = np.array(list((poolMdl[idxSort[iOr],:,:].ravel() 
                               for iOr in range(iOrder))))
#        print('ortogonalise_TSM: ', iOrder, np.max(x), np.max(poolMdl[tc_indices,:,idxSort[iOrder]]),
#              np.max(kernel))
        # Here the regression is made without sample weight: we are in model space, no
        # observational issues have any say here
        clf[-1].fit(x.T, poolMdl[idxSort[iOrder],tc_indices,:].ravel())  #, kernel.ravel())
        yPr = clf[-1].predict(xFull.T).reshape((poolMdl.shape[1], poolMdl.shape[2]))   # regression to previous models
        poolMdlOrt[idxSort[iOrder],:,:] = poolMdl[idxSort[iOrder],:,:] - yPr # added value

#        self.log.log('Verifying orthogonalization')
    v = np.zeros(shape=(len(idxSort), len(idxSort)))
    for iO1 in range(len(idxSort)):
        for iO2 in range(len(idxSort)):
            v[iO1,iO2] = np.sqrt(np.abs(np.mean(poolMdlOrt[idxSort[iO1],tc_indices,:] * 
                                                poolMdlOrt[idxSort[iO2],tc_indices,:] * 
                                                kernel)))
        if v[iO1,iO1] * 1.01 < np.sum(v[iO1,:]):
            log.log('Problematic orthogonaliozation, line %i: diagonal / sum = %g' % 
                  (iO1, v[iO1,iO1] / np.sum(v[iO1,:])))
#            if True:
            log.log(' '.join('%5g' % vv for vv in v[iO1]))

    # Return the result of ortogonalization, rules to obtain it and its quality
    return (poolMdlOrt, clf, v)


####################################################################################

def apply_ortogonalization_TSM(poolMdl, clf, idxSort, log):
    #
    # Creates an orthogonal set of "models" of the same shape as the given pool
    # ATTENTION.
    # Note the order of dimensions variable-time-station, as in tsMatrices
    # Procedure: 
    # 1. Make linear regression of unity and zero-mean best model to He second-best model
    #    store residuals of regression as the added-value of this model
    # 2..N. Each next model is regressed with processed models leaving residual as added value
    # 3. The collection of residuals is stored as the new uncorrelated set of predictors.
    #
    poolMdlOrt= np.ones(shape=poolMdl.shape, dtype=np.float32) * np.nan    # will be output
    #
    # Store the first model: zero-mean best model
    # The corresponding fitting model is just the intercept
    poolMdlOrt[idxSort[0],:,:] = poolMdl[idxSort[0],:,:] - clf[0]
    #
    # Every next model is turned into its residuals from regression with all previous models
    for iOrder in range(1, len(idxSort)):
        x = np.array(list((poolMdl[idxSort[iOr],:,:].ravel() for iOr in range(iOrder))))
        xFull = np.array(list((poolMdl[idxSort[iOr],:,:].ravel() for iOr in range(iOrder))))
        yPr = clf[iOrder].predict(xFull.T).reshape((poolMdl.shape[1], poolMdl.shape[2]))   # regression to previous models
        poolMdlOrt[idxSort[iOrder],:,:] = poolMdl[idxSort[iOrder],:,:] - yPr # added value

    return poolMdlOrt

    
#######################################################################################

def basic_rmse4min(arCoefs, alpha_Tikhonov, C0_Tikhonov, alpha_low_pass, C_prev,
                   arObs, arMdl, metric_kernel, ifVerify=False):
    #
    # Generic function implementing the cost function for RMSE minimization with a metric kernel
    # The metric, in essence, tells the input of every SE term into the overall sum RMSE
    # The terms can originate from covariance matrices or simply from generial sensitivity or 
    # importance considerations. Since they all look same here, I require them as input.
    # Two regularising terms are included:
    # - Tikhonov-type term alpha_Tikhonov * sum_i(A_m  - A_m_0)**2
    # - low-pass filter term alpha_low_pass * sum_m(A_m(t) - A_m_(t-1))**2
    # The RMSE itself, of course, 
    # y_pred = a_0 + sum_i(A_m * C_m)
    # RMSE = mean_i( (y_i - y_pred_i)**2 * kernel_i )
    # Here, m is for models, i for stations and times, kernel_i is the weight of each term
    # in the RMSE, the very RMSE metric
    #
    prediction = arCoefs[0] + np.sum(np.array([arMdl[iMdl,:] * arCoefs[iMdl+1] 
                                               for iMdl in range(arMdl.shape[0])]), axis=0)
    
    RMSE = np.sqrt( (np.square(prediction - arObs) * metric_kernel).mean() )
    reg_Tikhonov = alpha_Tikhonov * np.square(arCoefs[1:] - C0_Tikhonov).sum()  # intercept does not affect 
    reg_low_pass = alpha_low_pass * np.square(arCoefs[:] - C_prev).sum()  # all coefficients here
    
    if ifVerify:
        return (RMSE, reg_Tikhonov, reg_low_pass)     # costs separately
    else:
        return RMSE + reg_Tikhonov + reg_low_pass     # total cost


################################################################

def alpha_linreg_corr4min(arAlpha, chMdl, 
                          x_train, y_train, krnl_train, 
                          x_test, y_test, krnl_test, ifVerify, log): 
    #
    # function for minimization for linear model, Ridge and Lasso. 
    # Takes the given alpha, fits the model requested and return skills for test subset 
    # ATTENTION.
    # It returns MINUS correlation because scipy has only minimizers as algorithms
    #
    if chMdl == 'RIDGE': 
        clf = linear_model.Ridge(alpha=arAlpha[0], tol=2e-4, max_iter=1000000)
    elif chMdl == 'LASSO': 
        clf = linear_model.Lasso(alpha=arAlpha[0], tol=2e-4, max_iter=1000000)
    else:
        log.log('alpha_linreg_corr4min. Do not support the model:  %s, only RIDGE and LASSO so-far' % 
                chMdl)
        return np.nan
    # make the fit
    try:
        clf.fit(x_train, y_train, sample_weight=krnl_train)   # x_train, y_train
    except:
        if ifVerify:
            log.error('Failed fit %s. x_train[:200], y_train[:200]: \n' % chMdl + 
                      ' '.join(('%g' % x for x in x_train.ravel()[:200])) + '\n' + 
                      ' '.join(('%g' % y for y in y_train.ravel()[:200])))
            return (-1., arAlpha[0] * np.sum(np.square(clf.coef_)))
        else: return -1
    # compute the quality with the test subset
    y_test_predicted = clf.predict(x_test)
    if np.std(y_test_predicted) < 1e-5 * np.mean(y_test_predicted):
        if ifVerify:
            print('y_test_predicted', y_test_predicted)
            print('y_test', y_test)
        if np.std(y_test) < 1e-5 * np.mean(y_test):
            if ifVerify: 
                log.log('All-zero observations. Void perfect corr. HOW COMES ?')
                return (0., arAlpha[0] * np.sum(np.square(clf.coef_)))
            else: return 0.
        else:
            if ifVerify: 
                log.log('Predicted constant observations. Void null corr')
                return (0., arAlpha[0] * np.sum(np.square(clf.coef_)))
            else: return 0.
    elif np.std(y_test) < 1e-5 * np.mean(y_test):
        if ifVerify: 
            log.log('Reported constant observations. Void null corr. HOW COMES?')
            print('y_test_predicted', y_test_predicted)
            print('y_test', y_test)
            return (0., arAlpha[0] * np.sum(np.square(clf.coef_)))
        else: return 0.
    else:
        return  (- spp.weightedCorrCoef(y_test_predicted, y_test, krnl_test),
                 arAlpha[0] * np.sum(np.square(clf.coef_)))
#        corr_matrix_test = np.corrcoef(y_test_predicted, y_test, rowvar=False)  #x_train, y_train, x_test, y_test)
##        print('corr_matrix_test[0,1], self.alpha', corr_matrix_test[0,1], arAlpha)
#        if np.isnan(corr_matrix_test[0,1]):
#            print('y_test_predicted', y_test_predicted)
#            print('y_test', y_test)
#            print('Correlation matrix:', corr_matrix_test)
#        if ifVerify:
#            return(-corr_matrix_test[0,1], np.nan)  # cannot compare
#        else:
#            return -corr_matrix_test[0,1]


################################################################

def alpha_linreg_rmse4min(arAlpha, chMdl, 
                          x_train, y_train, krnl_train, 
                          x_test, y_test, krnl_test, ifVerify, log): 
    #
    # function for minimization for linear model, Ridge and Lasso. 
    # Takes the given alpha, fits the model requested and return skills for test subset 
    # It returns RMSE in the given metric
    #
    # Choose the model for fitting 
    if chMdl == 'MLR': 
        clf = linear_model.LinearRegression()
    elif chMdl == 'RIDGE': 
        clf = linear_model.Ridge(alpha=arAlpha[0], tol=2e-4, max_iter=1000000)
    elif chMdl == 'LASSO': 
        clf = linear_model.Lasso(alpha=arAlpha[0], tol=2e-4, max_iter=1000000)
    else:
        log.log('\nalpha_linreg_rmse4min. Do not support:  %s, only MLR / Ridge / Lasso' % chMdl)
        if ifVerify: return (np.nan, arAlpha[0] * np.sum(np.square(clf.coef_)))
        else: return np.nan
    
    # make the fit for the training dataset
    try:
#        print('args_1\n', ' '.join(str(a) for a in args[1]))
#        print('\n\nargs_2\n', ' '.join(str(a) for a in args[2]))
        clf.fit(x_train, y_train, sample_weight=krnl_train)   # x_train, y_train
    except:
        if ifVerify: 
            log.log('Failed fit %s. x_trin[:200], y_train[:200]: \n' % chMdl + 
                    ' '.join(('%g' % x for x in x_train.ravel()[:200])) + '\n' + 
                    ' '.join(('%g' % y for y in y_train.ravel()[:200])) + 
                    '\nnans: ', np.any(np.isnan(x_train)), np.any(np.isnan(y_train)))
            return ((np.sum(np.abs(y_train))+1)*1e10, arAlpha[0] * np.sum(np.square(clf.coef_)))
        else: return (np.sum(np.abs(y_train))+1)*1e10  # really bad RMS.
    
    # compute the prediction for the test subset
    y_test_predicted = clf.predict(x_test)
    
    # finally, quality for the test dataset: RMSE
#    diff = np.subtract(y_test_predicted, y_test)
#    diff_2 = np.square(np.subtract(y_test_predicted, y_test))
#    mse = np.square(np.subtract(y_test_predicted, y_test)).mean()
    rmse_test = np.sqrt((np.square(y_test_predicted - y_test) * krnl_test).mean())
#    y_train_predicted = clf.predict(x_train)
#    rmse_train = np.sqrt((np.square(np.subtract(y_train_predicted, y_train))).mean())
#    print('Alpha ', arAlpha, 'RMSEtrain, ', rmse_train, 'RMSEtest', rmse_test)
    
    if ifVerify:
        return (np.sqrt((np.square(y_test_predicted - y_test) * krnl_test).mean()),
                arAlpha[0] * np.sum(np.square(clf.coef_)))
    else:
        return np.sqrt((np.square(y_test_predicted - y_test) * krnl_test).mean())
    

################################################################

def fit_L_curve_log10_2(bckgr_err, a, b):
    return np.log10(np.log10(np.maximum(1e-10, (a * np.exp(-b * bckgr_err) + (1.0 - a) + 1.0))))

def fit_L_curve_log10(bckgr_err, a, b):
#    print('fit_L_curve_log10, bckgr_err: ', bckgr_err, a, b)
#    print('log10 argument',(a * np.exp(-b * bckgr_err) + (1.0 - a)))
#    print('log10 values',np.log10(a * np.exp(-b * bckgr_err) + (1.0 - a)))
    return np.log10(np.maximum(1e-10, a * np.exp(-b * bckgr_err) + (1.0 - a)))

def fit_L_curve(bckgr_err, a, b):
    return a * np.exp(-b * bckgr_err) + (1.0 - a)

#----------------------------------------------------------------------

def l_curve_find_best_iteration(obs_err_, bckgr_err, outFNm=None, caseNm=None):
    #
    # Find the best iteration looking at the L-curve shape. 
    # Vertical position of the curve does not matter - as long as it is decaying all is fine
    #
#    critical_derivative = -0.1     # absolute decay  rate
    critical_grad_slowdown = 0.1   # relative slowdown compare to the initial slope
    #
    # Ensure positiveness, normalise and find the best fitting
    #
    if np.min(obs_err_) <= 0:
        shift_y = np.min(obs_err_) * 1.01
        if shift_y == 0: shift_y -= 0.01 * abs(np.max(obs_err_))
    else: shift_y = 0
    obs_err = obs_err_ - shift_y
    maxobs = np.max(obs_err)
    maxbckgr = np.max(bckgr_err)
    obs_err_rel = obs_err / maxobs
#    if not np.all(obs_err_rel > 0) or np.min(obs_err_) * np.max(obs_err_) < 0:
#        print('obs_err_rel', obs_err_rel)
#        print('log10 of obs_err_rel', np.log10(obs_err_rel))
    obs_err_rel_log10 = np.log10(obs_err_rel)
#    obs_err_rel_log10_2 = np.log10(np.log10(obs_err_rel+1))
    bckgr_err_rel = bckgr_err / maxbckgr
    # Do the log10 type of fitting to limit the effect of huge spikes
    # The rest of processing works with real fitting curve
    try:
        params = optimize.curve_fit(fit_L_curve_log10, bckgr_err_rel, obs_err_rel_log10)   # find the best fit
    except:
        return (bckgr_err_rel.shape[0]-1, bckgr_err[-1])
    [a,b] = params[0]
    # critical derivative of the fit
#    xFitCrit = -1.0 / b * np.log(-critical_derivative / a / b)
    # critical slowdown
    xFitCritSlopeDecayRel = -np.log(critical_grad_slowdown) / b
    # Need to find the iteration closest to the critical points
    # It is not monotonous along either of the axes
    # last element with small deviation from background
    try:
        idx_last_sml_bckgr_err = np.argwhere(bckgr_err_rel < xFitCritSlopeDecayRel)[-1][0]
    except:
        idx_last_sml_bckgr_err = np.argwhere(bckgr_err_rel < xFitCritSlopeDecayRel)[-1][0]
    # the closest to the threshold but above it
    idx_closest_lrg_bcgr_err = np.argmin(bckgr_err_rel[idx_last_sml_bckgr_err:]) + idx_last_sml_bckgr_err
    # The one closer to the threshold is the stopping point
    if (xFitCritSlopeDecayRel - bckgr_err_rel[idx_last_sml_bckgr_err] < 
        0.5 * (bckgr_err_rel[idx_closest_lrg_bcgr_err] - xFitCritSlopeDecayRel)):
        iterCrit = idx_last_sml_bckgr_err
    else:
        iterCrit = idx_closest_lrg_bcgr_err
    
    if iterCrit == bckgr_err_rel.shape[0]: iterCrit -= 1
    if bckgr_err_rel[iterCrit]**2 + obs_err_rel[iterCrit]**2 > bckgr_err_rel[iterCrit-1]**2 + obs_err_rel[iterCrit-1]**2:
        iterCrit -= 1
    try:                              # can we get a win-win at the next iteration? But careful: it may not exist
        if bckgr_err_rel[iterCrit+1] < bckgr_err_rel[iterCrit] and obs_err_rel[iterCrit+1] < obs_err_rel[iterCrit]: 
            iterCrit += 1
    except: pass
    if iterCrit < 1: iterCrit = bckgr_err_rel.shape[0]-1
        
    print('L-curve fit (a,b), iterCrit:', a,b, iterCrit)
    #
    # Plot what was obtained
    #
    if outFNm is not None and iterCrit < bckgr_err_rel.shape[0]-1:
        yFitCrit = maxobs * fit_L_curve(xFitCritSlopeDecayRel, a, b)
        xCrit = np.array([xFitCritSlopeDecayRel * maxbckgr, bckgr_err[iterCrit]])
        yCrit = np.array([yFitCrit, obs_err[iterCrit]])
        
#        draw_l_curve(obs_err, bckgr_err, xCrit, yCrit, iterCrit, a,b, outFNm, caseNm)

        ind = np.arange(len(obs_err)) + 1
        xdim = np.arange(0,1,0.02)                   # prepare fitted arrrays for plotting
        ydim = fit_L_curve(xdim, a, b)
        xdim *= max(bckgr_err)                             # scale them to the actual dimensions
        ydim *= max(obs_err)
        fig, ax = plt.subplots()
        ax.plot(bckgr_err, obs_err+shift_y, marker='o', linestyle='-', color='Blue')
        ax.plot(xdim, ydim+shift_y, marker='', linestyle='-',color='Green')            # its fit
        ax.plot(xCrit, yCrit+shift_y, marker='*', markersize=15, linestyle='',color='Red')     # critical point
        if len(obs_err) < 50:
            for X, Y, Z in zip(bckgr_err, obs_err+shift_y, ind):
                # Annotate the data entries, text 5 points above and to the right of the vertex
                ax.annotate('{}'.format(Z), xy=(X,Y), xytext=(5, 5), ha='right',
                            textcoords='offset points',fontsize=8)
        ax.set_xlabel('[deviation from bckgr]', fontsize=10)
        ax.set_ylabel('[deviation from observations]',fontsize=10)
        ax.set_ylim(min(0.9*np.float32(obs_err))+shift_y, 1.1*np.float32(obs_err[0])+shift_y)
        ax.set_xlim(0.0, 1.1*np.float32(bckgr_err[-1]))
        ax.set_title(caseNm + ', best iter = ' + str(iterCrit+1), fontsize=10)
        ax.legend(['iterations','approximation','optimal iteration'],fontsize=10)
        plt.savefig(outFNm, dpi = 600)
        plt.close()

    return (iterCrit, xFitCritSlopeDecayRel * maxbckgr)




################################################################
################################################################
#
# Fitting models for finding optimal Alpha.
# Somehow, BFGS failed in some applications. Until the reason is found
# I made own optimizer. Nothing fancy - for 1-dimensional problem, everything goes.
#
class linregr_regularised_model:

    #-------------------------------------------------------------------
 
    def __init__(self, chModelName, metric2miimise, AlphaMaxIter, log):
        self.chMdl = chModelName
        if self.chMdl.upper() == 'MLR':
            self.fMdl = linear_model.LinearRegression
        elif self.chMdl.upper() == 'RIDGE': 
            self.fMdl = linear_model.Ridge
        elif self.chMdl.upper() == 'LASSO': 
            self.fMdl = linear_model.Lasso
        else:
            self.log.log('linregr_regularised_model. Do not support the model:  %s, only MLR/Ridge/Lasso so-far' % 
                         self.chMdl)
            self.error_flag = -1.0  # failed initialization
            return
        self.metric2miimise = metric2miimise
        self.alpha = 1             # overwritten by get_initial_alpha
        self.alphaStep = 0.1       # overwritten by get_initial_alpha
        self.alphaSpeedUp = 1.5
        self.alphaSlowDown = 0.2
        self.AlphaTolerance = 1e-4  # overwritten by get_initial_alpha
        self.AlphaMaxIter = AlphaMaxIter
        self.log = log
        self.error_flag = 1.0     # no problem

    
    #-------------------------------------------------------------------
 
    def get_initial_alpha(self, *args):
        #
        # Simply scan a sufficiently wide range of alphas and take the best
        #
        chMdl, x_train, y_train, krnl_train, x_test, y_test, krnl_test, ifVerify, log = args
        argsLocal = chMdl, x_train, y_train, krnl_train, x_test, y_test, krnl_test, True, log
        # order of the minimal alpha, e.g. -6 for 1e-6
        pMin = int(np.log10(minimum_regularization_alpha))
        # values of the metric for test set
        arAlpha = list((10.0**iPwr for iPwr in range(pMin,10)))
        metric_full = np.array(list((self.metric2miimise([a], *argsLocal) for a in arAlpha)))
        # after max the regularizing term decreases: overregularization. Cut that tail
        # Careful: the 0-th element is < min_alpha, i.e.it will be MLR no matter, what
        idxMax = max(1,np.argmax(metric_full[:,1]))
        metric_main = metric_full[ : min(idxMax+1, metric_full.shape[0]), 0]
        regularizer = metric_full[ : min(idxMax+1, metric_full.shape[0]), 1]
        self.alpha = arAlpha[0]
        self.alphaStep = 0
        self.AlphaTolerance = 0
        self.idxLastGood = 0
        idxOK = np.isfinite(metric_main)
        if np.sum(idxOK) == 0: return   # all failed
        if np.std(metric_main[idxOK]) == 0: return   # nothing useful anyway
# wrong idea: metric is anything
#        if np.min(metric_main[idxOK]) <= 0: return   # something really strange
        #
        # Useful case
        # Need to take the min possible regularization, which ensures nearly-minimal 
        # main minimization term. Note that it is for the test subset, i.e. the lower 
        # it is the better.
        # But it cannot be lowered at costs of too high regulaization. No matter what,
        # regularization term should not exceed the main term
        if np.any(np.abs(metric_main[idxOK]) < regularizer[idxOK]):
            self.idxLastGood = np.argwhere(np.abs(metric_main[idxOK]) < regularizer[idxOK])[0][0]
        else:
            self.idxLastGood = np.argwhere(idxOK)[-1][-1]  # the last valid term
        # Weird case, report it and set something not too stupid
        if metric_main[idxOK][:self.idxLastGood].size < 1:
            log.log('get_initial_alpha cannot find alpha. self.idxLastGood: %g' % self.idxLastGood)
            log.log('Metric:' + ' '.join(list(('%g' % v for v in metric_full[:,0]))))
            log.log('Regularizer:' + ' '.join(list(('%g' % v for v in metric_full[:,1]))))
            log.log('pMin=%g, idxMax=%g, idxLastGood=%g' % (pMin, idxMax, self.idxLastGood))
            self.alpha = idxMax
        else:
            # ...just the minimum of the metric
            self.alpha = 10.0**(np.argmin(metric_main[idxOK][:self.idxLastGood]) + pMin)
        # Another possibility (not good, as it seems) is to require 10% above min for the main term
#        idxLastGood = np.argwhere(metric_main[idxOK] < 1.1 * np.min(metric_main[idxOK]))[0]
#        self.alpha = 10.0**(idxLastGood + pMin)
        self.alphaStep = self.alpha * 0.1
        self.AlphaTolerance = self.alphaStep * 0.01
        if self.alpha > 1000:
            log.log('get_initial_alpha sends suspicious alpha: %g' % self.alpha)
            log.log('Metric:' + ' '.join(list(('%g' % v for v in metric_full[:,0]))))
            log.log('Regularizer:' + ' '.join(list(('%g' % v for v in metric_full[:,1]))))
            log.log('pMin=%g, idxMax=%g, idxLastGood=%g' % (pMin, idxMax, self.idxLastGood))

        
    #-------------------------------------------------------------------
 
    def get_regulariser_weight_and_fit_BFGS(self, x_train, y_train, krnl_train, x_test, y_test, krnl_test):
        #
        # Look for optimal Alpha with BFGS. 
        # ATTENTION. It failed in at least one case: found the optimum and
        # then continued 
        #
        if self.chMdl == 'MLR':
            self.clf = linear_model.LinearRegression().fit(x_train, y_train, krnl_train)
            return self.clf
        # more sophisticated models allow optimization of the fit
        # what is the starting alpha?
        self.get_initial_alpha(self.metric2miimise, self.chMdl, x_train, y_train, krnl_train, 
                               x_test, y_test, krnl_test, False, self.log)
        
        # optimise
        self.alpha = optimize.minimize(self.metric2miimise, [0.0], self.chMdl, 
                                       x_train, y_train, krnl_train, x_test, y_test, krnl_test, 
                                       False, self.log) #, bounds=[(0.0,None)], options={"maxiter":20,"disp":True}) # method, jac, hess, hessp, bounds, constraints, tol, callback, options)

        # finally, fit the model with the found alpha
        if self.alpha < minimum_regularization_alpha:
            self.clf = linear_model.LinearRegression()  # if no regularization, no need to bother
        else: 
            self.clf = self.fMdl(alpha=self.alpha, tol=2e-4, max_iter=1000000)

        self.clf.fit(x_train, y_train, sample_weight=krnl_train)
        return self.clf
    
    #-------------------------------------------------------------------
 
    def get_regulariser_weight_and_fit(self, x_train, y_train, krnl_train, x_test, y_test, krnl_test):
        #
        # Looking for the optimal Alpha manually
        #
        if self.chMdl == 'MLR':
            self.clf = linear_model.LinearRegression().fit(x_train, y_train, sample_weight=krnl_train) 
            return self.clf
        # more sophisticated models allow optimization of the fit
        args = (self.chMdl, x_train, y_train, krnl_train, x_test, y_test, krnl_test, True, self.log)
        # what is the starting alpha?
        self.get_initial_alpha(*args)

        # now, can optimise
        # But note that alpha can only be lowered: its further increase is dangerous
        metric, regularizer = self.metric2miimise([self.alpha], *args)
        #
        # First of all, make sure that metric is larger than the regularizer
        #
        while metric < regularizer and self.alpha > minimum_regularization_alpha:
            self.alpha /= 2.0
            metric, regularizer = self.metric2miimise([self.alpha], *args)
            if np.isnan(metric):
                print('ERROR. Nan metric in the initial alpha %g. Input / output data are:' % self.alpha)
                args = (self.chMdl, x_train, y_train, krnl_train, x_test, y_test, krnl_test, True, self.log)
                metric = self.metric2miimise([self.alpha], *args)
                print('Cannot continue')
                return None
        #
        # Now, can start fine tuning
        cnt = 0
        alphas = []
        while self.alphaStep > self.AlphaTolerance:
            cnt += 1
            if cnt > self.AlphaMaxIter or self.alphaStep < self.AlphaTolerance: 
                break
            metric_up, regularizer_up = self.metric2miimise([self.alpha + self.alphaStep], *args)
            if (metric_up < metric * (1.0 - 1e-5 * np.sign(metric)) and  # metric is improving 
                metric_up > regularizer_up * 5. and              # regularizer is less than metric              
                regularizer_up > regularizer):              # regularizer grows, no saturation
                #
                # larger alpha is a good idea
                self.alpha += self.alphaStep
                self.alphaStep *= self.alphaSpeedUp
                metric = metric_up
                regularizer = regularizer_up
            else:
                # larger alpha is not a good idea
                if self.alpha - self.alphaStep > 0.0:  # anywhere to go downwards?
                    metric_down, regularizer_down = self.metric2miimise([self.alpha - 
                                                                         self.alphaStep], *args)
                    if (metric_down < metric * (1.0 - 1e-5 * np.sign(metric)) or 
                        metric < regularizer * 5.0):
                        #
                        # smaller alpha is good 
                        self.alpha = self.alpha - self.alphaStep
                        self.alphaStep *= self.alphaSpeedUp
                        metric = metric_down
                        regularizer = regularizer_down
                    else:
                        self.alphaStep *= self.alphaSlowDown
                else: # neither direction helps: cut the step
                    self.alphaStep *= self.alphaSlowDown
            alphas.append((self.alpha, self.alphaStep, metric, regularizer))

        if self.alpha > 1000:
            self.log.log('Potential problem: strong regularization')
            self.log.log('%s model . Mainterm %g, regularization %g, weight %g' % 
                         (self.chMdl, metric, regularizer, self.alpha))
            metric_test = np.array([self.metric2miimise([self.alpha * 1.5**iPwr], *args) 
                                for iPwr in range(-5,5)])
            self.log.log('alpha: ' + ' '.join(list(('%g' % (self.alpha * 1.5**iPwr) 
                                                    for iPwr in range(-5,5)))))
            self.log.log('regularizer: ' + ' '.join(list(('%g' % v for v in metric_test[:,1]))))
            self.log.log('metric: ' + ' '.join(list(('%g' % v for v in metric_test[:,0]))))
            self.log.log('Trajectory:')
            for a in alphas: self.log.log('alpha = %g, alphaStep = %g, metric = %g, regul = %g' % a)
        # Select the fitting model
        if self.alpha < minimum_regularization_alpha:
            self.clf = linear_model.LinearRegression()  # if no regularization, no need to bother
        else: 
            self.clf = self.fMdl(alpha=self.alpha, tol=2e-4, max_iter=1000000)
        # Fit the model and return
        try:
            self.clf.fit(x_train, y_train, sample_weight=krnl_train)
        except:
            self.clf = linear_model.LinearRegression()
            self.clf.fit(x_train, y_train, sample_weight=krnl_train)
#        return self.clf


    #-------------------------------------------------------------------
 
    def verify_regularization(self, x_train, y_train, krnl_train, x_test, y_test, krnl_test):
        #
        # Regularization term can be nonsense if the problem is really bad
        # Then we either drop the story of sharply reduce regularizatrion
        # See what can be done
        #
        args = (self.chMdl, x_train, y_train, krnl_train, x_test, y_test, krnl_test, True, self.log)
        main_term, regularizer = self.metric2miimise([self.alpha], *args)
        main_term_init = main_term
#        self.log.log('%s model verifies regularization. Mainterm %g, regularization %g, weight %g' % 
#                     (self.chMdl, main_term, regularizer, self.alpha))
        if False:   #main_term_init < 10. * regularizer:
#            self.log.log('Reduce regularization to 0.01 of the main term')
            #
            # recompute the scores to verify that we did not damage it too much
            # Note that the main term is for the test subset, i.e. with lower 
            # regularization it will increase due to overfitting
            #
            alphaTmp = self.alpha
            main_term_new = main_term
            regularizer_new = regularizer
            while main_term_new < main_term_init * 2. and main_term_new < 10. * regularizer_new: 
                self.alpha /= 2.0
                if self.alpha < 1e-10:
                    self.log.log('Turning off regularization')
                    self.clf = linear_model.LinearRegression()  # if things go wrong way
                    self.clf.fit(x_train, y_train, sample_weight=krnl_train)
                    return 0
                main_term_new, regularizer_new = self.metric2miimise([self.alpha], *args)
#                if main_term < 0.01:
#                    print(self.alpha, main_term, main_term_new, regularizer_new)
            self.alpha *= 2.0   # the last iteration is dropped
            self.log.log('alpha changed from %g to %g, new main term %g, new reguarizer %g' %
                         (alphaTmp, self.alpha, main_term_new, regularizer_new))
            #
            # After changing reulaiser weight, have to refit the model
            #
            try:
                self.clf = self.fMdl(alpha=self.alpha, tol=2e-4, max_iter=1000000)
                self.clf.fit(x_train, y_train, sample_weight=krnl_train)
            except:
                self.clf = linear_model.LinearRegression()  # if things go wrong way
                self.clf.fit(x_train, y_train, sample_weight=krnl_train)
        if self.alpha > 1000:
            self.log.log('Potential problem: strong regularization')
            self.log.log('%s model verifies regularization. Mainterm %g, regularization %g, weight %g' % 
                     (self.chMdl, main_term, regularizer, self.alpha))
#            self.get_initial_alpha(*args)
#            main_term, regularizer = self.metric2miimise([self.alpha], *args)
        return self.alpha  # no matter what, we return the new / old regularization weight

    #-------------------------------------------------------------------
 
    def fit(self, x, y, kernel):     # Fits the current model assuming that all parameters are OK.
#        print(self.clf)
        return self.clf.fit(x,y, sample_weight=kernel)
    
    #-------------------------------------------------------------------
 
    def predict(self, x):    # generate prediction with predictors given
#        print(x.shape)
        return (self.clf.predict(x) * self.error_flag)

    #-------------------------------------------------------------------
 
    def score(self, x, y, kernel):   # compute the quality of prediction from predictors and answers 
        return self.clf.score(x, y, sample_weight=kernel)



#################################################################################
#
# Parameter optimization via random walk
# Receives the parameters to fit, the input data for the computation and the function,
# which uses the input data and the parameters.
#
#################################################################################
        
class Param_refine_MonteCarlo():

    #--------------------------------------------------------------------------
 
    def __init__(self, object_to_fit, 
                 rndA, rndB, cost_function, check_params, tsObserved, training_fraction, 
                 subset_selection, sample_weights, ifVerbose, chLabel, randomGen, log):
        self.iStep = 0                        # iteration
        self.log = log                        # log file
        self.outDir = os.path.split(self.log.get_fname())[0]
        try: 
            os.makedirs(os.path.join(self.outDir,'cost'))
            os.makedirs(os.path.join(self.outDir,'params'))
        except: pass
        self.object_to_fit = object_to_fit
        self.best_params = self.object_to_fit.get_params()    # parameters vector to fit
        self.param_names = self.object_to_fit.get_param_names()    # their names
        self.rndA = rndA                      # additive part for random step
        self.rndB = rndB                      # multiplier for random walk
        self.cost_function = cost_function  # function that computes the cost
        self.check_params = check_params      # function that checks the consistency of parameters
        self.tsObservations = tsObserved      # observations to compare with, full set
        self.sample_weights = sample_weights
        self.randomGen = randomGen
        self.train_indices, self.test_indices = split_times(self.tsObservations.times,
                                                            self.sample_weights, 
                                                            training_fraction, 
                                                            subset_selection, self.randomGen, self.log)
        self.ifVerbose = ifVerbose
        self.chLabel = chLabel


    #--------------------------------------------------------------------------

    def fit_params(self, maxIter, reportStep):
        #
        # Implementation of MCMC in a very simple manner, with periodic drawer
        #
        cost = self.cost_function(self.object_to_fit, self.best_params, 
                                  self.tsObservations, self.train_indices, self.sample_weights)
        cost_init = cost
        self.log.log('Initial cost = %g' % cost)
        arIterCost = []
        x_test = []
        costTest = []
        colors = []
        mSize = []
        par_vals = np.zeros(shape=(len(self.best_params),maxIter))
        nVoidIter = 0
        iterCrit = None
        dicParamsBestIter = {}
        for iter in range(maxIter):
            #
            # Get next iteration for parameters
            #
            ifCorrect = False
            for cnt in range(1000):
                newParams = self.randomGen.uniform(self.best_params + self.rndA + self.best_params *
                                                   self.rndB,
                                                   self.best_params - self.rndA - self.best_params *
                                                   self.rndB)
                ifCorrect = self.check_params({self.object_to_fit.get_name() : newParams})
                if ifCorrect: break
            if not ifCorrect: 
                self.log.log('Failed next set of parameters. Iterations stopped')
                return None
            # Set the new parameters
            self.object_to_fit.set_params(newParams)
            
            # New cost
            costNew = self.cost_function(self.object_to_fit, newParams, 
                                         self.tsObservations, self.train_indices, self.sample_weights)
            if self.ifVerbose: 
                self.log.log('Iter %g cost = %g' % (iter, costNew))
            arIterCost.append(costNew)
            for iPar in range(len(self.best_params)):
                par_vals[iPar,iter] = newParams[iPar]
            # better ?
            if costNew < cost:
                # Better!
                if cost - costNew > 1e-3 * np.abs(cost):
                    cost = costNew
                    # store the parameters
                    self.best_params = newParams.copy()
                    dicParamsBestIter[iter] = (newParams.copy(), costNew)
                    self.log.log('>>>> Best iteration %g, cost %g' % (iter, cost))
                    self.report_params()
                    self.log.log('<<<<')
                    colors.append(0.95)
                    mSize.append(20)
                else:
                    self.log.log('==== slightly better iter %g cost %g' % (iter, cost))
                    colors.append(0.65)
                    mSize.append(10)
                # If improved, get test dataset for this iteration
                x_test.append(iter)
                costTest.append(self.cost_function(self.object_to_fit, newParams, 
                                                  self.tsObservations, self.test_indices, 
                                                  self.sample_weights))
                nVoidIter = 0
                #
                # If L-curve is beyond the midpoint, stop the iterations.
                # L-curve receives only successful iteratoins: nobody knows where wrong ones can go
                #
                if iter > 20:
                    itersGood = sorted(list(dicParamsBestIter.keys()))
                    costsGood = np.array(list((dicParamsBestIter[iter][1] for iter in itersGood)))
                    idxIterCrit, xFitCrit = l_curve_find_best_iteration(costsGood, itersGood)
                    iterCrit = itersGood[idxIterCrit]
                    if iterCrit < 0.5 * iter: 
                        print('L-curve iterCrit:', iterCrit)
                        # do L-curve again, with drawing
                        l_curve_find_best_iteration(costsGood, itersGood,
                                                    os.path.join(self.outDir,'cost','l_curve_%s.png' % 
                                                                 self.chLabel), self.chLabel)
                        break
                else: 
                    iterCrit = iter
            else: 
                # Void try - not better
                colors.append(0.35)
                mSize.append(5)
                nVoidIter += 1

#                if cost * costNew < 0:
#                    costNew = self.cost_function(self.object_to_fit, newParams, 
#                                                 self.tsObservations, self.train_indices, self.sample_weights)
                    
                
                
#            # Time to draw figures ?
#            if (np.mod(iter,reportStep) == 0 or iter == maxIter-1 or nVoidIter == 0) and iter > 0:
#                if iter == maxIter-1: dpi = 200
#                else:dpi = 100
#                self.draw_progress(iter, arIterCost, x_test, costTest, par_vals, colors, mSize, 
#                                   iterCrit, dpi)
            #
            # If firmly stalled, end the tries
            #
            if nVoidIter > 50: break
        #
        # If iterCrit has been found prior to stopping the iterations, use it instead ofthe last best one
        #
        print('iterCrit < iter', iterCrit, iter)
        if iterCrit < iter:
            self.best_params, cost = dicParamsBestIter[iterCrit]
            self.object_to_fit.set_params(self.best_params)
            self.log.log('L-curve stop: %g' % iterCrit)
            # Draw the final status
            self.draw_progress(iter, arIterCost, x_test, costTest, par_vals, colors, mSize, iterCrit, 200)
        else:
            # the last iteration is taken forwards
            self.draw_progress(iter, arIterCost, x_test, costTest, par_vals, colors, mSize, 
                               sorted(list(dicParamsBestIter.keys()))[-1], 200)

        return (self.best_params, cost_init, cost)

    #---------------------------------------------------------------------------

    def report_params(self):
        for chNm, fPar in zip(self.param_names, self.best_params):
            self.log.log('%s = %g' % (chNm, fPar))
    
    #---------------------------------------------------------------------------

    def draw_progress(self, iter, arIterCost, x_test, costTest, par_vals, colors, mSize, iterCrit, dpi):
        #
        # Draws cost for training and test subsets
        #
        fig,ax = plt.subplots()
        ax.scatter(range(iter+1), arIterCost, c=colors, s=mSize, marker='o',linewidth=0, 
                   label='training',cmap='tab10')
        ax.scatter(x_test, costTest, color='orange', s=20, marker='o', linewidth=0,label='test')
        if iterCrit is not None:
            ax.scatter([iterCrit], [arIterCost[iterCrit]], color='red', s=100, marker='*', 
                       linewidth=0, label='stop iter.')
        ax.set_xlabel('iteration')
        ax.set_ylabel('cost')
        ax.legend()
        ax.set_title('Cost for  ' + self.chLabel)
#        flds = os.path.split(self.log.get_fname().replace('.log',''))
        plt.savefig(os.path.join(self.outDir,'cost', 'cost_%s.png' % self.chLabel), dpi=dpi)
        plt.clf()
        plt.close()
        # parameters
        for iPar in range(len(self.param_names)):
            fig,ax = plt.subplots()
            p = ax.scatter(range(iter+1), par_vals[iPar,:iter+1], c=colors, marker='o',
                           linewidth=0, cmap='viridis_r', norm=plt.Normalize(0.,1.), s=mSize)
            if iterCrit is not None:
                ax.scatter([iterCrit], [par_vals[iPar,iterCrit]], color='red', s=70, marker='*', 
                           linewidth=0, label='stop iter.')
            ax.set_xlabel('iteration')
            ax.set_ylabel(self.param_names[iPar])
            ax.set_ylim(np.min(par_vals[iPar,:iter+1]) * 0.99 - 1e-10, 
                        np.max(par_vals[iPar,:iter+1]) * 1.01 + 1e-10)
            ax.set_title(self.chLabel + '   ' + self.param_names[iPar])
            plt.savefig(os.path.join(self.outDir, 'params', 'params_%s_%s.png' % 
                                     (self.chLabel, self.param_names[iPar])), dpi=dpi)
            plt.clf()
            plt.close()


#################################################################################
#################################################################################
#################################################################################
#
# Classes for static and differential mapping models
# Based on Eugene Genikhovich et al ideas in interpretation and implementation
# and adaptation of Mikhail Sofiev and Olga Sozinova 
#
#################################################################################
#################################################################################
#################################################################################

#
# The idea is:
# Pre-process the input data so that they become linearly related to the predicted
# variables, then do a simple multi-linear regression
#
class input_data_holder():

    #-------------------------------------------------------------------
 
    def __init__(self,
                 ID,             # ID of the dataset
                 tsPoolPredictors,   # (nTimes, nStations, nPredictors) or list((nTimes, nStations))
                 tsMatrixObserv,     # (nTimes, nStations)
                 sample_weights4stat,    # metric weighting kernel, stations2use already accounted
                 training_fraction,  # [0..1] 
                 subset_selection,   # native_rnd, daily_rnd, daily_first_test, daily_last_test
                 idxStations2use,    # filter for stations in the tsMatrices
                 log): 
        #
        # Stores the input dataset (no copying unless necessary)
        # Handles two types of input: 3D tsMatrix and a list of 2D tsMatrices
        # List is needed for the case of predictors made a separate, possibly, named
        # datasets. One element of teh list is then the 2D tsMatrix
        # The 3D tsMatrix is when the predictors are put into the fastest-changing
        # index of teh 3D tsMatrix and thus can treated as a set of equal-rights predictors
        #
        self.log = log
        self.idxStations2use = idxStations2use
        #
        # Split the input data for training and sets, Removes the missing data, flattens if needed 
        #
        if isinstance(tsPoolPredictors, list) :
            self.ifList_of_2D_tsm = True
            self.nTimes, self.nStat = tsPoolPredictors[0].vals.shape
            if idxStations2use is None:
                idxStLoc = range(self.nStat)
            else:  
                idxStLoc = idxStations2use
                self.nStat = len(idxStations2use)
            self.nPred = len(tsPoolPredictors)
            self.log.log('input_data_holder received a list of 2D tsMatrix with %i predictors, %i times, %i stations' % 
                         (self.nPred, self.nTimes, self.nStat))
        else:
            self.ifList_of_2D_tsm = False
            self.nTimes, self.nStat, self.nPred = tsPoolPredictors.vals.shape
            if idxStations2use is None: 
                idxStLoc = range(self.nStat)
            else:
                self.nStat = len(idxStations2use)
                idxStLoc = idxStations2use
            print(self.nPred, self.nTimes, self.nStat)
            self.log.log('input_data_holder received 3D tsMatrix with %i predictors, %i times, %i stations' % 
                         (self.nPred, self.nTimes, self.nStat))
        idxStLoc = np.array(idxStLoc, dtype=np.int32)
        #
        # Split times to training and test subsets
        #
        train_indices, test_indices = split_times(tsMatrixObserv.times, sample_weights4stat, 
                                                  training_fraction, subset_selection, self.randomGen,
                                                  log)
                
        self.nTraining = len(train_indices)
        self.nTest = len(test_indices)
        #
        # Organization of the dataset
        #
        self.train_obs = tsMatrixObserv.vals[train_indices,idxStLoc]
        self.test_obs = tsMatrixObserv.vals[test_indices,idxStLoc]
        # predictors are arranged differently...
        if self.ifList_of_2D_tsm:
            self.train_pred = []   # copying the data is the simplest option...
            self.test_pred = []
            # Careful: if we ask for a single station, it will not work. Have to explicitly reshape
            for tsm in tsPoolPredictors:
                self.train_pred.append(MyTimeVars.TsMatrix(tsm.times[train_indices],
                                                           np.array(tsm.stations)[idxStLoc], 
                                                           tsm.variables, 
                                                           tsm.vals[train_indices,idxStLoc].
                                                           reshape(len(tsm.times[train_indices]), len(idxStLoc)), 
                                                           tsm.units, fill_value=tsm.fill_value))
                self.test_pred.append(MyTimeVars.TsMatrix(tsm.times[test_indices], 
                                                          np.array(tsm.stations)[idxStLoc], 
                                                          tsm.variables, 
                                                          tsm.vals[test_indices,idxStLoc].
                                                          reshape(len(tsm.times[test_indices]), len(idxStLoc)), 
                                                          tsm.units, fill_value=tsm.fill_value))
        else:
            self.train_pred = tsPoolPredictors.vals[train_indices,idxStLoc,:]
            self.test_pred = tsPoolPredictors.vals[test_indices,idxStLoc,:]


    #---------------------------------------------------------------------

    def get_data_training_flatten(self):
        #
        # Returns flattened and filtered for nans input data for training 
        #
        # missing values are in observations.
        train_obs_flat = np.ndarray.flatten(self.train_obs)   # (nTimes, nStations), flatten
        idxTrainOK = np.isfinite(train_obs_flat)   # identifying individual missing observations
        if self.ifList_of_2D_tsm:
            return (list( (np.reshape(tsm, (self.nPred, self.nTraining * nStat)).T[idxTrainOK] 
                           for tsm in self.train_pred)),
                    train_obs_flat[idxTrainOK])
        else:
            return (np.reshape(self.train_pred, (self.nTraining * nStat, self.nPred))[idxTrainOK,:],
                    train_obs_flat[idxTrainOK])

    #---------------------------------------------------------------------

    def get_data_test_flatten(self):
        #
        # Returns flattened and filtered for nans input data for training 
        #
        # missing values are in observations.
        test_obs_flat = np.ndarray.flatten(self.test_obs)
        idxTestOK = np.isfinite(test_obs_flat)     # identifying individual missing observations
        if self.ifList_of_2D_tsm:
            return (list( (np.reshape(tsm, (self.nPred, self.nTest * nStat)).T[idxTestOK] 
                           for tsm in self.test_pred)),
                    train_obs_flat[idxTrainOK])
        else:
            return (np.reshape(self.test_pred, (self.nTest * nStat, self.nPred))[idxTestOK,:],
                    test_obs_flat[idxTestOK])

    #--------------------------------------------------------------------

    def get_data_training(self):
        # Just returns the training datasets
        return(self.train_pred, self.train_obs) #, self.test_pred, self.test_obs)

    #--------------------------------------------------------------------

    def get_data_test(self):
        # Just returns the test datasets
        return(self.test_pred, self.test_obs)


##########################################################################
#
# Class for holding the pre-processed data, input for the fitting routines 
#
class processed_data:
    def __init__(self, ID, quality_flag, validPredictors=None, 
                 train_days=None, train_predictorsTr=None, train_outvars=None, train_kernel=None, 
                 test_days=None, test_predictorsTr=None, test_outvars=None, test_kernel=None, 
                 times2fcst=None, all_predictorsTr=None):
        self.ID = ID
        self.quality_flag = quality_flag 
        if quality_flag != flg_OK: return # do not bother if something is wrong
        self.validPredictors = validPredictors
        self.train_days = train_days
        self.train_predictorsTr = train_predictorsTr
        self.train_outvars = train_outvars
        self.train_kernel = train_kernel
        self.test_days = test_days
        self.test_predictorsTr = test_predictorsTr
        self.test_outvars = test_outvars
        self.test_kernel = test_kernel
        self.times2fcst = times2fcst
        self.all_predictorsTr = all_predictorsTr
         

##########################################################################
#
# Class for holding the prediction rules. The entirety of the fitting results are to be here
# An alias of this class allows an application of the model without refitting it again and again
# The rules constitute of three components:
# - control parameters
# - pre-processing rules
# - predictive model (mapping)  
#
class prediction_rules:
    #
    def __init__(self, mapping_type=None, timeStep=None, predictors=None, predictor_interv_edges=None, 
                 LUT_outvar_4_ranges=None, ortogonalization_clf=None, pred_ranking=None, 
                 idxPredictorsInformative=None, main_clf=None, 
                 percents=None, percentiles=None, quantile_rescale=None):
        # Overall initialization: set all variab;es, None if not known
        # metadat
        self.mapping_type = mapping_type
        self.timeStep = timeStep
        self.predictors = predictors
        # projection to output vcariables
        self.predictor_interv_edges = predictor_interv_edges
        self.LUT_outvar_4_ranges = LUT_outvar_4_ranges
        # ortogonalization & censoring
        self.ort_clf = ortogonalization_clf
        self.pred_ranking = pred_ranking
        self.idxPredictorsInformative = idxPredictorsInformative
        # main fitting
        self.main_clf = main_clf
        self.percents = percents
        self.percentiles = percentiles
        self.quantile_rescale = quantile_rescale


    #-----------------------------------------------------------
    def update(self, mapping_type=None, timeStep=None, predictors=None, predictor_interv_edges=None, 
                 LUT_outvar_4_ranges=None, ortogonalization_clf=None, pred_ranking=None, 
                 idxPredictorsInformative=None, main_clf=None, 
                 percents=None, percentiles=None, quantile_rescale=None):
        # Update the items of your choice. Skip those not to be touched 
        # metadata
        if not mapping_type is None: self.mapping_type = mapping_type
        if not timeStep is None: self.timeStep = timeStep
        if not predictors is None: self.predictors = predictors
        # projection to output vcariables
        if not predictor_interv_edges is None: self.predictor_interv_edges = predictor_interv_edges
        if not LUT_outvar_4_ranges is None: self.LUT_outvar_4_ranges = LUT_outvar_4_ranges
        # ortogonalization & censoring
        if not ortogonalization_clf is None: self.ort_clf = ortogonalization_clf
        if not pred_ranking is None: self.pred_ranking = pred_ranking
        if not idxPredictorsInformative is None: 
            self.idxPredictorsInformative = idxPredictorsInformative
        # main fitting
        if not main_clf is None: self.main_clf = main_clf
        if not percents is None: self.percents = percents
        if not percentiles is None: self.percentiles = percentiles
        if not quantile_rescale is None: self.quantile_rescale = quantile_rescale
        return self

    #-----------------------------------------------------------
    def to_pickle(self, chPickleFNm):
        with open(chPickleFNm,'wb') as handle:
            pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)

    #-----------------------------------------------------------
    @classmethod
    def from_pickle(self, chPickleFNm):
        with open(chPickleFNm,'rb') as handle:
            return pickle.load(handle)

    #-----------------------------------------------------------
    def get_all_predictors(self):
        return self.predictors

    #-----------------------------------------------------------
    def get_valid_predictors(self):
        return self.predictors[self.idxPredictorsInformative]

    #-----------------------------------------------------------
    def get_clf_fit(self):
        return self.main_clf

    #-----------------------------------------------------------
    def get_quantile_scale(self):
        return self.quantile_rescale


##########################################################################
#
# Main class for mapping models
#
class mapping_models:

    minStaticMapSample = 6     # minimum input timeseries length
    minIncrementMapSample = 6  # minimum input timeseries length
    nIntervals = 11
    
    def __init__(self, ID, output_variable, wrkDir, randomGen, log, mapping_type=None, 
                 metric2minimise=None, chRunType=None, training_fraction=None, subset_selection=None,
                 iDebug=None, max_fcst_len=None, dicDumpZip=None):
        # Basic initialization of the environment
        self.ID = ID
        self.output_variable = output_variable
        self.mapping_type = mapping_type
        self.metric2minimise = metric2minimise
        self.nMinPredictors = {'MLR':1, 'RIDGE':2, 'RIDGECV':2, 'LASSO':2, 'PERSIST':0} 
        self.fm_model_type = 'MLR'
        self.run_type = chRunType
        self.training_fraction = training_fraction
        self.subset_selection = subset_selection # native_rnd, daily_rnd, daily_first_test, daily_last_test
        self.iDebug = iDebug
        self.max_fcst_len = max_fcst_len
#        random.seed(111)        # it makes the results reproducible exactly
        self.wrkDir = wrkDir
        self.randomGen = randomGen
        self.dicDumpZip = dicDumpZip
        self.log = spp.log(log, wrkDir, 'run_mapping_models_%Y%m%d_%H%M.log')

    #-------------------------------------------------------------

    def lin_intervals(self, values):
        # Splitting the data into equal-size linear intervals
        # returns edges of the intervals
        min_value = np.nanmin(values)
        max_value = np.nanmax(values)
        interval = (max_value - min_value) / self.nIntervals
        return np.array(list((min_value + i*interval for i in range(self.nIntervals+1))),
                        dtype=np.float32)
    
    #-------------------------------------------------------------

    def log_intervals(self, values):
        # Splitting the data into equal-size logarithmic intervals
        # Returns edges of the intervals
        min_value = np.nanmin(values)
        max_value = np.nanmax(values)
        if min_value < 1: delta = 1.0
        else: delta = 0.0
        interval = ((max_value+delta)/(min_value+delta)) ** (1. / self.nIntervals)
        return np.array(list(((min_value+delta)*(interval ** i)-delta 
                              for i in range(self.nIntervals+1))), dtype=np.float32)

    #-------------------------------------------------------------

    def intervals(self, species, values, chLinLogSwitch=None):
        # Splitting the data into equal-ssize intervals.
        # Linear or logarithmic is decided at the top of the module
        if chLinLogSwitch is None:
            if species in log_predictors:  return self.log_intervals(values)
            else:  return self.lin_intervals(values)
        else:
            if chLinLogSwitch.upper() == 'LINEAR':  return self.lin_intervals(values)
            elif  chLinLogSwitch.upper() == 'LOGARITHM':  return self.log_intervals(values)
            else:
                print('Unknown lin-log switch %s, Must LINEAR or LOGARITHM' % chLinLogSwitch)
                return None
            

    #=====================================================================
    #
    # PREPROCESS the input data preparing them to the work
    #
    def preprocess_input_data(self, times2fcst, timesObs, timesPredictors, valObserv, valPoolPredict, 
                              sample_weights, predictors, ifDifferential, stations_handling):
        # Input is 
        #     ID
        #     valPoolPredict,     # (nPredictors, nTimes, nStations) or list((nTimes, nStations))
        #     valObserv,          # (nTimes, nStations)
        #     sample_weights,     # (nTimes, nStations)
        #     predictors          # names of predictors
        #     ifDifferential      # differential or static_mapping
        #     stations_handling   # stations_together or stations_times_together  
        # Output is the object of processed_data
        #    ID
        #    quality_flag 
        #    validPredictors
        #    train_days, train_predictorsTr, train_outvars   # aligned with available observations
        #    test_days,  test_predictorsTr,  test_outvars    # aligned with available observations
        #    species
        #    unit
        #    fcst_predictorsTr                        # the whole set of predictors, full coverage
        # Processing will include:
        # - stupidity checks
        # - projection of predictors to output variables
        # - orthogonalization
        # - reduction of dimensions if some predictors are of low added value
        # - split to training and test sets
        # 
        # The nStations dimension is for spatially-distributed model, whcih has more than one 
        # observation per time. It will be still a single model and single prediction made
        # for all stations. If someone wants them separate, call this class many times.
        #
        # The predictors set is for the whole time period - both fitting and forecasting
        # Then prediction can be called straight after fitting with the same dpp. If a new set of 
        # predictors or times is involved, preprocessing with existing transformations must be
        # called prior to prediction.
        #
        #--------------------------------------------------------------------------
        # Basic stupidity tests
        # valid_obstimes are the TIMES WITH VALID OBSERATIONS, at least at some place
        #
        valid_obstimes = np.array(timesObs[np.any(np.isfinite(valObserv), axis=1)])  
        # Anything left?
        if len(valid_obstimes) == 0:
            self.log.log('No observation for %s, out of: %s times' % 
                         self.ID,','.join(([x.strftime('%Y.%m.%d') for x in valid_obstimes])))
            return (processed_data(self.ID, flg_noObs), (None, None), None)
        # Forecast may be forbidden far outside the reported period
        if self.run_type.upper() == 'FORECAST' and self.max_fcst_len is not None:
            # Check that the symptom reports are not too ancient
            # Here we require the hole being shorter than the length of supplied time series
            nDaysLag = (sorted(days2fcst)[-1] - valid_obstimes[-1]).days 
            print('Age of observations: ', nDaysLag, sorted(days2fcst), '\n', valid_obstimes)
            # Too large gap?
            if nDaysLag > self.max_fcst_len.days:
                return (processed_data(self.ID, flg_oldObs % nDaysLag), (None, None), None)
#        self.log.log('Full date list: ' + ' '.join(d.strftime('%Y:%m:%d') for d in valid_obstimes))
        self.log.log('Full date list size: %i' % len(valid_obstimes))
        #
        # Observations must not be the same. All stations are optimized as one set, so:
        #
        vals, counts = np.unique(valObserv, return_counts=True)
        if np.sum(counts[:-1]) <= 3: 
            return (processed_data(self.ID, flg_fewOutVarData % np.sum(counts[:-1])), (None, None), None)
        #
        #-------------------------------------------------------------------------
        #
        # Differential model? 
        # Then turn the time series into differences 
        #
        if ifDifferential:
            #
            # Time-increment, first for the past (output variables & predictors), then future predictors
            #
            indValidDays = []
            indOutDay = 0
            timesdiff = valid_obstimes[1:] - valid_obstimes[:-1]
            timeStep = np.min(timesdiff)
            # There must be no holes in the data, those should be removed
            indValidTimes = timesdiff == timeStep
            # 
            # Now make the differential dat and observaations
            pdata_obs = (valObserv[1:,:] - valObserv[:-1,:])[indValidTimes,:]
            pdata_predictors = (valPoolPredict[:,1:,:] - valPoolPredict[:,:-1,:])[:,indValidTimes,:]
            
            # Difference is always -1 interval long and there may be holes, so, redefine
            valid_obstimes = copy.copy(valid_obstimes[indValidTimes])

            if self.run_type.upper() == 'FORECAST':
                # Get the reported day that is the days2fcst[0] - one_day
                iLastDay = np.searchsorted(valid_obstimes, times2fcst[0]) - 1  # need the previous element
                try: lastAnalTime = valid_obstimes[iLastDay]
                except: 
                    self.log.log('Strange last analysis day index: %g' % iLastDay); 
                    return (data_processed(self.ID, patients.flg_unknown), (None, None), None)
        else:
            # same names as wrappers
            pdata_obs = valObserv
            pdata_predictors = valPoolPredict
            timeStep = None
        #
        # Intermediate dimensions of the problem
        #
        nPredictors = len(predictors)
        nSites = valObserv.shape[1]
        if nSites > 1:
            print('Multi-cell case')
        nPredTimes = len(timesPredictors)
        idxObsTimes_in_PredTimes = np.searchsorted(timesPredictors, timesObs)
        #
        #--------------------------------------------------------------------
        #
        # Transformation 1. 
        # Project predictors to observation-defined space of output variable
        #
        # list of intervals for the predictors, the whole set
        #
        predictor_interv_edges = np.zeros(shape=(nPredictors, self.nIntervals + 1))  # edges 
        predictors_transformed_all = np.zeros(shape=(nPredictors, nPredTimes, nSites),
                                                    dtype=np.float32) * np.nan
        LUT_outvar_4_ranges = np.zeros(shape=(nPredictors, self.nIntervals),
                                       dtype=np.float32) * np.nan
        LUTstd_outvar_4_ranges = np.zeros(shape=(nPredictors, self.nIntervals), dtype=np.float32) * np.nan
        LUT_02prc = np.zeros(shape=(nPredictors, self.nIntervals), dtype=np.float32) * np.nan
        LUT_98prc = np.zeros(shape=(nPredictors, self.nIntervals), dtype=np.float32) * np.nan
        #
        # Get intervals for the predictors. Each interval will be then projected to
        # the output variables
        for iPred, pred in enumerate(predictors):  #patient_data.predictors:
            predictor_interv_edges[iPred,:] = self.intervals(pred, pdata_predictors[iPred,:,:])
            #
            # A lookup table for projection of each predictor interval to output variable
            # get the indices of predictors that fall into the corresponding intervals
            # Note: searchsorted gives 0 only for exact array-minimum value(s), i.e. it 
            # belongs to the interval 0
            #
            indices_4_ranges = np.zeros(shape=(pdata_predictors[iPred,:,:].shape), dtype=np.int16)
            indices_4_ranges[:,:] = np.minimum(np.maximum(np.searchsorted(predictor_interv_edges[iPred,:], 
                                                                          pdata_predictors[iPred,:,:]) - 1, 
                                                          0), self.nIntervals-1)
            for iInt in range(self.nIntervals):
                idxRng = (indices_4_ranges == iInt)[idxObsTimes_in_PredTimes]
                if np.sum(np.isfinite(pdata_obs[idxRng])) > 0: 
                    # weighted mean
                    LUT_outvar_4_ranges[iPred,iInt] = spp.weighted_mean(pdata_obs[idxRng], 
                                                                        sample_weights[idxRng])
                    # for plain mean:
                    #LUT_outvar_4_ranges[iPred,iInt] = np.nanmean(pdata_obs[idxOK])
                    LUTstd_outvar_4_ranges[iPred,iInt] = np.nanstd(pdata_obs[idxRng])
                    LUT_02prc[iPred,iInt] = np.nanpercentile(pdata_obs[idxRng], 2)
                    LUT_98prc[iPred,iInt] = np.nanpercentile(pdata_obs[idxRng], 98)
            #
            # handle holes and out-of-range cases: extrapolate to left and right if start-end is missing
            # and linearly interpolate over the holes
            if np.any(np.isnan(LUT_outvar_4_ranges[iPred,:])):
                indValid = np.nonzero(np.isfinite(LUT_outvar_4_ranges[iPred,:]))[0]  # indices with valid values
                arValid = LUT_outvar_4_ranges[iPred, indValid]                           # valid values
                if len(indValid) < 1:
                    self.log.log('Failed LUT %s, zero valid indices' % predictors[iPred])
                    LUT_outvar_4_ranges[iPred,:] = 0.0
                else:
                    LUT_outvar_4_ranges[iPred,:] = np.interp(range(self.nIntervals),  # points where values are needed
                                                             indValid, arValid)   # points with values and the values 
            #
            # final output of transformation: predictors projected to observations
            #
            # What if predictor is constant? Interpolate only if there is 1e-3 variability
            #
            if predictor_interv_edges[iPred,0] < 0.999 * predictor_interv_edges[iPred,-1]:
                # Previous range: LUT[prev] < value
                indRangePrev = np.maximum(indices_4_ranges-1, 0)
                # Interpolate
                v_i_1 = LUT_outvar_4_ranges[iPred, indRangePrev]
                v_i = LUT_outvar_4_ranges[iPred, indices_4_ranges]
                Pr = pdata_predictors[iPred,:,:]
                Pr_i = predictor_interv_edges[iPred,:][indices_4_ranges]
                Pr_i_1 = predictor_interv_edges[iPred,:][indRangePrev]
    
                # This is better but causes warnings
    #            predictors_transformed_all[iPred,:,:] = np.where(indices_4_ranges == 0,
    #                                                             v_i,
    #                                                             ((v_i_1 * (Pr_i - Pr) + 
    #                                                               v_i * (Pr - Pr_i_1)) / 
    #                                                               (Pr_i - Pr_i_1)))
                predictors_transformed_all[iPred,:,:] = v_i
                predictors_transformed_all[iPred,:,:][indices_4_ranges > 0] = (
                                    (v_i_1 * (Pr_i - Pr) + v_i * (Pr - Pr_i_1))[indices_4_ranges > 0] / 
                                    (Pr_i - Pr_i_1)[indices_4_ranges > 0])
            else:
                # without interpolation, simply:
                predictors_transformed_all[iPred,:,:] = LUT_outvar_4_ranges[iPred, indices_4_ranges]
            #
            # Check for nans
            if np.any(np.isnan(LUT_outvar_4_ranges[iPred,:])):
                self.log.log('Nan in LUT for %s' % tsPoolPredict.variables[iPred])
                self.log.log('Observations: ' + str(pdata_obs))
                self.log.log('Intervals: ' + str(predictor_interv_edges[iPred,:]))
                self.log.log('Predictors:' + str(pdata_predictors[iPred,:,:]))
                self.log.log('indices for ranges:' + str(indices_4_ranges))
                self.log.log('LUT intervals' + str(LUT_outvar_4_ranges[iPred,:]))
                self.log.log('Predictors transformed:' + str(predictors_transformed_all[iPred,:,:]))
                self.log.log('######################################################################')
                raise ValueError   # This is a severe problem. Must never happen
            #
            # Check for nan-s
            if np.any(np.isnan(predictors_transformed_all[iPred,:,:])):
                self.log.log('Error: nans in predictors_transformed_all')
                self.log.log('Nbr of nans: %g, dimensions: %s' % 
                             (np.sum(np.isnan(predictors_transformed_all[iPred,:,:])),
                              str(predictors_transformed_all.shape)))
                for iTime in range(predictors_transformed_all.shape[1]):
                    if np.all(np.isfinite(predictors_transformed_all[iPred,iTime,:])): continue
                    self.log.log(valid_obstimes[iTime].strftime('NaN in predictors for %Y-%m-%d %H:%M'))
                    self.log.log('NaN for predictor %s, actual value = ' % predictors[iPred] +
                                 str(pdata_predictors[iPred,iTime,:]))
                    self.log.log('Projection intervals: ' + str(predictor_interv_edges[iPred,:]))
                    self.log.log('LUT: ' + str(LUT_outvar_4_ranges[iPred,:]))
#                    self.log.log('indices in LUT:' + str(indRangeFound[iTime,:]))
                raise ValueError   # This is a severe problem. Must never happen
            
            self.log.log('Predictor LUT: ' + predictors[iPred] + ' ' +
                         ' '.join(('%g' % vLUT for vLUT in LUT_outvar_4_ranges[iPred,:])))

        if self.iDebug > 0:
            self.DrawLUT(LUT_outvar_4_ranges, LUT_02prc, LUT_98prc, predictors, predictor_interv_edges,
                         self.output_variable, self.ID)
        
        #
        # Energy in projected predictors
        #
        self.log.log('Projected predictors: ' + ' '.join(predictors))
        predEnergy = list((np.square(predictors_transformed_all[iP,:,:]).mean() 
                           for iP in range(len(predictors))))
        self.log.log('Projected predictor energy:' + ' '.join(list(('%g' % E for E in predEnergy))))

#        #
#        # Check for nan-s
#        #
#        indNAN_predictors = np.any(np.isnan(predictors_transformed_all), axis=2)
#        indNAN_obs = np.any(np.isnan(pdata_obs), axis=1)
#        if np.any(indNAN_predictors):
#            print('indNAN_predictors and indNAN_obs: ', np.sum(indNAN_predictors), np.sum(indNAN_obs))
#            for iTime in range(predictors_transformed_all.shape[1]):
#                if not np.any(indNAN_predictors[iTime]): continue
#                self.log.log(valid_obstimes[iTime].strftime('NaN in predictors for %Y-%m-%d %H:%M'))
#                for iPred in range(nPredictors):
#                    if np.any(np.isnan(predictors_transformed_all[iPred,iTime,:])):
#                        self.log.log('NaN for predictor %s, actual value = ' % predictors[iPred] + 
#                                     str(pdata_predictors[iPred,iTime,:]))
#                        self.log.log('Projection intervals: ' + str(predictor_interv_edges[iPred,:]))
#                        self.log.log('LUT: ' + str(LUT_outvar_4_ranges[iPred,:]))
#            raise ValueError   # This is a severe problem. Must never happen

        #-----------------------------------------------------------------------
        #
        # Transformation 2
        # Orthogonalization of the projected datasets
        #
        # For sorting along the RMSE with de-biasing, use this one
        if True:
            bias = np.array(list((np.nanmean((predictors_transformed_all[iPred,idxObsTimes_in_PredTimes,:] - 
                                              pdata_obs[:,:])) 
                                  for iPred in range(nPredictors))))
    
            MSE = list((np.nanmean(np.square(predictors_transformed_all[iPred,idxObsTimes_in_PredTimes,:] - 
                                             pdata_obs[:,:] - bias[iPred])) # * times_kernel[:,None])
                        for iPred in range(nPredictors)))
            pred_ranking = np.argsort(np.array(MSE))   # indices that sort the RMSE array
            #
            # For sorting along the correlatoin use this one
            #
    #        corr = np.array(list((spp.nanCorrCoef(predictors_transformed_all[iMdl,:,:],
    #                                                  pdata_obs[:,:])
    #                                  for iMdl in range(nPredictors))))
    #        self.model_ranking = np.argsort(corr)[::-1]   # indices that sort corr array in DEscending order
            #
            # Ortogonalize the ranked predictors
            pdata_predictors_ort, ort_clf, ortQA = ortogonalise_TSM(predictors_transformed_all, 
                                                                    pred_ranking, 
                                                                    sample_weights,  # np.ones(shape=pdata_obs.shape), # kernel 
                                                                    idxObsTimes_in_PredTimes,  # tc_indices_mdl
                                                                    self.log)        # log 
        else:
            pred_ranking = np.array(list(range(nPredictors)))
            predInfoRanked = [1.0] * pred_ranking.shape[0] 
            pdata_predictors_ort = predictors_transformed_all
            ort_clf = None
            ortQA = 1.0

        self.log.log('Ordered predictors: ' + ' '.join(predictors[iP] for iP in pred_ranking))
        predEnergyRanked = list((np.square(pdata_predictors_ort[iP,idxObsTimes_in_PredTimes,:]).mean() 
                                 for iP in pred_ranking))
        self.log.log('Ordered predictor energy:' + ' '.join(list(('%g' % E 
                                                                  for E in predEnergyRanked))))
        predInfoRanked = list((np.nanmean(pdata_predictors_ort[iP,idxObsTimes_in_PredTimes,:] * pdata_obs) 
                               for iP in pred_ranking))
        self.log.log('Ordered predictor info:' + ' '.join(list(('%g' % v for v in predInfoRanked))))
        #
        #-----------------------------------------------------------------------------
        #
        # Transformation 3
        # Select only informative predictors in the ordered array
        #
        idxPredictorsInformative = pred_ranking[np.abs(predInfoRanked) > 
                                                0.05 * np.max(np.abs(predInfoRanked))]
        self.log.log('Informative predictors: %i: ' % len(idxPredictorsInformative) + 
                     ' '.join(predictors[iP] for iP in idxPredictorsInformative))
        #
        # Final predictor dimension of the problem, still full time coverage: obs + fcst
        # Predictors are ordered and filtered
        #
        pdata_pred_rdy = pdata_predictors_ort[idxPredictorsInformative,:,:]
        nPredictors = len(predictors[idxPredictorsInformative])
        #
        #-----------------------------------------------------------------------------
        # Transformation 4
        #
        # Data for the the time period covered by observations, split the fitting time series 
        # to training and test dataset. 
        #
        if stations_handling == 'stations_together':
            # split along times, once for all stations
            train_idx, test_idx = split_times(timesObs, sample_weights, self.training_fraction, 
                                              self.subset_selection, self.randomGen, self.log)
        elif stations_handling == 'stations_times_together':
            # time and stations are combined into one dimension and split together
            train_StTidx, test_StTidx = split_times_stations(pdata_pred_rdy, 
                                                             sample_weights * valObserv[:,iSt], 
                                                             self.training_fraction, 
                                                             self.subset_selection,
                                                             self.randomGen, self.log)
        else:
            self.log.log('Unknown station handling: %s' % stations_handling)
            return (processed_data(self.ID, flg_unknown),
                    (LUT_outvar_4_ranges, ort_clf),    # rules for transformations
                    ortQA)                           # QA for ortogonalization
        # We might use deterministic time split, then None is a legitimate answer
        if train_idx is None or test_idx is None:
            self.log.log('Failed train-test splitting')
            return (processed_data(self.ID, flg_unknown),
                    (LUT_outvar_4_ranges, ort_clf),    # rules for transformations
                    ortQA)                           # QA for ortogonalization
        #
        #------------------------------------------------------------------------------
        # Final cleaning
        #
        # Training and test datasets: days, transformed predictors, observations
        # Note that some of them can be trivial, all-zeroes, for instance
        # Remove them: thin-down the list of valis predictors and valid observations
        #
        indPredictorsOK = np.array(np.mean(np.std(pdata_pred_rdy[:,train_idx,:], axis=1),axis=1) > 
                                   np.mean(np.mean(pdata_pred_rdy[:,train_idx,:], axis=1), axis=1) * 1.0e-6)
        validPredictors = np.extract(indPredictorsOK, predictors[idxPredictorsInformative])
        nPredictors = len(validPredictors)
        if nPredictors < 1: 
            return (data_processed(self.ID, flg_noPredictors),
                    (LUT_outvar_4_ranges, ort_clf),    # rules for transformations
                    ortQA)                           # QA for ortogonalization
        # thin-down the predictor dataset
        # training / test datasets without trivialities 
        train_predictorsTr = pdata_pred_rdy[:,train_idx,:][indPredictorsOK,:,:]
        test_predictorsTr = pdata_pred_rdy[:,test_idx,:][indPredictorsOK,:,:]
        if self.iDebug > 0:
            print('All predictors, ranked:', predictors[idxPredictorsInformative])
            print('all_predictors_transformed:', pdata_pred_rdy.shape)
            print('prep: pdata_pred_rdy', pdata_pred_rdy)
            print(validPredictors)
            print('prep: indPredictorsOK',indPredictorsOK)
            print('prep: train_predictorsTr',train_predictorsTr)
            print('prep: test_predictorsTr',test_predictorsTr)
            print('prep: train_obs', pdata_obs[train_idx,:])
            print('prep: test_obs', pdata_obs[test_idx,:])
        #
        # Eliminate the nans from trining and tst datasets. For that, they have to be flattened
        # However, do not touch forecasting predictors
        #
        idxTrainObsOK = np.isfinite(pdata_obs[train_idx,:].ravel())
        idxTestObsOK = np.isfinite(pdata_obs[test_idx,:].ravel())
        idxTrainTimesOK = np.any(np.isfinite(pdata_obs[train_idx,:]),axis=1).ravel()
        idxTestTimesOK = np.any(np.isfinite(pdata_obs[test_idx,:]),axis=1).ravel()
        all_obstimes = np.repeat(timesObs[:, np.newaxis], pdata_obs.shape[1], axis=1)
        #
        # We return: (i) the bunch of processed data
        # (ii) rules of preprocesssing
        # (iii) quality of preprocessing
        # ATTENTION. For fitting, predictors must be the last index. Have to transpose.
        # 
        return (processed_data(self.ID, flg_OK, validPredictors,          # metadata 
#                               valid_obstimes[train_idx][idxTrainTimesOK],          # training times
                               all_obstimes[train_idx,:].ravel()[idxTrainObsOK],          # training times
                               train_predictorsTr.reshape((nPredictors,  # training predictors
                                                           len(train_idx) * nSites))[:,idxTrainObsOK].T, 
                               pdata_obs[train_idx,:].ravel()[idxTrainObsOK],       # training observations
                               sample_weights[train_idx,:].ravel()[idxTrainObsOK],  # training weights
#                               valid_obstimes[test_idx][idxTestTimesOK],          # test times
                               all_obstimes[test_idx,:].ravel()[idxTestObsOK],          # training times
                               test_predictorsTr.reshape((nPredictors, # test predictors
                                                          len(test_idx) * nSites))[:,idxTestObsOK].T,
                               pdata_obs[test_idx,:].ravel()[idxTestObsOK],       # test observations
                               sample_weights[test_idx,:].ravel()[idxTestObsOK],  # test kernel weights
                               times2fcst,               # time period covered by predictors
                               pdata_pred_rdy.reshape((nPredictors, # whole period, informative predictors, transformed 
                                                       len(times2fcst) * nSites)).T),
                prediction_rules(timeStep = timeStep, predictors = predictors,
                                 predictor_interv_edges = predictor_interv_edges, 
                                 LUT_outvar_4_ranges = LUT_outvar_4_ranges, 
                                 ortogonalization_clf = ort_clf,
                                 pred_ranking = pred_ranking, 
                                 idxPredictorsInformative = idxPredictorsInformative),
                ortQA)                           # QA for ortogonalization


    #======================================================================
    #
    # Preprocessing the input data without determining parameters of preprocessing:
    # use the pre-defined processing rules
    #
    def preprocess_input_data_ready_model(self, timesPredictors, valPoolPredict, predictors, rulesPred):
        # Input is 
        #     timesPredictors  # (nTimes) also times to forecast
        #     valPoolPredict,  # (nTimes, nStations, nPredictors) or list((nTimes, nStations))
        #     predictors       # names of predictors
        #     rulesPred          # rules of preprocessing: what, how, where to...
        # Output is the object of processed_data
        #    ID
        #    fcst_predictorsTr                        # the whole set of predictors, full coverage
        # Processing will include:
        # - stupidity checks
        # - projection of predictors to output variables
        # - orthogonalization
        # - reduction of dimensions for predictors with low added value
        # 
        # The nStations dimension is for spatially-distributed model, whcih has more than one 
        # observation per time. It is still a single model and single prediction made
        # for all stations. If someone wants them separate, call this class many times.
        # Prediction can be called straight after preprocessing.

        # Basic stupidity tests
        # We just have to make sure that predictors are the same as were when the odel was fitted
        #
        if predictors.shape[0] != rulesPred.predictors.shape[0]:
            self.log.log('Different sets of predictors. Fitted for %g, now given %g' % 
                         (predictors.shape[0], rulesPred.shape[0]))
            raise ValueError
        if not np.all(predictors == rulesPred.predictors):
            self.log.log('Predictors are not the same as were used for fiting:')
            for pr, rules_pr in zip(predictors, tulesPred.predictors):
                self.log.log('Fitting predictor: %s, given predictor: %s', (pr, rules_pr))
            raise ValueError
        #
        # Differential model? 
        # Then turn the time series into differences 
        #
        if rulesPred.mapping_type == 'DIFFERENTIAL':
            timesdiff = timesPredictors[1:] - timesPredictors[:-1]
            timeStep = np.min(timesdiff)
            if timeStep != rulesPred.timeStep:
                self.log.log('Timestep in predictors is not the same as in fitting data')
                self.log.log('Fitting timestep = %s, given timeStep = %s' % 
                             (str(rulesPred.timeStep), str(timeStep)))
                raise ValueError
            # There must be no holes in the data, those should be removed
            indValidTimes = timesdiff == timeStep
            # Now make the differential dat and observaations
            pdata_predictors = (valPoolPredict[:,1:,:] - valPoolPredict[:,:-1,:])[:,indValidTimes,:]
        else:
            # same names as wrappers
            pdata_predictors = valPoolPredict
        #
        # Intermediate dimensions of the problem
        #
        nPredictors = len(predictors)
        nSites = valPoolPredict.shape[2]
        if nSites > 1: print('Multi-cell case')
        nPredTimes = len(timesPredictors)
        #
        # Transformation 1. 
        # Project predictors to observation-defined space of output variable
        #
        # list of intervals for the predictors, the whole set
        #
        predictors_transformed_all = np.zeros(shape=(nPredictors, nPredTimes, nSites),
                                              dtype=np.float32) * np.nan
        #
        # Project the intervals midpoints to the observed phase space, each predictor
        for iPred in range(nPredictors):
            #
            # A lookup table for projection of each predictor interval to output variable
            # get the indices of predictors that fall into the corresponding intervals
            # Note: searchsorted gives 0 only for exact array-minimum value(s), i.e. it 
            # belongs to the interval 0
            indices_4_ranges = np.minimum(np.maximum(np.searchsorted(rulesPred.predictor_interv_edges[iPred,:], 
                                                                     pdata_predictors[iPred,:,:]) - 1, 
                                                     0), self.nIntervals-1)
            # predictors projected to observations
            #
            # What if predictor is constant? Interpolate only if there is 1e-3 variability
            #
            if rulesPred.predictor_interv_edges[iPred,0] < 0.999 * rulesPred.predictor_interv_edges[iPred,-1]:
                # Previous range: LUT[prev] < value
                indRangePrev = np.maximum(indices_4_ranges-1, 0)
                # Interpolate
                v_i_1 = rulesPred.LUT_outvar_4_ranges[iPred, indRangePrev]
                v_i = rulesPred.LUT_outvar_4_ranges[iPred, indices_4_ranges]
                Pr = pdata_predictors[iPred,:,:]
                Pr_i = rulesPred.predictor_interv_edges[iPred,:][indices_4_ranges]
                Pr_i_1 = rulesPred.predictor_interv_edges[iPred,:][indRangePrev]
    
                # This is better but causes warnings
    #            predictors_transformed_all[iPred,:,:] = np.where(indices_4_ranges == 0,
    #                                                             v_i,
    #                                                             ((v_i_1 * (Pr_i - Pr) + 
    #                                                               v_i * (Pr - Pr_i_1)) / 
    #                                                               (Pr_i - Pr_i_1)))
                predictors_transformed_all[iPred,:,:] = v_i
                predictors_transformed_all[iPred,:,:][indices_4_ranges > 0] = (
                                    (v_i_1 * (Pr_i - Pr) + v_i * (Pr - Pr_i_1))[indices_4_ranges > 0] / 
                                    (Pr_i - Pr_i_1)[indices_4_ranges > 0])
            else:
                # without interpolation, simply:
                predictors_transformed_all[iPred,:,:] = rulesPred.LUT_outvar_4_ranges[iPred, indices_4_ranges]
            #
            # Check for nan-s
            if np.any(np.isnan(predictors_transformed_all[iPred,:,:])):
                for iTime in range(predictors_transformed_all.shape[1]):
                    if np.all(np.isfinite(predictors_transformed_all[iPred,iTime,:])): continue
                    self.log.log(timesPredictors[iTime].strftime('NaN in predictors for %Y-%m-%d %H:%M'))
                    self.log.log('NaN for predictor %s, actual value = ' % predictors[iPred] +
                                 str(pdata_predictors[iPred,iTime,:]))
                    self.log.log('Projected values:' + str(predictors_transformed_all[iPred,iTime,:]))
                    self.log.log('Projection intervals: ' + str(rulesPred.predictor_interv_edges[iPred,:]))
                    self.log.log('LUT: ' + str(rulesPred.LUT_outvar_4_ranges[iPred,:]))
                    self.log.log('indices in LUT:' + str(indRangePrev[iTime,:]))
                raise ValueError   # This is a severe problem. Must never happen
        #
        # Energy in projected predictors
        #
        self.log.log('Projected predictors: ' + ' '.join(predictors))
        predEnergy = list((np.square(predictors_transformed_all[iP,:,:]).mean() 
                           for iP in range(len(predictors))))
        self.log.log('Projected predictor energy:' + ' '.join(list(('%g' % E for E in predEnergy))))

        #
        # Transformation 2
        # Orthogonalization of the projected datasets
        #
        if rulesPred.ort_clf is None:
            return  processed_data(self.ID, flg_OK, predictors[rulesPred.idxPredictorsInformative], 
                                   times2fcst = timesPredictors, 
                                   all_predictorsTr = predictors_transformed_all)
        #
        # Ortogonalize the ranked predictors
        # Here we still have full list of predictors, filtering the non-informatiove ones comes below
        #
        pdata_predictors_ort = apply_ortogonalization_TSM(predictors_transformed_all,
                                                          rulesPred.ort_clf,
                                                          rulesPred.pred_ranking,
                                                          self.log)
        #
        # A bit of reporting and information
        self.log.log('Ordered predictors: ' + ' '.join(predictors[iP] for iP in rulesPred.pred_ranking))
        predEnergyRanked = list((np.square(pdata_predictors_ort[iP,:,:]).mean() 
                                 for iP in rulesPred.pred_ranking))
        self.log.log('Ordered predictor energy:' + ' '.join(list(('%g' % E 
                                                                  for E in predEnergyRanked))))
        #
        # Transformation 3
        # Report the informative predictors in the ordered array
        #
        self.log.log('Informative predictors: %i: ' % len(rulesPred.idxPredictorsInformative) + 
                     ' '.join(predictors[iP] for iP in rulesPred.idxPredictorsInformative))
        #
        # Transformation 4: split to training and test subsets i irrelevant here
        #
        # We return the processed data
        # 
        return  processed_data(self.ID, flg_OK, 
                               predictors[rulesPred.idxPredictorsInformative], 
                               times2fcst = timesPredictors, 
                               all_predictorsTr = 
                                        pdata_predictors_ort[rulesPred.idxPredictorsInformative,:,:].
                                        reshape((len(rulesPred.idxPredictorsInformative),
                                                 len(timesPredictors) * pdata_predictors_ort.shape[2])).T)

    #=====================================================================
    #
    # The DUMMY symptom model: dummy values for all symptoms
    #
    def dummy_model(self, day_0, times2fcst, dpp=None, ifDrawDumpAll=False):
        #
        # Does not use any input, generates random output
        #
        if dpp is None: 
            ID = 'Dummy_patient'
            predictors = ['dummy_predictor']
        else: 
            ID = dpp.ID
            predictors = dpp.validPredictors
        return (MyTimeVars.TsMatrix(times2fcst, [ID], predictors, 
                                    self.randomGen.rand(len(times2fcst)), None),
                (None, 'dummy')) 

 
    #=====================================================================
    #
    # The PERSISTENCE symptom model: the last level of symptoms is repeated in all further days
    #
    def persistence_model(self, day_0, times2fcst, dpp, ifDrawDumpAll=False):
        #
        # Repeats the last-observed value for all forecsting dates
        #
        # Find the last observed valueclosest to the start of forecast
        times = np.append(dpp.train_days. dpp.test_days)
        obs = np.append(dpp.train_outvars, dpp.test_outvars)
        argSortTimes = np.argsort(times)
        indNow = bisect.bisect_left(times[argSortTimes], times2fcst[0])
        if indNow == len(times[argSortTimes]):
            last_obs_date_idx = -1
        elif times2fcst[0] == times[argSortTimes][indNow]:
            last_obs_date_idx = indNow
        else:
            last_obs_date_idx = indNow-1
        if (times2fcst[0] - times[argSortTimes][last_obs_date_idx]).days < 3:   # close enough, can use as persistence approach
            return (MyTimeVars.TsMatrix(times2fcst, [dpp.ID], dpp.validPredictors, 
                                        np.ones(shape=(len(times2fcst)), dtype=np.float32) * 
                                        obs[argSortTimes][last_obs_date_idx], None),
                    (None,'persistence'))
        else:
            return (MyTimeVars.TsMatrix(times2fcst, [dpp.ID], dpp.validPredictors, 
                                        np.ones(shape=(len(times2fcst)), dtype=np.float32) * np.nan, None),
                    (None,'persistence'))


    #=====================================================================
    #
    # Static mapping the predictors to output variables, whatever they are
    #
    # Structure of the input datset is decided by the data_processed class above
    # The procedure returns the tsMatrix object
    # 
    #
    # dpp is from class processed_data:
    # def __init__(self, ID, quality_flag, validPredictors=None,  
    #             train_days=None, train_predictorsTr=None, train_outvars=None,
    #             test_days=None, test_predictorsTr=None, test_outvars=None, fcst_predictorsTr=None):
    #    self.ID = ID
    #    self.quality_flag = quality_flag 
    #    if quality_flag != flg_OK: return # do not bother if something is wrong
    #    self.validPredictors = validPredictors
    #    self.train_days = train_days
    #    self.train_predictorsTr = train_predictorsTr
    #    self.train_outvars = train_outvars
    #    self.test_days = test_days
    #    self.test_predictorsTr = test_predictorsTr
    #    self.test_outvars = test_outvars
    #    self.fcst_predictorsTr = fcst_predictorsTr
    #
    # The data dimensions are as in tsMatrices: (nTimes, nStations, nPredictors)
    # and (nTimes, nStations) for observations. The stations are treated together as a multitude
    # of measurements for a single time. The model will be just one.
    #
    def fit_static_mapping(self, dpp):
        #
        # Stupidity test
        if dpp.quality_flag != flg_OK:
            self.log.log('static_mapping received bad quality flag in input:' + dpp.quality_flag) 
            return
        #
        # dpp stands for data pre-processed. Will be used below as a set of pointers (hopefuly)
        #
        nPredictors = len(dpp.validPredictors)
        #
        # Patient statistic
        #
        self.log.log('ID %s: %g/%g training/test times, %g valid predictors' % 
                     (dpp.ID, len(dpp.train_days), len(dpp.test_days), nPredictors))
        # Prepare reporting
        self.log.log("PREDICTORS " + dpp.ID + ' ' + 
                     " ".join([str(x) for x in dpp.validPredictors])) #AQF.available_species]))
        self.log.log('SCORES. model pID target_var n_train n_test r_train r_test nCoefs intercept  [coefs]')
        #
        # Multilinear regression. Make use of all models, then choose the best one
        #
        clf = {}
#        train_outvars_predicted = np.zeros(shape=(len(dpp.train_days)), dtype=np.float32)
        test_outvars_predicted = {}
        rBestMdl = (-2.0, '')
        #
        # Note that now we deal only with non-trivial predictors and non-trivial symptoms
        #
        for chMdl in ['RIDGE']: #,'LASSO']:   #'MLR', 
#            print(chMdl)
#            test_outvars_predicted[chMdl] = np.zeros(shape=(len(dpp.test_days)), dtype=np.float32)

            if self.iDebug > 0:
                print(dpp.ID)
                print('dpp.train_predictorsTr', dpp.ID, dpp.train_predictorsTr,'\nstdev:',
                      np.std(dpp.train_predictorsTr,axis=0))
                print('dpp.train_outvars', dpp.ID, dpp.train_outvars[:])
                print('dpp.test_predictorsTr', dpp.ID, dpp.test_predictorsTr)
                print('dpp.test_outvars', dpp.ID, dpp.test_outvars[:])
            
            # fitting...
            # note the optimal criterion function: can be correlation (corr4min) or 
            # RMSE (rmse4min). Mind that all functions minimise
            # Optimization requires predictor array (nTimes, nPredictors), just as in tsPools with 
            # flattened station-time dimensions
            #, metric2miimise, AlphaMaxIter, log
            clf[chMdl] = linregr_regularised_model(chMdl, self.metric2minimise, 100, self.log)
            clf[chMdl].get_regulariser_weight_and_fit(dpp.train_predictorsTr, dpp.train_outvars, 
                                                      dpp.train_kernel,
                                                      dpp.test_predictorsTr, dpp.test_outvars, 
                                                      dpp.test_kernel)
            # Making prediction...
#                print('\n\n\n\n\n', clf[chMdl][organ].predict(dpp.train_predictorsTr.T))
            train_outvars_predicted = clf[chMdl].predict(dpp.train_predictorsTr)
            test_outvars_predicted[chMdl] = clf[chMdl].predict(dpp.test_predictorsTr)
            # Model formal skills: correlation coefficient for test subset
            # Note that some of those will become NAN if any of the standard
            # deviations is 0.
            stdev_train = np.std(dpp.train_outvars[:]) 
            stdev_test = np.std(dpp.test_outvars[:])
            stdev_train_pred = np.std(train_outvars_predicted[:]) 
            stdev_test_pred = np.std(test_outvars_predicted[chMdl][:]) 
            if stdev_train < 1e-6 or stdev_train_pred < 1e-6: r_train = 0.0
            else:
                r_train = np.corrcoef(train_outvars_predicted[:], 
                                      dpp.train_outvars[:])[0,1]
            if stdev_test < 1e-6 or stdev_test_pred < 1e-6: r_test = 0.0
            else:
                r_test = np.corrcoef(test_outvars_predicted[chMdl][:], 
                                     dpp.test_outvars[:])[0,1]
                if np.isnan(r_test):
                    print('Nan test corr, r_train/r_test = %g/%g, bestCorr = %g:' % 
                          (r_train, r_test, rBestMdl[0]))
                    print('Training:', train_outvars_predicted[:], dpp.train_outvars[:])
                    print('Test:', test_outvars_predicted[chMdl][:], dpp.test_outvars[:])
                    print('')
                    raise ValueError
            if r_test > rBestMdl[0]: rBestMdl = (r_test, chMdl)
            self.log.log("%s %s %g %g %g %g %g %s %s" % (chMdl, dpp.ID,  
                                                         len(dpp.train_outvars[:]), 
                                                         len(dpp.test_outvars[:]),
                                                         r_train, r_test, nPredictors, 
                                                         clf[chMdl].clf.intercept_, 
                                                         " ".join([str(x) for x in clf[chMdl].clf.coef_])))
            if self.iDebug:
                self.DrawDump(dpp.train_days, dpp.train_outvars, train_outvars_predicted,
                              dpp.test_days, dpp.test_outvars, test_outvars_predicted[chMdl],
                              dpp.ID, chMdl, dpp.ID, r_train, r_test)
                
        # anything useful?
        if rBestMdl[1] == 'None': self.log.log('All models failed')
        else: self.log.log('Best model skills %s: %s %g' % (dpp.ID, rBestMdl[1], rBestMdl[0]))
        
        return  (clf[rBestMdl[1]],  rBestMdl)   # the best model, its skills and name
        

        
    ####################################################################################

    def predict_static_mapping(self, dpp, clf):
        #
        # Called after input data pre-processor and model fitting, which determine 
        # dpp and clf, respectively
        #
        fcst_outvars_predicted = clf.predict(dpp.all_predictorsTr)
        if not self.iDebug is None:
            if self.iDebug > 1:
                for iDay in range(len(dpp.times2fcst)):
                    self.log.log('Forecasted for %s %g' % (dpp.times2fcst[iDay].strftime('%Y%m%d'), 
                                                           fcst_outvars_predicted[iDay]))
        return fcst_outvars_predicted



    ######################################################################################
    
    def fit_differential(self, day_0, dates, ID, tser_input):
        patient_data = ID               # new alias of the PatientData class
        patient_fc_data = ID
        print ("=============== Patient No %s   started..... " % ID) 
        ### default option assumes the availability of all AQ data and predictors construction is based on the length of symptom records only
        ### in testing mode the AQ data are shorter, thus additional list of AQ available dates is needed 
        delta_days_aq = []  
        delta_days_organs = {}
        result_symptoms = {}                

        ifAqTest = True
        ifPatientValid = False
        ifVerification = True
        
        aq_species = 'NO2 PM2_5 SO2 NO2_max PM2_5_max SO2_max'.split()
        organsss = 'eye nos lun'.split()               
        patient_data.days = tser_input[sorted(tser_input.keys())[0]][-1].times()    ### days are read from ['NO2'] fields
        patient_fc_data.days = sorted(dates)
                    
        for organ in organsss:
            if organ in tser_input.keys():
                delta_days_organs[organ] = [] 
                for day in tser_input[organ][0].times():     #patient_data.days[0:-3]:     ### [day_0,today] - [two nearest forecasts]
                    if (day+dt.timedelta(days=1)) in tser_input[organ][0].times(): #patient_data.symptoms[organ]:
                        if day not in delta_days_organs[organ]:
                            delta_days_organs[organ].append(day)
                    else: 
                        #print "information from day %s is not available in symptom time series to compute the increment for %s, patient No %s ..." % (day+dt.timedelta(days=1), organ, ID)
                        continue
        if (len(delta_days_organs['eye'])>3 or len(delta_days_organs['nos'])>3 or len(delta_days_organs['lun'])>3) and len(patient_data.days)>3:
            ifPatientValid = True
        else:
            print ("****************    patient No %s  have not enough information, result_symptoms = -999" % ID)
            for organ in organsss: 
                result_symptoms[organ] = {}
                for d in dates: result_symptoms[organ][d] = -999
            return symptoms(ID, day_0, dates, result_symptoms, patients.flg_OK)

        if ifPatientValid:
            patient_verif = ID
            for organ in organsss:
                patient_data.predictors[organ] = {}
                patient_fc_data.predictors[organ] = {}
                patient_verif.forecasts[organ] = {} ### preparing dic for verification >>> patient_verif.forecasts[organ][num_fc_day][calendar_day:value on this day]
                patient_verif.days = patient_fc_data.days
                for i in range(len(dates)): 
                    patient_verif.forecasts[organ][i] = {}
                    #patient_verif.forecasts[organ][i] = patient_fc_data.days[i]
                   
                for var in tser_input.keys():
                    if var in AQF.available_species:
#                    if var in aq_species: #AQF.available_species:
                        patient_data.predictors[organ][var] = {}
                        patient_fc_data.predictors[organ][var] = {}
                        for day in delta_days_organs[organ]: 
                            if day in patient_data.days and (day+dt.timedelta(days=1)) in patient_data.days:
                                patient_data.predictors[organ][var][day] = tser_input[var][-1][day+dt.timedelta(days=1)] - tser_input[var][-1][day] # [next] minus [prev/current] in historical data range
                                if day not in delta_days_aq:
                                    delta_days_aq.append(day)    
                            else:
                                continue
                        for fc_days in patient_fc_data.days:
                            if tser_input[var][-1][fc_days]:
                                try:
                                    patient_fc_data.predictors[organ][var][fc_days] = tser_input[var][-1][fc_days]-tser_input[var][-1][fc_days-dt.timedelta(days=1)]
                                except:
                                    print ("Axxxxxx: New patient with no AQ/Symptom history, only one day report --> var %s for day %s is not found in field [-1]" % (var, (fc_days-dt.timedelta(days=1))))
                            else:
                                patient_fc_data.predictors[organ][var][fc_days] = tser_input[var][-1][fc_days-dt.timedelta(days=1)]-tser_input[var][-1][fc_days-dt.timedelta(days=2)]    ### replacing nonexistent data with data from previous day
                    else: continue        
    ###----------------------------------######## created dictionary[patient_data.predictors] which consist of the delta-concentrations  of previous days
      
            
            for organ in patients.organs:
                if ifAqTest:
                    delta = delta_days_aq
                else:
                    delta = delta_days_organs[organ]
        
                for day in delta:
                    hist_delta_symptoms = {}              ### symptom's delta in the past (until day_0,today)
                    #for organ in patients.target_organs:
                    if organ in organsss:
                        try:
                            hist_delta_symptoms[organ]= tser_input[organ][0][day+dt.timedelta(days=1)] - tser_input[organ][0][day]
                        except:
                            pass
    
    ###            correction  of symptom data by medicine
                        med = 0
                        if "med" in tser_input.keys():
                            if day in tser_input["med"][0].times(): med = tser_input["med"][0][day]
                        real_symptoms = [key for key in hist_delta_symptoms.keys() if hist_delta_symptoms[key] > 0]
                        if med != 0 and len(real_symptoms) > 0:
                            for key in real_symptoms:
                                hist_delta_symptoms[key] = int(round(hist_delta_symptoms[key] + float(med) / len(real_symptoms)))   ### ???? what is the "med" range ? more intensify the symptom's delta?
                        for key in hist_delta_symptoms.keys():
                            if not (key in patient_data.symptoms):
                                patient_data.symptoms[key] = {}
                            patient_data.symptoms[key][day] = hist_delta_symptoms[key]
    ###-------------------------------------- ######## created dictionary[patient_data.symptoms] which consist of the delta-symptoms  from previous days
          
            for organ in organsss:  # patient_data.symptoms:
                if not organ in patient_data.symptoms.keys(): continue
                
                if ifAqTest:
                    DeltaDays = delta_days_aq
                else:
                    DeltaDays = delta_days_organs[organ]
                
                model_indices = sorted(range(len(DeltaDays)))
                test_indices = sorted(range(len(patient_fc_data.days)))
                model_predictors = {}
                model_symptoms = {}   
                test_predictors = {}
                test_symptoms = {}
                
                for key in patient_data.predictors[organ]:
                    model_predictors[key] = []     
                    for i in model_indices:
                        model_predictors[key].append(patient_data.predictors[organ][key][DeltaDays[i]])
                    test_predictors[key] = []
                    for j in test_indices:
                        test_predictors[key].append(patient_fc_data.predictors[organ][key][dates[j]])
                               
                model_symptoms[organ] = []
                test_symptoms[organ] = []
                for i in model_indices:
                    try:
                        model_symptoms[organ].append(patient_data.symptoms[organ][DeltaDays[i]])
                    except:
                        print ('AAAAAAAAAAAAAAAAAAAAAAAAAAAAA')
    
                ### ============   construction of multilinear regression model for every organ
                
                lm = linear_model.LinearRegression()
                sorted_keys = sorted(patient_data.predictors[organ].keys())
                x_test = [[test_predictors[key][j] for key in sorted_keys] for j in test_indices]
                y_train = [model_symptoms[organ][i] for i in model_indices]
                x_train = [[model_predictors[key][i] for key in sorted_keys] for i in model_indices]
                
                linregr_model = lm.fit(x_train, y_train)
                test_symptoms[organ] = lm.predict(x_test)
                
                #===================================================================
                # print 'test data ...', x_test
                # print 'predictions.....', test_symptoms[organ]
                # print 'lm.score...', lm.score(x_train, y_train)
                # print 'lm.coef_ ....', lm.coef_
                # print 'lm.intercept_......',lm.intercept_
                #===================================================================
                #===================================================================
                # mpl.pyplot.scatter(x_train[:,1], y_train, color= 'blue')
                # mpl.pyplot.scatter(x_train[:,0], y_train, color= 'green')
                # mpl.pyplot.plot(test_symptoms[organ], 'r.-')
                # #py.plot(x_test[:,1], predictions, 'r.-')
                # #py.plot(x_test[:,0], predictions, 'g.-')
                # mpl.pyplot.show()
                #===================================================================
            ###### construction of result forecast by adding computed [deltas] to the known symptoms 
                result_symptoms[organ] = {}
                if (day_0-dt.timedelta(days=1)) in tser_input[organ][0].times():
                    result_symptoms[organ][day_0] = tser_input[organ][0][day_0-dt.timedelta(days=1)] + test_symptoms[organ][0]
                    #print "yesterday report (%s) for %s is available, value is %s " %((day_0-dt.timedelta(days=1)),organ,tser_input[organ][0][day_0-dt.timedelta(days=1)])
                    #print "computation day_0 ...: %s + %s " %(tser_input[organ][0][day_0-dt.timedelta(days=1)], test_symptoms[organ][0])
                    
                else:
                    result_symptoms[organ][day_0] = tser_input[organ][0][delta_days_organs[organ][-1]] + test_symptoms[organ][0]
                    #result_symptoms[organ][day_0] = tser_input[organ][0][day_0-dt.timedelta(days=2)] + test_symptoms[organ][0]
                    #print "NO yesterday report, picked up %s value for %s from LAST reported day %s  " %(tser_input[organ][0][delta_days_organs[organ][-1]],organ,delta_days_organs[organ][-1])    
                    #print "computation day_0 ...: %s + %s " %(tser_input[organ][0][delta_days_organs[organ][-1]], test_symptoms[organ][0])
                for iday in range(1,len(patient_fc_data.days)):
                    #print 'iday.....', iday
                    #print 'len(patient_fc_data.days).....', len(patient_fc_data.days)
                    dd = patient_fc_data.days[iday]
                    #print 'dd .....', dd
                    #print 'result_symptoms[organ][dd-dt.timedelta(days=1)]...........', result_symptoms[organ][dd-dt.timedelta(days=1)]
                    #print 'test_symptoms[organ][iday-1]............', test_symptoms[organ][iday-1]
                    result_symptoms[organ][dd] = result_symptoms[organ][dd-dt.timedelta(days=1)] + test_symptoms[organ][iday-1]
                    #print "computation....: %s + %s " %(result_symptoms[organ][dd-dt.timedelta(days=1)], test_symptoms[organ][iday-1])
                    #print " final fc symptom for %s is result_symptoms[organ][dd] ..... %s" %(organ, result_symptoms[organ][dd])
                    if result_symptoms[organ][dd] < 0: 
                        result_symptoms[organ][dd] = 0.0
                    #    print " final ZERO fc symptom for %s is  %s" %(organ, result_symptoms[organ][dd])
                    else:
                        continue                             
            print (" <-----> Done for patient No  %s  ..." % ID)
            return symptoms(ID, day_0, dates, result_symptoms, patients.flg_OK)

    ####################################################################################

    def predict_differential_mapping(self, dpp, clf):
        #
        # Called after input data pre-processor and model fitting, which determine 
        # dpp and clf, respectively
        #
        raise NotImplemented 
        # Something like this:
        times = np.append(dpp.train_days. dpp.test_days)
        obs = np.append(dpp.train_outvars, dpp.test_outvars)
        argSortTimes = np.argsort(times)
        indNow = bisect.bisect_left(times[argSortTimes], times2fcst[0])
        fcst_outvars_predicted = obs[argSortTimes] + clf.predict(dpp.all_predictorsTr)
        if self.iDebug > 1:
            for iDay in range(len(dpp.times2fcst)):
                self.log.log('Forecasted for %s %g' % (dpp.times2fcst[iDay].strftime('%Y%m%d'), 
                                                       fcst_outvars_predicted[iDay]))
        return fcst_outvars_predicted


    ###########################################################################################
    #
    # The distributor of the tasks: calls the model pointed out by the symptom_method
    # and generates the needed forecast for one patient
    #
    def fit_and_predict(self, valObserved, valPredict, sample_weights, times2fcst, timesObs, 
                        timesPredictors, predictors, stations_handling):
        #
        # Trivial prediction methods do not require transformation of the data
        #
        if self.mapping_type == 'DUMMY':
            return self.dummy_model(times2fcst[0], times2fcst)
        #
        # Non-trivial options require pre-processing, which returns preprocessed data, rules 
        # for repeating it if needed and QA
        #
        # STATIC or OPTIMAL require static model:
        #
        if self.mapping_type == 'OPTIMAL' or self.mapping_type == 'STATIC_MAPPING':
            # OPTIMAL means that we choose between static, differential and persistence forecast
            # static input data
            dppStat, rulesStat, QAStatPP = self.preprocess_input_data(times2fcst, timesObs, 
                                                                      timesPredictors,
                                                                      valObserved, valPredict,
                                                                      sample_weights, predictors, 
                                                                      False,   # ifDifferential 
                                                                      stations_handling)
            if dppStat.quality_flag != flg_OK:
                self.log.log('Static preprocessing failed: ' + dppStat.quality_flag)
                return None
            #
            # static fitting
            clfStat, QAStatFit = self.fit_static_mapping(dppStat)  # data, ifDrawDump
            #
            # and prediction
            predictStat = self.predict_static_mapping(dppStat, clfStat)
        #
        # DIFFERENTIAL or OPTIMAL require Differential model
        #
        if self.mapping_type == 'OPTIMAL' or self.mapping_type == 'DIFFERENTIAL':
            #
            # preprocessing
            dppDiff, rulesDiff, QADiff = self.preprocess_input_data(times2fcst, timesObs, 
                                                                    timesPredictors, 
                                                                    valObserved, valPredict,
                                                                    sample_weights, predictors, 
                                                                    True,   # ifDifferential
                                                                    stations_handling)
            if dppDiff.quality_flag != flg_OK:
                self.log.log('Differential preprocessing failed: ' + dppDiff.quality_flag)
                return None
            #
            # Differential fitting
            clfDiff, QADiff = self.fit_differential(dppDiff, False)
            #
            # and prediction
            predictDiff = self.predict_differential_mapping(dppDiff, transfDiff)
        #
        # PERSISTENCE or OPTIMAL require persistence
        #
        if self.mapping_type == 'OPTIMAL' or self.mapping_type == 'PERSISTENCE':
            #
            # no preprocessing and basically no fitting, just quality assessment
            clfPersist, QAPersistPP = self.persistence_model(times2fcst, dpp_static, False)
            # 
            # Prediction
            predictPersist, QAPersistFit = self.predict_persistence(dppPersist, None)
        #
        # OPTIMAL searches for the best fit
        #
        if self.mapping_type == 'OPTIMAL':
            #
            # Get the winner
            mapping_type = ['STATIC_MAPPING', 'DIFFERENTIAL',
                            'PERSISTENCE'][np.argmax([QAStat[0], QADiff[0], QAPersist[0]])]
        else: 
            mapping_type = self.mapping_type
        #
        # Return: prediction, quality of fit and rules for prediction
        #
        if mapping_type == 'STATIC_MAPPING':
            return (predictStat, 
                    (QAStatPP, QAStatFit), 
                    rulesStat.update(mapping_type = mapping_type, main_clf = clfStat))

        elif mapping_type == 'DIFFERENTIAL':
            return (predictDiff, 
                    (QADiffPP,QADiffFit), 
                    rulesDiff.update(mapping_type = mapping_type, main_clf = clfDiff))

        elif mapping_type == 'PERSISTENCE':
            return (predictPersist, (QAPersistPP,QAPersistFit), None)
        
        #
        # If we are here, none of the above has worked out...
        #
        spp.fatal(self.log, 'Unknown forecasting method: ' + self.mapping_type, 
                  'mapping_models__make_symptom_fcst')



    #########################################################################
    
    def predict(self, rulesPrediction, day_0, timesPredictors, valPoolPredict):
        #
        # Trivial prediction methods do not require transformation of the data
        #
        if rulesPrediction.mapping_type == 'DUMMY': return self.dummy_model(day_0, timesPredictors)
        if rulesPrediction.mapping_type == 'PERSISTENCE': 
            self.log.log('Persistence cannot be predicted without fitting')
            return None
        #
        # Non-trivial options require pre-processing
        # OPTIMAL fitting is not available in this regime
        #
        # STATIC model
        #
        if rulesPrediction.mapping_type == 'STATIC_MAPPING':
            # static input data
            dppStat = self.preprocess_input_data_ready_model(timesPredictors, 
                                                             valPoolPredict, 
                                                             rulesPrediction.predictors,
                                                             rulesPrediction)
            if dppStat.quality_flag != flg_OK:
                self.log.log('Static ready-model preprocessing failed: ' + dppStat.quality_flag)
                return None

            return self.predict_static_mapping(dppStat, rulesPrediction.main_clf)
        #
        # DIFFERENTIAL model
        #
        if rulesPrediction.mapping_type == 'DIFFERENTIAL':
            # preprocessing is same, rules say if it is differential
            dppDiff = self.preprocess_input_data_ready_model(timesPredictors, 
                                                             valPoolPredict, 
                                                             rulesPrediction.predictors,
                                                             rulesPrediction)
            if dppDiff.quality_flag != flg_OK:
                self.log.log('Differential preprocessing failed: ' + dppDiff.quality_flag)
                return None
            return self.predict_differential_mapping(dppDiff, rulesPrediction.main_clf)
        #
        # If we are here, none of the above has worked out...
        #
        spp.fatal(self.log, 'Unknown forecasting method: ' + rulesPrediction.mapping_type, 
                  'mapping_models__make_symptom_fcst')


    #------------------------------------------------
    
    def get_method(self):
        return self.mapping_type

    
    #------------------------------------------------

    def get_minimum_sample_size(self):
        if self.mapping_type == 'DUMMY':
            return None
        elif self.mapping_type == 'PERSISTENCE':
            return 3   #510
        elif self.mapping_type == 'STATIC_MAPPING':
            return self.minStaticMapSample
        elif self.mapping_type == 'INCREMENTAL':
            return self.minIncrementSample
        else:
            spp.fatal(self.log, 'Unknown mapping method: ' + self.mapping_type, 
                      'mapping_models__get_minimum_sample_size')


    #------------------------------------------------

    def DrawDump(self, train_days, train_outvars, train_outvars_predicted,
                 test_days, test_outvars, test_outvars_predicted,
                 ID, chMdl, outvariable, r_train, r_test):
        # A quick drawer for training and test subsets, both observed and predicted 
        spp.ensure_directory_MPI(os.path.join(self.wrkDir,"outvars_output"))
        #
        # Printing to zip? Then dictionary must exist
        if not self.dicDumpZip is None:
            if not 'outvars' in self.dicDumpZip.keys():    # valid zip in the dictionary?
                self.dicDumpZip['outvars'] = ZipFile(os.path.join(self.wrkDir,"outvars_output", 
                                                                  '%s_%s_%s_etal.zip_tmp' % 
                                                                  (chMdl, outvariable, ID)),'w')
        # draw
        fig, axs = plt.subplots(2,1, figsize=(10,10))
        plPredTrain = axs[0].plot(train_days, train_outvars_predicted, 'b-', 
                                  label='training, modelled')
        plRepTrain = axs[0].plot(train_days, train_outvars, 'r', linewidth=0, marker='.', markersize=2,
                                 label='training, reported')
        axs[0].legend(loc='upper left')
        axs[0].set_title("Correlation training: %s, %s %g" % (chMdl, outvariable, r_train))
        plPredTrain = axs[1].plot(test_days, test_outvars_predicted, 'b-', label='test, modelled',
                                  linewidth=1)
        plRepTrain = axs[1].plot(test_days, test_outvars, 'g', linewidth=0, marker='.', markersize=2,
                                 label='test, reported')
        axs[1].legend(loc='upper left')
        axs[1].set_title("Correlation test: : %s, %s %g" % (chMdl, outvariable, r_test))
        axs[0].tick_params(axis='x', rotation=20)
        axs[1].tick_params(axis='x', rotation=20)
        #
        # print to file or common zip archive?
        if self.dicDumpZip is None:
            plt.savefig(os.path.join(self.wrkDir,"outvars_output", '%s_%s_%s.png' % 
                                     (chMdl, outvariable, ID)), dpi=200)
        else:
            streamOut = io.BytesIO()
            plt.savefig(streamOut, dpi=200) #, bbox_inches='tight')
            with self.dicDumpZip['outvars'].open('%s_%s_%s.png' % (chMdl, outvariable, ID), 'w') as pngOut:
                pngOut.write(streamOut.getbuffer())
            streamOut.close()
        plt.clf()
        plt.close()
        print(os.path.join(self.wrkDir,"outvars_output"))
        return


    #------------------------------------------------

    def DrawLUT(self, LUT, LUT_02prc, LUT_98prc, predictors, interval_edges, variableNm, chTitle):
        #
        # Draws a multi-plot chart for projection lookup table
        #
        spp.ensure_directory_MPI(os.path.join(self.wrkDir,"LUT_output"))
        #
        # Printing to zip? Then dictionary must exist
        if not self.dicDumpZip is None:
            if not 'LUT' in self.dicDumpZip.keys():    # valid zip in the dictionary?
                self.dicDumpZip['LUT'] = ZipFile(os.path.join(self.wrkDir,"LUT_output", 
                                                              'LUT_' + chTitle + '_etal.zip_tmp'),'w')
        nPred = LUT.shape[0]
        nRows = int(np.floor(np.sqrt(nPred)))
        nCols = int(np.ceil(nPred / nRows))
#        x = range(LUT.shape[1])
        fig, axs = plt.subplots(nRows, nCols, figsize=(nCols * 4, nRows * 4))
        
        iPred = 0
        for iR in range(nRows):
            for iC in range(nCols):
                if iPred == nPred: break
                x = (interval_edges[iPred, :-1] + interval_edges[iPred, 1:]) / 2.
                ax = axs[iR, iC]
#                ax.errorbar(x,LUT[iPred,:], yerr = LUTstd[iPred,:])
                ax.vlines(x, LUT_02prc[iPred,:], LUT_98prc[iPred,:], color='blue', linestyles='-')
                ax.plot(x,LUT[iPred,:],'o')
                ax.set_title(predictors[iPred])
                if iC == 0: ax.set_ylabel(variableNm)
                if iR == nRows-1: ax.set_xlabel('interval')
                iPred += 1
        plt.suptitle('Predictors for %s. 2prc, mean, 98prc, %s' % (variableNm, chTitle))
        #
        # print to file or common zip archive?
        if self.dicDumpZip is None:
            plt.savefig(os.path.join(self.wrkDir,"LUT_output", 'LUT_' + chTitle + '.png'), dpi=200)
        else:
            streamOut = io.BytesIO()
            plt.savefig(streamOut, dpi=200) #, bbox_inches='tight')
            with self.dicDumpZip['LUT'].open('LUT_' + chTitle + '.png', 'w') as pngOut:
                pngOut.write(streamOut.getbuffer())
            streamOut.close()
            
        print(os.path.join(self.wrkDir,"LUT_output", chTitle + '.png'))
        fig.clf()
        plt.close()



if __name__ == '__main__':
    print(sklearn.__version__)
    
    
    