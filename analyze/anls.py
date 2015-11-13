# author: Karl Marrett
# analyze pupillometry data

#if there is time
#TODO use remove_blink_artifacts to nullify certain target
#windows use position of eye to clean the results later
#TODO think about degrees of freedom
#TODO think about subtracting by condition

import glob
from os import path as op
from scipy import io as sio 
from scipy import stats
import pyeparse as pp 
import numpy as np
import matplotlib.pyplot as plt
import pickle as pck
from collections import namedtuple
from datetime import datetime

#assert(pp.__version__ == .01)
# if something was changed in the file saving process
force_reprocess = False
subjects = ['HP', 'HL', 'GH', 'GG', 'GN', 'GI', 'HT', 'HI', 'HN', 'HK', 'HJ', 'GR', 'GU'] 

#subjects = ['HP', 'HL']
#shorten for debugging

if force_reprocess:
    reprocess_list = subjects
else:
    reprocess_list = []
N = len(subjects)

#assert version code
stim_version_code = 8010
# asserted fs
fs = 1000.0  
sig_thresh = .05
FONT_SIZE = 8
TITLE_SIZE = 8 

#data_dir = os.path.abspath(os.path.join(os.pardir, 'Data'))
results_dir = op.abspath(op.join(op.pardir, 'paperFiles'))
data_dir = '/home/kdmarrett/lab/FilesScript/Data'

def cleanCogData(weighted=False, weighting_type='default'):
    """ Return formatted cognitive load survey
    data"""
    gen_qnum = 6 #factors 
    rel_qnum = 15
    dicts = [ 
            {1 : 2, 4 : 1, 7 : 2, 13: 1, 14 : 1},#Mental
            {0 : 1, 2 : 2, 5 : 1, 12: 2, 14 : 2},#Physical
            {0 : 2, 4 : 2, 6 : 1, 8 : 1, 10 : 2}, #temporal
            {6 : 2, 9 : 1, 11: 2, 12: 1, 13 : 2},#performance
            {1 : 1, 3 : 1, 5 : 2, 9 : 2, 10 : 1}, #effort
            {2 : 1, 3 : 2, 7 : 1, 8 : 2, 11 : 1}  #frustration
            ]
    cog_subj = np.zeros((N, condition_nums))
    rel = np.zeros((N, condition_nums, gen_qnum))
    gen = np.ndarray(shape=(N, condition_nums, gen_qnum))
    #tally the importance of different selections
    for i in range(N):
        for j in range(condition_nums):
            total_relative_responses = 0
            if weighted:
                for k in range(rel_qnum):
                   response = \
                        cog[i][j]['rel_' + para[j] + '_qnum_' + str(k)][0,0]
                   for dind, survey_dict in enumerate(dicts):
                       if survey_dict.has_key(k):
                           if survey_dict[k] == response:
                                rel[i, j, dind] += 1
                                total_relative_responses += 1
                                break
               
                one_modified = False
                for dind, survey_dict in enumerate(dicts):
                   if rel[i, j, dind] == 0:
                       #give all factors at least some partial weighting
                       #if weighting is not WWL type
                       if weighting_type is 'default':
                           rel[i, j, dind] += 1
                           total_relative_responses += 1
                       #if two have 0 there must be a bug
                       assert(one_modified == False)
                       one_modified = True
            score = 0
            for k in range(gen_qnum):
                response =  cog[i][j]['gen_' + para[j] + '_qnum_' + \
                    str(k)][0,0]
                gen[i, j, k] = response
                if k in (3, 5):
                    response = 10 - response
                assert(response >= 0)
                if weighted:
                    if weighting_type is 'WWL':
                        weight = float(rel[i, j, k])
                    else:
                        weight = float(rel[i, j, k]) / total_relative_responses
                    response *= weight
                assert(response >= 0)
                score += response
            if  weighting_type is 'WWL':
                score /= total_relative_responses
            cog_subj[i, j] = score
    cog_mean = np.nanmean(cog_subj, axis=0)
    cog_ste = stats.sem(cog_subj, axis=0)
    return cog_subj, cog_mean, cog_ste, gen

def simpleaxis(ax):
    """Taken from timday on S.O.,
     Remove the plot frame lines."""
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

def con_2_ind(pattern):
    return int(pattern[-3:].replace(" ", ""), 2) - 1

def pResults(header, var):
    """Print some numpy var or vector to results file, 
    if the vector happens to be of condition_nums len
    then preface the output by the condition name"""

    resultstxt.write(header + ':\n')
    if len(var) == condition_nums:
        for nind, name in enumerate(names):
            resultstxt.write(name + ': ')
            resultstxt.write('%.4f\n' % var[nind])
    else:
        np.savetxt(resultstxt, var, delimiter=', ', fmt='%.2f')
    resultstxt.write('\n')
    return

def pGroupedResults(stats_tuple, group):
    pResults('Pupil global %s uncorrected means' % str(group), stats_tuple.global_mean)
    #pResults('Pupil global %s standard error' % str(group), stats_tuple.global_ste)
    printSignificant('PS uncorrected %s' % str(group), stats_tuple.ps_subj_means)
    pResults('Pupil global %s bc means' % str(group), stats_tuple.global_bc_mean)
    #pResults('Pupil global %s bc standard error' % str(group), stats_tuple.global_bc_ste)
    printSignificant('PS baseline corrected %s' % str(group), stats_tuple.ps_subj_bc_means)
 
def combinedSigTest(header, subj_combined, strategy):
    """ Takes a matrix of N subjects by condition nums of 
    some statistic and computes the relative and independent 
    t test scores.  Prints these scores to results.txt, with 
    significant relations with * appended."""
    #assert(subject_data.shape == (N, condition_nums))
    significant = np.zeros(shape=(len(measure_names),
        condition_nums), dtype=bool)
    p = np.zeros(shape=(len(measure_names),
        condition_nums), dtype=bool)
    trel = np.zeros(shape=(len(measure_names),
        condition_nums), dtype=bool)
    resultstxt.write(header + ' sig. testing:\n')
    combo = (0, 1)
    trel[0], p[0], significant[0] = testSigComb(combo,subj_combined,
            strategy)
    combo = (0, 2)
    trel[1], p[1], significant[1] = testSigComb(combo,subj_combined,
            strategy)
    combo = (1, 2)
    trel[2], p[2], significant[2] = testSigComb(combo,subj_combined,
            strategy)
    resultstxt.write('\n')
    return significant, p

def testSigComb(combo, subj_combined, strategy):
    significant = False
    significant = np.zeros(condition_nums, dtype=bool)
    p_arr = np.zeros(condition_nums, dtype=bool)
    resultstxt.write(measure_names[combo[0]] + ' and ')
    resultstxt.write(measure_names[combo[1]] + ':\n')
    for cind in range(condition_nums):
        resultstxt.write(names[cind] + ':')
        (trel, pval) =\
            strategy(subj_combined[combo[0],:, cind],\
                    subj_combined[combo[1],:, cind])
        resultstxt.write('%.4f' % pval)
        resultstxt.write(', with R:%.4f' % trel)
        p_arr[cind] = pval
        if pval < sig_thresh:
            resultstxt.write('**')
            significant[cind] = True
        resultstxt.write('\n')
    resultstxt.write('\n')
    return trel, p_arr, significant

def printSignificant(header, subject_data):
    """ Takes a matrix of N subjects by condition nums of 
    some statistic and computes the relative and independent 
    t test scores.  Prints these scores to results.txt, with 
    significant relations with * appended."""
    assert(subject_data.shape == (N, condition_nums))
    significant = np.zeros(condition_nums, dtype=bool)
    prel = np.zeros(condition_nums, dtype=float)
    resultstxt.write(header + ' sig. testing:\n')
    combo = (0, 1)
    trel, prel[0], significant[0] = testSig(combo,subject_data)
    combo = (0, 2)
    trel, prel[1], significant[1] = testSig(combo,subject_data)
    combo = (1, 2)
    trel, prel[2], significant[2] = testSig(combo,subject_data)
    resultstxt.write('\n')
    return significant, prel

def printSignificantInter(header, subject_data,
        subject_data2):
    """ Takes two matrices of N subjects by condition nums of some
    statistic and computes the relative and independent t test
    scores between conditions.  Prints these scores to results.txt,
    with significant relations with * appended."""

    assert(subject_data.shape == (N, condition_nums))
    significant = np.zeros(condition_nums, dtype=bool)
    prel = np.zeros(condition_nums, dtype=float)
    resultstxt.write(header + ' sig. testing:\n')
    for ci, name in enumerate(names):
        resultstxt.write(name + ':\n')
        (trel, prel[ci]) = stats.ttest_rel(subject_data[:, ci],\
            subject_data2[:, ci])
        #(trel, prel[ci]) = stats.ttest_ind(subject_data[:, ci],\
            #subject_data2[:, ci])
        resultstxt.write('%.4f' % prel[ci])
        if prel[ci] < sig_thresh:
            resultstxt.write('**')
            significant[ci] = True
        resultstxt.write('\n')
    resultstxt.write('\n')
    return significant, prel

def testSig(combo, subject_data):
    significant = False
    #(tind, pind) = stats.ttest_ind(subject_data[:, combo[0]],\
        #subject_data[:, combo[1]])
    (trel, prel) = stats.ttest_rel(subject_data[:, combo[0]],\
        subject_data[:, combo[1]])
    resultstxt.write(names[combo[0]] + ' and ')
    resultstxt.write(names[combo[1]] + ':\n')
    resultstxt.write('%.4f' % prel)
    if prel < sig_thresh:
        resultstxt.write('**')
        significant = True
    resultstxt.write('\n')
    return trel, prel, significant


def getTrialInfo(block_ind, bnum, advance=False):
    """ Returns generic information about particular trial wav data """

    data_file_name = 'b' + str(bnum) + '_tr' + str(block_ind[bnum]) + \
    'trial'
    trial_data_path = op.join(param_data_dir, data_file_name)
    trial_vars = sio.loadmat(trial_data_path)
    target_cycles = trial_vars['target_cycles']
    target_letter = trial_vars['target_letter'][0][0][0].encode('ascii')
    possible_letters = trial_vars['possible_letters'][0]
    id_list = trial_vars['trial_id'][0].tolist()
    paradigm = trial_vars['paradigm'][0]
    if advance:
        block_ind[bnum] += 1
    return paradigm

def getFinalInfo(block_ind, bnum, subj_data_dir, advance=False):
    """Returns final information about the trial (i.e. whether subject
    response was correct and stamped id_list)"""

    data_file_name = 'b' + str(bnum) + '_tr' + str(block_ind[bnum]) + 'final'
    trial_data_path = op.join(subj_data_dir, data_file_name)
    trial_vars = sio.loadmat(trial_data_path)
    correct = trial_vars['correct'][0]
    id_list = trial_vars['trial_id'][0].tolist()
    if advance:
        block_ind[bnum] += 1
    return id_list, correct

def getGeneralInfo(block_ind, bnum, advance=False):
    """Returns final information about the trial (i.e. whether subject
    response was correct and stamped id_list)"""

     #save(gen_final_data_dir, 'target_letter', 'target_time',...
     #'tot_wav_time', 'paradigm', ...
     #'possible_letters', 'target_cycles', 'location_code',...
     #'wheel_matrix', 'replacement_letter', 'targ_cyc_ind', ...
     #'replacement_time', 'trial_id');

    data_file_name = 'b' + str(bnum) + '_tr' + str(block_ind[bnum]) + \
    'general'
    trial_data_path = op.join(param_data_dir, data_file_name)
    trial_vars = sio.loadmat(trial_data_path)
    #try:
        #replacement_time = trial_vars['replacement_time'][0]
    #except:
        #print 'replacement time not found'
        #replacement_time = None
    target_time = trial_vars['target_time'][0]
    location_code = trial_vars['location_code'][0][0][0].encode('ascii')
    id_list = trial_vars['trial_id'][0].tolist()
    target_letter = \
        trial_vars['target_letter'][0][0][0].encode('ascii')
    # check loading of correct mat file
    #assert (trial_vars['paradigm'][0].encode('ascii') == \
            #id_), \
    #"Error: id_ and paradigm from mat file do not match"
    if advance:
        block_ind[bnum] += 1
    return target_time

def getCogResponses(cog, subj_data_dir):
    temp = []
    for param in para.values():
        data_file_name = param + '_cog_resp.mat'
        cog_data_path = op.join(subj_data_dir, data_file_name)
        temp.append(sio.loadmat(cog_data_path))
    cog.append(temp)
    return cog

def getConditionEpochs(fnames):
    for ci, pattern in condition_pattern:
        event_id = ci
        raw = pp.Raw(fname)
        raw.remove_blink_artifacts
        event = raw.find_events(pattern, event_id)
        #parse ps data and add to dicts/matrices
        tmin = 0.0
        tmax = trial_len + postblock
        trial_epoch = pp.Epochs(raw, events=events, 
            event_id=event_id, tmin=tmin, tmax=tmax)
        epochs.append(trial_epoch)
        ps.append(trial_epoch.get_data('ps')[0])

def subj_accuracy_stats(accuracy_data):
    subj_means = np.zeros((N, condition_nums), dtype=np.double)
    subj_stds = np.zeros((N, condition_nums), dtype=np.double)
    for i, subj_accuracy in enumerate(accuracy_data):
        for c_ind in range(condition_nums):
            subj_means[i][c_ind] = np.nanmean(subj_accuracy[c_ind])
            subj_stds[i][c_ind] = np.nanstd(subj_accuracy[c_ind])
    global_mean = []
    global_ste = []
    for i in range(condition_nums):
        #across all subjects for each condition 
        global_mean.append(np.nanmean(subj_means[:, i]))
        global_ste.append(stats.sem(subj_means[:, i]))
    #convert to percent
    global_mean= global_mean * np.tile(100, len(global_mean))
    global_ste = global_ste * np.tile(100, len(global_ste))
    subj_means = subj_means * 100
    subj_stds = subj_stds * 100
    return acc_tuple(global_mean, global_ste, subj_means, subj_stds)

def subj_ps_stats(ps_data, data_type='trial',\
        window_start=0, window_end='end', take_trials='all',
        normalize_by_std=False):
    """ ps[subject, cond_ind, block, trial, sample] 
    Params:
        window_start : start sample to analyze
        window_end : end sample to analyze
        data_type : if 'trial' trim around ps data from the window of
        the task if 'target' use all of the target window passed
        without windowing"""

    # get # samples (last dim) of the ps data
    local_samp_len = ps_data.shape[-1]

    #end of trial for trial type
    if window_end is 'end':
        #include up to last sample in window
        window_end = local_samp_len
    #print '%d to %d' % (window_start, window_end)
    if take_trials is 'all':
        trials_to_process = trials_per_cond
    else:
        trials_to_process = trials_per_cond / 3

    if data_type is 'target':
        assert(local_samp_len == target_samp)
    elif data_type is 'trial':
        assert(local_samp_len == trial_samp)

    # dat holds each time trace
    mean_dat = np.full((N, condition_nums, local_samp_len), np.nan)
    bc_mean_dat = np.full((N, condition_nums, local_samp_len), np.nan)
    #get a mean value for the baseline period proceeding each trial 
    base_mean = np.full((N, trials_to_process), np.nan)
    #FIXME collapse this
    for s_ind, subj_ps in enumerate(ps_data):
        for c_ind in range(condition_nums):
            if data_type is 'target':
                total_targets = trials_to_process * max_targets
                if take_trials is 'all':
                    mean_dat[s_ind, c_ind] = np.nanmean(
                            subj_ps[c_ind].reshape(total_targets,
                                local_samp_len), axis=0)
                    raw_windows = subj_ps[c_ind].reshape(
                            total_targets, local_samp_len)
                    base_mean[s_ind, :] = np.nanmean(ps_base[s_ind,
                            c_ind].reshape(trials_to_process,
                                int(base_samp)), axis=1)
                else:
                    #take only specified third of the data
                    if take_trials is 'start':
                        #take only first 9 trials (first three blocks)
                        mean_dat[s_ind, c_ind] = np.nanmean(
                                subj_ps[c_ind,:3].reshape(total_targets,
                                    local_samp_len), axis=0)
                        raw_windows = subj_ps[c_ind,:3].reshape(
                                total_targets, local_samp_len)
                        base_mean[s_ind, :] = np.nanmean(ps_base[s_ind,
                            c_ind, :3].reshape(trials_to_process,
                                    int(base_samp)), axis=1)
                    elif take_trials is 'end':
                        #take last 9 trials
                        mean_dat[s_ind, c_ind] = np.nanmean(
                                subj_ps[c_ind, -3:].reshape(total_targets,
                                    int(target_samp)), axis=0)
                        raw_windows = subj_ps[c_ind, -3:].reshape(
                                total_targets, local_samp_len)
                        base_mean[s_ind, :] = np.nanmean(ps_base[s_ind,
                            c_ind, -3:].reshape(trials_to_process,
                                    int(base_samp)), axis=1)
                #for each subject for each trial find the corresponding
                #single baseline value to subtract per trial
                bc_window = np.zeros(shape=(total_targets,
                    local_samp_len))
                bc_window[:] = np.nan
                for rti in range(trials_to_process):
                    for targi in range(max_targets):
                        #subtract the mean of each trial baseline 
                        bc_window[max_targets * rti + targi, :] =\
                            raw_windows[max_targets * rti + targi] -\
                            base_mean[s_ind, rti]
                bc_mean_dat[s_ind, c_ind] = np.nanmean(bc_window,
                        axis=0)
            elif data_type is 'trial':
                if take_trials is 'all':
                    ## raw mean stack for each subject and condition
                    mean_dat[s_ind, c_ind] = np.nanmean(
                            subj_ps[c_ind].reshape(trials_to_process,
                                local_samp_len), axis=0)
                    raw_windows = subj_ps[c_ind].reshape(
                            trials_to_process, local_samp_len)
                    #for each subject for each trial find the corresponding
                        #baseline to subtract
                    base_mean[s_ind, :] = np.nanmean(ps_base[s_ind,
                            c_ind].reshape(trials_to_process,
                                int(base_samp)), axis=1)
                    #subtract the mean of each trial baseline 
                    bc_trial = np.zeros(shape=(trials_to_process,
                        local_samp_len))
                    bc_trial[:] = np.nan
                    for rti, raw_trial in enumerate(raw_windows):
                        bc_trial[rti,:] = raw_trial - base_mean[s_ind, rti]
                    bc_mean_dat[s_ind, c_ind] = np.nanmean(bc_trial,
                            axis=0)
                else:
                    #take only specified third of the data
                    if take_trials is 'start':
                        #take only first 9 trials
                        mean_dat[s_ind, c_ind] = np.nanmean(
                                subj_ps[c_ind,:3].reshape(trials_to_process,
                                    local_samp_len), axis=0)
                        raw_windows = subj_ps[c_ind,:3].reshape(
                                trials_to_process, local_samp_len)
                        base_mean[s_ind, :] = np.nanmean(ps_base[s_ind,
                            c_ind,:3].reshape(trials_to_process,
                                    int(base_samp)), axis=1)
                    elif take_trials is 'end':
                        #take last 9 trials
                        mean_dat[s_ind, c_ind] = np.nanmean(
                                subj_ps[c_ind,
                                    -3:].reshape(trials_to_process,
                                        local_samp_len), axis=0)
                        raw_windows = subj_ps[c_ind, -3:].reshape(
                                trials_to_process, local_samp_len)
                        base_mean[s_ind, :] = np.nanmean(ps_base[s_ind,
                            c_ind,-3:].reshape(trials_to_process,
                                    int(base_samp)), axis=1)
                    #for each subject for each trial find the corresponding
                        #baseline to subtract
                    #ps_base.shape=(N, condition_nums, s2_blocks,
                        #block_len / condition_nums, base_samp))
                    #subtract the mean of each trial baseline 
                    bc_trial = np.zeros(shape=(trials_to_process,
                        local_samp_len))
                    bc_trial[:] = np.nan
                    for rti, raw_trial in enumerate(raw_windows):
                        bc_trial[rti,:] = raw_trial - base_mean[s_ind, rti]
                    bc_mean_dat[s_ind, c_ind] = np.nanmean(bc_trial, axis=0)
    mean_dat_trim = mean_dat[:,:,window_start:window_end]
    bc_mean_dat_trim = bc_mean_dat[:,:,window_start:window_end]
    #subject means for sig testing and plotting
    ps_subj_means = np.nanmean(mean_dat_trim, axis=2)
    ps_subj_std = np.nanstd(mean_dat_trim, axis=2)
    ps_subj_bc_std = np.nanstd(bc_mean_dat_trim, axis=2)
    #get the standard deviation for every subject
    #temp = np.nanstd(bc_mean_dat, axis=2)
    #subj_experiment_stds = np.nanmean(temp, axis=1)
    subj_experiment_stds= [np.nanstd(dat) for dat in ps_data]
    #import pdb; pdb.set_trace()
    #normalize each subjects baseline corrected data
    if normalize_by_std:
        for si, subject_exp_std in enumerate(subj_experiment_stds):
            bc_mean_dat_trim[si, ...] = bc_mean_dat_trim[si, ...] / subject_exp_std
            bc_mean_dat[si, ...] = bc_mean_dat[si, ...] / subject_exp_std
    ps_subj_bc_means = np.nanmean(bc_mean_dat_trim, axis=2)
    assert(ps_subj_bc_means.shape == (N, condition_nums))
    # means across all subjects
    full_mean_trace = np.nanmean(mean_dat, axis=0) 
    full_mean_bc_trace = np.nanmean(bc_mean_dat, axis=0) 
    full_ste_trace = stats.sem(mean_dat, axis=0) 
    full_ste_bc_trace = stats.sem(bc_mean_dat, axis=0) 
    mean_trace = np.nanmean(mean_dat_trim, axis=0) 
    bc_mean_trace = np.nanmean(bc_mean_dat_trim, axis=0)
    #sem is sensitive to np.nan values
    ste_trace = stats.sem(mean_dat_trim, axis=0)
    bc_ste_trace = stats.sem(bc_mean_dat_trim, axis=0)

    #single number for global average for each condition
    global_mean = np.nanmean(mean_trace, axis=1)
    global_ste = stats.sem(ps_subj_means, axis=0)
    global_bc_mean = np.nanmean(bc_mean_trace, axis=1)
    #import pdb; pdb.set_trace()
    global_bc_ste = stats.sem(ps_subj_bc_means, axis=0)

    #find peak for each subject
    subj_bc_peaks = np.nanmax(bc_mean_dat_trim, axis=2)
    subj_mean_corrected_peaks = subj_bc_peaks - ps_subj_bc_means
    ##then average those peaks
    global_bc_peak_mean = np.nanmean(subj_bc_peaks, axis=0)
    global_mc_peak_mean = np.nanmean(subj_mean_corrected_peaks,
            axis=0)
    global_bc_peak_ste = stats.sem(subj_bc_peaks, axis=0)
    global_mc_peak_ste = stats.sem(subj_mean_corrected_peaks, axis=0)

    assert(len(global_bc_peak_ste) == condition_nums)
    assert(len(global_bc_peak_mean) == condition_nums)
    assert(len(mean_trace) == condition_nums)
    assert(len(global_mean) == condition_nums)
    assert(len(global_bc_mean) == condition_nums)
    assert(len(global_ste) == condition_nums)
    return stats_tuple(full_mean_trace, full_mean_bc_trace, full_ste_trace,\
            full_ste_bc_trace, mean_trace,\
            bc_mean_trace, ste_trace, bc_ste_trace, global_mean,\
            global_ste, global_bc_mean, global_bc_ste,\
            ps_subj_means, ps_subj_std,\
            ps_subj_bc_means, ps_subj_bc_std,\
            global_bc_peak_mean, global_mc_peak_mean,\
            global_bc_peak_ste, global_mc_peak_ste,\
            subj_bc_peaks, subj_mean_corrected_peaks)

    #return full_mean_trace, full_mean_bc_trace, full_ste_trace,\
            #full_ste_bc_trace, mean_trace,\
            #bc_mean_trace, ste_trace, bc_ste_trace, global_mean,\
            #global_ste, global_bc_mean, global_bc_ste,\
            #ps_subj_means, ps_subj_std,\
            #ps_subj_bc_means, ps_subj_bc_std,\
            #global_bc_peak_mean, global_mc_peak_mean,\
            #global_bc_peak_ste, global_mc_peak_ste,\
            #subj_bc_peaks, subj_mean_corrected_peaks

roundToIncrement = lambda y, inc: round(float(y) / inc) * inc

def double_barplot(name, ylabel, y_increment, pre, post,
    yrange='default', subject_lines=False, 
    draw_sig=False, show=True):

    if name is 'Accuracy':
        means = zip(pre.global_mean, post.global_mean)
        stes = zip(pre.global_ste, post.global_ste)
        all_subject_data = np.concatenate((pre.subj_means,
            post.subj_means))
        #subject_data.shape is [condition, subject, pre or post]
        subject_data = [zip(pre.subj_means[:, i],
            post.subj_means[:,i]) for i in
            range(post.subj_means.shape[-1])]
        #import pdb; pdb.set_trace()
    else:
        means = zip(pre.global_bc_mean, post.global_bc_mean)
        stes = zip(pre.global_bc_ste, post.global_bc_ste)
        all_subject_data = np.concatenate((pre.ps_subj_bc_means,
            post.ps_subj_bc_means))
        #subject_data.shape is [condition, subject, pre or post]
        subject_data = [zip(pre.ps_subj_bc_means[:, i],
            post.ps_subj_bc_means[:,i]) for i in
            range(post.ps_subj_bc_means.shape[-1])]
    N = 2
    ind = np.arange(N) / 1.5  # the x locations for the groups
    width = 0.15       # the width of the bars
    opacity = .2
    #fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(1.0, 2.0))
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(1.5, 2.5)) # original
    #fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(3.0, 4.5))
    simpleaxis(ax)
    if not subject_lines:
        all_subject_data = [max(means), min(means)]
    #find range
    if yrange is 'default':
        lim_buffer = y_increment
        yrange = np.zeros((2,1))
        yrange[0] = roundToIncrement((np.nanmin(all_subject_data) - lim_buffer),
                y_increment)
        yrange[1] = roundToIncrement(np.nanmax(all_subject_data) +\
                lim_buffer, y_increment)

    colors = ['w','w','w']
    hatchs = ['','..','//']
    rects = []
    error_config = {'ecolor': 'k', 'elinewidth': 2, 'ezorder': 5}
    x_list = []
    for i, mean in enumerate(means): #iterate through conditions
        x = ind + width * (i - 1)
        x_list.append(x)
        #plot global data
        rects.append(ax.bar(x, mean, width,
            color=colors[i],lw=2.0,
            yerr=stes[i], error_kw=error_config, hatch=hatchs[i]))
        x += (.5 * width)
        # Call significance bar func
        #plot individual subjects
        if subject_lines:
            for subj_mean in subject_data[i]:
                ax.plot(x, subj_mean, color='k', alpha=opacity, lw=.5,
                        marker='o', zorder=10)

    # significance bars from Stack over. post
    def label_diff(i,j,k,l,text, texty, X, ymax, shrink):
        y = ymax
        #order by condition then initial or final
        xtext = X[i][j] + abs(X[i][j]-X[k][l]) / 2
        props = {'connectionstyle':'bar, fraction=0.2','arrowstyle':'-',\
             'shrinkA':shrink,'shrinkB':shrink,'lw':1.0}
        ax.annotate(text, xy=(xtext, texty),
                fontsize=14, zorder=10)
                #fontsize=18, zorder=10, fontweight='bold')
        #import pdb; pdb.set_trace()
        ax.annotate('', xy=(X[i][j],y), xytext=(X[k][l],y),
            xycoords='data', textcoords='data', 
            zorder=10, arrowprops=props)

    # add some text for labels, title and axes ticks
    #import pdb; pdb.set_trace()
    if draw_sig:
        ymax = np.max(means)
        #label_diff(1, 0, 2, 0,'*', 600, x_list, 600, 400)
        #label_diff(0, 0, 1, 0,'*', 640, x_list, 640, 400)
        #label_diff(1, 0, 1, 1,'*', 745, x_list, 700, 5)
        label_diff(1, 0, 2, 0,'*', 620, x_list, 600, 400)
        label_diff(0, 0, 1, 0,'*', 660, x_list, 640, 400)
        label_diff(1, 0, 1, 1,'*', 765, x_list, 680, 10)
        #label_diff(1, 0, 1, 1,'*', 725, x_list, 560, 1)
    #import pdb; pdb.set_trace()
    ax.set_ylabel(ylabel, fontsize=FONT_SIZE)
    ax.set_title(name, fontsize=TITLE_SIZE)
    ax.set_ylim(yrange)
    ax.set_xticks(ind + width / 2)
    ax.set_xticklabels( ('Initial', 'Final'), fontsize=FONT_SIZE )
    ax.set_yticklabels(range(yrange[0], yrange[1], y_increment),
            fontsize=FONT_SIZE )
    #ax.legend((rects[0], rects[1], rects[2]),
            #('Alphabetic', 'Fixed-order', 'Random'))
    if show:
        plt.show()
    fn = name.replace(" ", "") + '_2bplt.png'
    print 'Saving figure:\n%s' % fn
    #fig.set_size_inches(3, 1.7)
    fig.savefig(fn, dpi=DPI_NO)
    plt.close()

def barplot(title, ylabel, y_increment, subject_data, global_subj_mean,\
        global_subj_ste, yrange='default', ylines=False, show=False):
    #fig, ax = plt.subplots(figsize=(12, 14)) 
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(4,3))
    # Remove the plot frame lines.
    simpleaxis(ax)
    x = [.5, 1.0, 1.5]
    ax.set_xlim((.3, 2))
    bar_width = .25
    opacity = .3
    #find range
    if yrange is 'default':
        lim_buffer = y_increment
        yrange = np.zeros((2,1))
        yrange[0] = roundToIncrement((np.nanmin(subject_data) - lim_buffer),
                y_increment)
        yrange[1] = roundToIncrement(np.nanmax(subject_data) +\
                lim_buffer, y_increment)
        #yrange[0] = roundToIncrement(np.nanmin(global_subj_mean) -\
        #np.nanmax(global_subj_ste) - lim_buffer, y_increment)
        #yrange[1] = roundToIncrement(np.nanmax(global_subj_mean) +\
                #np.nanmax(global_subj_ste) + lim_buffer, y_increment)

    # significance bars from Stack over. post
    def label_diff(i,j,text, texty, X, ymax, shrink):
        y = 1 + ymax
        xtext = X[i] + abs(X[i]-X[j]) / 2
        props = {'connectionstyle':'bar, fraction=0.2','arrowstyle':'-',\
             'shrinkA':shrink,'shrinkB':shrink,'lw':1.0}
        ax.annotate(text, xy=(xtext, texty),
                fontsize=14, zorder=10)
        #import pdb; pdb.set_trace()
        ax.annotate('', xy=(X[i],y), xytext=(X[j],y),
            xycoords='data', textcoords='data', arrowprops=props)

    #plot yaxis lines
    if ylines:
        for y in range(yrange[0], yrange[1], y_increment):  
            ax.plot(range(0,3), [y] * len(range(0,3)), "--",
                    lw=0.5, color="black", alpha=0.3, zorder=1) 

    #plot global data
    error_config = {'ecolor': 'k', 'elinewidth': 3, 'ezorder': 5}
    hatchs = ['','..','//']
    rects = []
    for local_x, cond_mean, cond_ste, hatch in zip(x,
        global_subj_mean, global_subj_ste, hatchs):
        rects.append(ax.bar(local_x, cond_mean, bar_width, color='w', #zorder=3,
                yerr=cond_ste, error_kw=error_config, lw=2.0,
                hatch=hatch))
    #ax.bar(x, global_subj_mean, bar_width, color='w', zorder=3,
            #yerr=global_subj_ste, error_kw=error_config, lw=1)
    #shift x to be in center of bars
    x = x + np.tile(bar_width / 2, condition_nums)

    #plot individual subjects
    for subj_mean in subject_data:
        ax.plot(x, subj_mean, color='k', alpha=opacity, lw=.5,
                marker='o', zorder=10)

    ymax = np.max(subject_data)
    #import pdb; pdb.set_trace()
    label_diff(1,2,'*', 9.4, x, ymax - .1, 7)
    label_diff(0,2,'*', 8.7, x, ymax - 1.1, 20)
    ax.set_ylabel(ylabel, fontsize=FONT_SIZE)
    ax.set_ylim(yrange)
    ax.legend((rects[0], rects[1], rects[2]),
            ('Alphabetic', 'Fixed-order', 'Random'), loc=4,
            prop={'size':9})
    #ax.set_title('%s N=%d' % (title, N))
    ax.set_title(title, fontsize=TITLE_SIZE)
    plt.xticks(x, ('Alphabetic', 'Fixed-order', 'Random'),
            fontsize=FONT_SIZE)
    ax.set_yticklabels(range(yrange[0], yrange[1], y_increment),
            fontsize=FONT_SIZE )
    #plt.tight_layout()
    if show:
        plt.show()
    fn = title.replace(" ", "") + '_barplot.png'
    print 'Saving figure:\n%s' % fn
    #fig.set_size_inches(3, 1.7)
    fig.savefig(fn, dpi=DPI_NO)
    plt.close()

def plot_ps(trace, ste_trace, name, ax='default',
        final_sub_plot=False):
    """plot a stack of subject mean ps data of all conditions
    Params:   trace can either by full trial or a letter window"""

    if ax is 'default':
        fig, local_ax = plt.subplots()
    #clean the ticks
    local_ax.xaxis.set_ticks_position('bottom')
    local_ax.yaxis.set_ticks_position('left')
    opacity = .10
    local_samp_len = trace.shape[1]
    x = np.linspace(0, local_samp_len / fs, local_samp_len)
    #colors = ('r', 'g', 'b')
    colors = ('#44AA99', '#999933', '#AA4499')
    for c_num in range(condition_nums):
        local_ax.plot(x, trace[c_num], color=colors[c_num],
                linewidth=1, label=names[c_num], alpha=1)
        local_ax.fill_between(x, trace[c_num] - ste_trace[c_num],\
                trace[c_num] + ste_trace[c_num],\
                color=colors[c_num], alpha=opacity)  

    #mark visual_primer if length is full trial
    ymin, ymax = local_ax.get_ylim()
    yspan = ymax - ymin
    topFig = ymax - yspan / 7
    topFigLower = ymax - yspan / 6
    if local_samp_len == trial_samp: # a trial
        local_ax.text(end_primer / 3, topFig, 'Visual\nprimer',\
                size=10) 
        #ax.axvspan(0, end_primer, color='k', alpha=.12)
        local_ax.text(end_primer + fix_dot_time / 3.75,\
                topFig, 'Fixation\n   dot',\
                size=10) 
        local_ax.axvspan(end_primer, preblock_prime_sec,\
                color='k', alpha=.07)
        local_ax.text(24, topFig, 'Task\n', size=10) 
        local_ax.legend(loc=4, prop={'size':10})
        #local_ax.set_title(name + ' trial pupil size N=%d' % N)
        local_ax.set_title(name + ' trial pupil size')
        for i in cycle_start_sec[1:]:
            local_ax.axvline(x=i, linestyle='--', color='k',
                    alpha=.4)
        #indicate with arrow
        #ax.annotate('End visual primer', xy=(end_primer, 
            #global_mean[c_num]), xytext=(5, 2000),
            #arrowprops=dict(facecolor='black', shrink=0.02))
    elif (local_samp_len == target_samp): # a window
        local_ax.axvline(x=preslice_time, linestyle='--', color='k',
                alpha=.4)
        local_ax.text(preslice_time + .1, topFigLower, 
                'Target\n letter\n onset', size=10) 
        local_ax.legend(loc=2, prop={'size':10})
        local_ax.set_title(name + ' window pupil size N=%d' % N)

    local_ax.set_ylabel('Corrected pupil pixel area')
    #Render stats to plot
    #info = r'$\mu$=%.1f, $\sigma$=%.3f, N=%d' % \
            #(global_mean[c_num],\
            #global_std[c_num], N)
    #plt.text(20, global_mean[c_num] + 500, info)
    #plt.show()
    local_ax.set_xlabel('Trial Time (s)')
    local_ax.set_xlim((0, local_samp_len / fs))
    name = name.replace(" ", "")
    fn = name + '_trace.png'
    print 'Saving figure:\n%s' % fn
    #fig.set_size_inches(3, 1.7)
    fig.savefig(fn, dpi=DPI_NO)
    plt.close()
    #if final_sub_plot:
        ##plt.show()
        #name = name.replace(" ", "")
        #fn = name + '_trace.png'
        #print 'Saving figure:\n%s' % fn
        #fig.savefig(fn)
        #plt.close()
    #else:
        #local_ax.set_xlabel('Trial Time (s)')
        #local_ax.set_xlim((0, local_samp_len / fs))
        #return local_ax

names = ['Alphabetic', 'Fixed-Order', 'Random']
# read in global stimuli parameters
param_data_dir = op.join(data_dir, 'Params')
global_vars = sio.loadmat(op.join(param_data_dir, 'global_vars.mat'))
preblock = global_vars['preblock_prime_sec'][0]
trial_len = 36.000
DPI_NO = 200
trial_samp = np.floor(trial_len*fs).astype(int)
# time of visual primer (s)
end_primer = global_vars['vPrimerLen'] 
end_primer_samp = int(end_primer * fs)
end_stim = int(trial_len * fs)
#seconds until letters start playing
preblock_prime_sec = 13 
cycle_time = 4.5
fix_dot_time = preblock_prime_sec - end_primer
target_window = 6000
usable_criterion = .15
#accept pupil data within deviations of mean
deviations = 3
#TODO get actual screen distance from Marlo
#TODO convert the physical distances to pixels
screen_distance = 1
angle_criterion = 40
# time after each trial to record pupil
postblock = global_vars['postblock'] 
#trial_len = global_vars['tot_wav_time'] 
order = global_vars['order'][0]
s2_blocks = global_vars['s2_blocks']
# trials_per_block in middle section
block_len = len(order[1][0])
trials_exp = block_len*s2_blocks[0][0]
wheel_matrix_info = global_vars['wheel_matrix_info'][0]

# build condition strings
condition_uni = global_vars['condition_bin']  # Unicode by default
condition_asc = []  # ASCII
for i in range(len(condition_uni)):
    condition_asc.append(''.join(map(str, condition_uni[i].tolist())))
#condition_asc.append(condition_uni[i].encode('ascii'))
#old naming habits of paradigm
para = dict()
para[0] = '0000000'
para[1] = '0100000'
para[2] = '0101000'
condition_nums = len(condition_asc)
trials_per_cond = trials_exp / condition_nums
wav_indices = dict(
    zip(condition_asc, np.zeros(len(condition_asc), dtype=int)))
# build String condition ids for raw 'messages'
base = 'TRIALID'
condition_pattern = [''] * condition_nums;
patternToCond = dict()
for i in range(condition_nums):
    condition_pattern[i] = base
    for j in range(len(condition_asc[i])):
        condition_pattern[i] += ' ' + condition_asc[i][j]
    patternToCond[condition_pattern[i]] =  \
        int(condition_pattern[i][-3:].replace(" ", ""), 2)

#data structures
usable_trials = np.zeros(shape=(N, 1))
usable_trials[:] = np.nan

ps = np.ndarray(shape=(N, condition_nums, s2_blocks, 
    block_len / condition_nums, trial_samp))
ps[:] = np.nan

ps_incorrect = np.ndarray(shape=(N, condition_nums, 
    s2_blocks, block_len / condition_nums, trial_samp))
ps_incorrect[:] = np.nan

accuracy = np.ndarray(shape=(N, condition_nums,
    s2_blocks * block_len / condition_nums))
accuracy[:] = np.nan

#ps data of base correction
base_slice_time = 2
base_samp = np.ceil(base_slice_time * fs)
ps_base = np.ndarray(shape=(N, condition_nums, s2_blocks,
    block_len / condition_nums, base_samp))
ps_base[:] = np.nan
#ps data of targets
preslice_time = 4
preslice_samp = preslice_time * fs
postslice_time = 2.5
max_targets = 2
target_slice_time = preslice_time + postslice_time
target_samp = np.ceil(target_slice_time * fs)
ps_target = np.ndarray(shape=(N, condition_nums, s2_blocks,\
    block_len / condition_nums, max_targets, target_samp))
ps_target[:] = np.nan

#ps data of replacement letter
#replacement_slice_time = 2
#replacement_samp = np.ceil(replacement_slice_time * fs)
#ps_replacement = np.ndarray(shape=(N, condition_nums, s2_blocks,
    #block_len / condition_nums, max_targets, replacement_samp))
#ps_replacement[:] = np.nan

cog = []
accuracy_dict = dict()
ps_dict = dict()
status = ['correct', 'incorrect']
for pattern in (condition_pattern):
    accuracy_dict[pattern] = []
    for stat in (status):
        ps_dict[pattern + stat] = []
    
for s_ind, subj in enumerate(subjects):
    print('Subject %s...' % subj)

    # return all edfs files for a given subject
    fnames = sorted(glob.glob(op.join(data_dir, '%s_*' % subj, '*.edf')))
    subj_files = sorted(glob.glob(op.join(data_dir, '%s_*' % subj)))
    sessions = len(subj_files) / 3
    cond_acc_ind = np.zeros(shape=(condition_nums))
    subj_data_dir = []
    start_path = []
    startInfo = []

    for fi, file in enumerate(subj_files):
        if op.isdir(file):
            subj_data_dir.append(file)
            start_path.append(op.join(subj_data_dir[-1],
                'startInfo.mat'))
            startInfo.append(sio.loadmat(start_path[-1]))
    # check that mid_block order he same for this subject
    mid_block_order = startInfo[0]['mid_block_order'][0]
    subj_data_dir = subj_data_dir[0]
    processed_file = op.join(subj_data_dir, subj + '.obj')
    #assert it is not empty

    cog = getCogResponses(cog, subj_data_dir)

    try:
        assert(subj not in reprocess_list)
        fsubj = open(processed_file, 'r')
        (subj_accuracy, subj_ps, subj_ps_incorrect, subj_ps_base,
                subj_ps_target, subj_usable_trials) = pck.load(fsubj)
        accuracy[s_ind] = subj_accuracy
        ps[s_ind] = subj_ps
        ps_incorrect[s_ind] = subj_ps_incorrect
        ps_base[s_ind] = subj_ps_base
        #ps_replacement[s_ind] = subj_ps_replacement
        ps_target[s_ind] = subj_ps_target
        usable_trials[s_ind] = subj_usable_trials
        fsubj.close()
        continue
    except:
        #(re)process subject
        pass

    #new data struct templates
    subj_accuracy = accuracy[s_ind]
    subj_ps = ps[s_ind]
    subj_ps_incorrect = ps_incorrect[s_ind]
    subj_ps_base = ps_base[s_ind]
    #subj_ps_replacement = ps_replacement[s_ind]
    subj_ps_target = ps_target[s_ind]
    subj_usable_trials = 0
    #for pattern in (condition_pattern):
        #subj_accuracy[pattern] = []
        #for stat in (status):
            #subj_ps[pattern + stat] = []
    # keep track of which new wav file to use
    block_ind = dict(
        zip(range(len(order)), np.zeros(len(order), dtype=int)))

    # get Tab
    # subj_tab = glob.glob(op.join(data_dir, '%s_*.tab' % subj))
    #subj_tab = read_tab(subj_tab[0])

    #iterates over each edf file for each subject session
    #each edf represents one block from runKexp
    # find first edf file by session time and discard 
    fnames.sort()
    # don't include training block
    if (len(fnames) == 10):
        fnames = fnames[1:]
    for b_ind, fname in enumerate(fnames):
        bnum = mid_block_order[b_ind]
        trial_ind = np.zeros(condition_nums)
        print '\tProcessing bnum: ' + str(bnum)
        try:
            raw = pp.raw(fname)
        except:
            print 'Edf file: %s corrupted, skipping' % str(fname)
            continue
        raw_blinks = raw #hold on to blink data
        raw.remove_blink_artifacts()
        assert raw.info['sfreq'] == fs
        screen_coords = raw.info['screen_coords']
        #import pdb; pdb.set_trace()
        if (len(fnames) != s2_blocks):
            print '\t Warning: incorrect edf file count ' + \
            str(len(fnames))

        #block_final_mats = sorted(glob.glob(op.join(data_dir, '%s_*' %
            #subj, 'b'+ str(bnum) + '.*final.mat')))

        # for each trial in this block
        remove_trial = []
        remove_c_ind = []
        for tnum in range(block_len):
            #paradigm = getTrialInfo(block_ind, bnum)
            event_id = s_ind
            remove_flag = False

            #check for a mat file for the trial
            try:
                target_time =\
                        getGeneralInfo(block_ind, bnum, advance=False)
                id_list, correct = getFinalInfo(block_ind, bnum,
                        subj_data_dir, advance=True)
                trial_id = base + ' ' + ' '.join(map(str, id_list))
            except:
                print '\tWarning: trial ' + str(tnum) + \
                        ' not found for bnum ' + \
                        str(bnum) + ' in mat files'
                #advance indexer dict
                remove_flag = True
                block_ind[bnum] += 1

            #check the event for this trial
            events = raw.find_events(trial_id, event_id)
            if (len(events) == 0):
                print '\tWarning: trial ' + str(tnum) + \
                        ' not found for bnum ' + \
                        str(bnum) + ' in edf files'
                remove_flag = True
                block_ind[bnum] += 1
            else:
                print '\t\tProcessing trial ' + str(tnum)
            for pattern in condition_pattern:
                if trial_id.startswith(pattern):
                    cond = pattern
            c_ind = con_2_ind(cond)

            #remove invalid trials
            if remove_flag:
                remove_c_ind.append(c_ind)
                remove_trial.append(tnum)
                subj_accuracy[c_ind, cond_acc_ind[c_ind]] = np.nan
                trial_ind[c_ind] += 1
                continue

            #save base ps data for each subject
            base_epoch = pp.Epochs(raw, events=events, 
                event_id=event_id, tmin=-base_slice_time, tmax=0)
            ctemp = base_epoch.get_data('ps')[0]
            for i in range(int(base_samp)):
                subj_ps_base[c_ind, b_ind, trial_ind[c_ind], i] = ctemp[i] 

            #save target ps data for each subject
            for timeind, ttime in enumerate(target_time):
                tmin = ttime - preslice_time
                tmax = tmin + target_slice_time
                try:
                    #make sure the target window doesn't go over edf file
                    last_sample = events[0][0] + tmax * fs
                    assert(raw.n_samples > last_sample)
                    target_epoch = pp.Epochs(raw, events=events, 
                        event_id=event_id, tmin=tmin, tmax=tmax)
                except:
                    print 'One target lost due to clipping of window'
                    print 'Raw samples: %d end sample: %d '\
                            % (raw.n_samples, last_sample)
                    print 'Overbleed: %d' % (last_sample - raw.n_samples)
                ctemp = target_epoch.get_data('ps')[0]
                for i in range(int(target_samp)):
                    subj_ps_target[c_ind, b_ind, trial_ind[c_ind], timeind, i] = ctemp[i] 

            ##save replacement ps data for each subject
            #if replacement_time is not None:
                #for timeind, rtime in enumerate(replacement_time):
                    #tmin = rtime - preslice_time
                    #replacement_epoch = pp.Epochs(raw, events=events, 
                        #event_id=event_id, tmin=tmin,
                        #tmax=tmin + replacement_slice_time)
                    #ctemp = replacement_epoch.get_data('ps')[0]
                    #for i in range(int(replacement_samp)):
                        #subj_ps_replacement[c_ind, b_ind,
                        #trial_ind[c_ind], timeind, i] = ctemp[i] 

            #check for blink criteria
            base_mean = np.nanmean(subj_ps_base[c_ind, b_ind,
                trial_ind[c_ind]])
            base_std = np.nanstd(subj_ps_base[c_ind, b_ind,
                trial_ind[c_ind]])

            tmin = 0.0
            tmax = trial_len + postblock
            blink_epoch = pp.Epochs(raw_blinks, events=events, 
                event_id = event_id, tmin=tmin, tmax=tmax)
            blink_ps = blink_epoch.get_data('ps')[0]
            valid = len(np.where((blink_ps > (deviations *\
                base_std + base_mean)) & (blink_ps < \
                    (base_mean - deviations * base_std)))) /\
                len(blink_ps)
            remove_flag = (valid > usable_criterion) or remove_flag

            #check for large eyemovements 
            blink_xpos = blink_epoch.get_data('xpos')[0]
            blink_ypos = blink_epoch.get_data('ypos')[0]
            #center it
            #blink_xpos = blink_xpos - ((int)\
                #screen_coords[0] / 2)
            #blink_ypos = blink_ypos - ((int)\
                #screen_coords[1] / 2)
            #TODO get screen distance from Marlo
            #rho = np.sqrt(blink_xpos**2 + blink_ypos**2)
            #eye_angle = np.arctan2(rho / screen_distance)
            #remove_flag = any(eye_angle > angle_criterion) or \
                #remove_flag
            #remove_flag = False or remove_flag

            #remove invalid trials
            if remove_flag:
                remove_c_ind.append(c_ind)
                remove_trial.append(tnum)
                subj_accuracy[c_ind, cond_acc_ind[c_ind]] = np.nan
                trial_ind[c_ind] += 1
                continue

            #otherwise process date normally
            #parse ps data and add to matrices
            trial_epoch = pp.Epochs(raw, events=events, 
                event_id = event_id, tmin=tmin, tmax=tmax)
            temp = trial_epoch.get_data('ps')[0]
            #import pdb;pdb.set_trace()

            #save all other ps and accuracy info
            if correct:
                subj_accuracy[c_ind, cond_acc_ind[c_ind]] = 1
                for i in range(trial_samp):
                    subj_ps[c_ind, b_ind, trial_ind[c_ind], i] = temp[i] 
            else:
                subj_accuracy[c_ind, cond_acc_ind[c_ind]] = 0
                for i in range(trial_samp):
                    subj_ps_incorrect[c_ind, b_ind, trial_ind[c_ind], i] = temp[i] 
            #update indexer
            trial_ind[c_ind] += 1
            cond_acc_ind[c_ind] += 1
            subj_usable_trials += 1
    accuracy[s_ind] = subj_accuracy
    usable_trials[s_ind] = subj_usable_trials
    ps[s_ind] = subj_ps
    ps_incorrect[s_ind] = subj_ps_incorrect
    ps_base[s_ind] = subj_ps_base
    #ps_replacement[s_ind] = subj_ps_replacement
    ps_target[s_ind] = subj_ps_target
    subj_tuple = (subj_accuracy, subj_ps, subj_ps_incorrect,
            subj_ps_base, 
            subj_ps_target,
            subj_usable_trials)
    fsubj = open(processed_file, 'w')
    pck.dump(subj_tuple, fsubj) # overwrites
    fsubj.close()
    
##

resultstxt = open('results.txt', 'w')
#remove past data
resultstxt.truncate()
resultstxt.write('Text file for KEXP stats\n')
resultstxt.write('# of Subjects: %d\n' % N)
resultstxt.write('%s \n \n' % str(datetime.now()))

#Accuracy
acc_tuple = namedtuple('acc_tuple', 'global_mean, global_ste,\
        subj_means, subj_std')

acc_global = subj_accuracy_stats(accuracy)

#compare initial trials to end trials
accuracy_start = accuracy[:,:,:8]
acc_start = subj_accuracy_stats(accuracy_start)
accuracy_end = accuracy[:,:, 18:]
acc_end = subj_accuracy_stats(accuracy_end)

pResults('Accuracy global mean', acc_global.global_mean)
pResults('Accuracy global standard error', acc_global.global_ste)
printSignificant('Accuracy', acc_global.subj_means)

#start
pResults('Accuracy global mean start', acc_start.global_mean)
pResults('Accuracy global standard error start',
        acc_start.global_ste)
#end
pResults('Accuracy global mean end', acc_end.global_mean)
pResults('Accuracy global standard error end', acc_end.global_ste)

#stats on differences between start and end
printSignificantInter('Accuracy between start and end',
        acc_start.subj_means, acc_end.subj_means)

temp = zip(acc_start.subj_means, acc_end.subj_means)
delta_acc = np.array([subj[1] - subj[0] for subj in temp])
delta_acc_mean = np.nanmean(delta_acc, axis=0)
delta_acc_ste = stats.sem(delta_acc, axis=0)

pResults('Delta accuracy global mean', delta_acc_mean)
pResults('Delta Accuracy global standard error', delta_acc_ste)
printSignificant('Delta accuracy sig. testing', delta_acc)

#plot

#barplot('Accuracy', 'Accuracy (%)', 5, acc_global.subj_means,
        #acc_global.global_mean, acc_global.global_ste, yrange=(50, 105))
        #acc_subj_means_start, acc_subj_means_end)

#already saved as final
double_barplot('Accuracy', 'Accuracy (%)', 5,
        acc_start, acc_end, yrange=(60,100), show=True,
        draw_sig=False)

#PS
#trial
#trim all data from end of the visual primer plus 1 cycle
tot_cycs = 5 + 1 # get one more iteration to find end of 5th
cycle_start_sec = np.zeros(tot_cycs)
cycle_start_samp = np.zeros(tot_cycs)
for i in range(tot_cycs):
    cycle_start_sec[i] = preblock_prime_sec +\
            i * cycle_time
    cycle_start_samp[i] = cycle_start_sec[i] * fs
    #print cycle_start_sec[i]

stats_tuple = namedtuple('stats_tuple', 'full_mean_trace,\
        full_mean_bc_trace, full_ste_trace,\
        full_ste_bc_trace, mean_trace,\
        bc_mean_trace, ste_trace, bc_ste_trace, global_mean,\
        global_ste, global_bc_mean, global_bc_ste,\
        ps_subj_means, ps_subj_std,\
        ps_subj_bc_means, ps_subj_bc_std,\
        global_bc_peak_mean, global_mc_peak_mean,\
        global_bc_peak_ste, global_mc_peak_ste,\
        subj_bc_peaks, subj_mean_corrected_peaks')

## Discriminate cycle stats
#which_trials = ['all', 'start', 'end']
#for which_trial in which_trials:
    ##cycle_stats = list()
    ##parse results by cycle count
    #for i in range(tot_cycs - 1):
        #temp = subj_ps_stats(ps, window_start=cycle_start_samp[i],\
            #window_end=cycle_start_samp[i+1], take_trials=which_trial)
        ##cycle_stats.append(temp) 
        #pGroupedResults(temp, 'Cycle %d %s' % ((i + 1),
            #which_trial))

#trial_stats = subj_ps_stats(ps)
task_stats = subj_ps_stats(ps, window_start=cycle_start_samp[1],
    window_end=cycle_start_samp[5])
#primer_stats = subj_ps_stats(ps, window_end=end_primer_samp)
#start stats for 2nd through 5th cycle
start_stats = subj_ps_stats(ps, window_start=cycle_start_samp[1],
    window_end=cycle_start_samp[5], take_trials='start')
#end stats for 2nd through 5th cycle
end_stats = subj_ps_stats(ps, window_start=cycle_start_samp[1],
    window_end=cycle_start_samp[5], take_trials='end')
#target_stats = subj_ps_stats(ps_target, data_type='target',\
    #window_start=preslice_samp)

printSignificantInter('Start vs. end uncorrected raw PS',
        start_stats.ps_subj_means, end_stats.ps_subj_means)

printSignificantInter('Start vs. end bc PS',
        start_stats.ps_subj_bc_means, end_stats.ps_subj_bc_means)

#pGroupedResults(trial_stats, 'full trial')
pGroupedResults(task_stats, 'task')
#pGroupedResults(primer_stats, 'primer')
pGroupedResults(start_stats, 'start')
pGroupedResults(end_stats, 'end')
#pGroupedResults(target_stats, 'target')

##target peak base corrected
#pResults('Pupil global target base corrected peak',
        #target_stats.global_bc_peak_mean)
#pResults('Pupil global target bc standard error',
        #target_stats.global_bc_peak_ste)
#printSignificant('Peak target baseline corrected',
        #target_stats.subj_bc_peaks)

##target peak mean corrected
#pResults('Pupil global target mean corrected peak',
        #target_stats.global_mc_peak_mean)
#pResults('Pupil global target mean corrected standard error',
        #target_stats.global_mc_peak_ste)
##printSignificant('PS target', ps_subj_means_targ)
#printSignificant('Peak target mean corrected',
        #target_stats.subj_mean_corrected_peaks)

#plot bc ps data for all conditions
#plot_ps(task_stats.full_mean_bc_trace, task_stats.full_ste_bc_trace, 'Base corrected')

#fig, (ax1, ax2) = plt.subplots(2, sharex=True)
#plot_ps(start_stats.full_mean_bc_trace,
        #start_stats.full_ste_bc_trace, 'Base corrected initial\
        #trials', ax=ax1)
#plot_ps(end_stats.full_mean_bc_trace,
        #end_stats.full_ste_bc_trace, 'Base corrected final\
        #trials', ax=ax2, final_sub_plot=True)

##already saved final version
#plot_ps(start_stats.full_mean_bc_trace,
        #start_stats.full_ste_bc_trace, 'Initial')
#plot_ps(end_stats.full_mean_bc_trace,
        #end_stats.full_ste_bc_trace, 'Final')

#plot ps data for all conditions
#plot_ps(task_stats.full_mean_trace, task_stats.full_ste_trace, 'Raw')

##plot target ps data for all conditions
#plot_ps(target_stats.full_mean_bc_trace, target_stats.full_ste_bc_trace,\
        #'Base corrected target')

#barplot('Mean pupil size', 'Pupil Size', 50,\
        #ps_subj_means, global_mean, global_ste)

#barplot('Final pupil size', 'Relative pupil size',
        #250, task_stats.ps_subj_bc_means,
        #end_stats.global_bc_mean, end_stats.global_bc_ste)

#barplot('Initial pupil size', 'Relative pupil size',
        #250, task_stats.ps_subj_bc_means,
        #start_stats.global_bc_mean, start_stats.global_bc_ste)

double_barplot('Mean task pupil size', 'Corrected pupil pixel area', 100, 
    start_stats, end_stats, show=True, draw_sig=True,
    yrange=(-100, 800))

##baseline corrected target
#barplot('Peak base corrected target pupil size', 
    #'Relative pupil size', 250, target_stats.subj_bc_peaks,
    #target_stats.global_bc_peak_mean, target_stats.global_bc_peak_ste)

##mean corrected target
#barplot('Peak mean corrected target pupil size', 
   #'Relative pupil size', 50, target_stats.subj_mean_corrected_peaks,
   #target_stats.global_mc_peak_mean, target_stats.global_mc_peak_ste)

#Survey
cog_subj, cog_mean, cog_ste, gen = cleanCogData(weighted=False)

#pResults('Cognitive load unweighted means', cog_mean)
#pResults('Cognitive load unweighted standard error', cog_ste)
#printSignificant('Cognitive load unweighted', cog_subj)

#barplot('Unweighted cognitive load survey', 'Relative demand score\n'+\
#r'low $\hspace{8} \rightarrow \hspace{8}$high', 5,\
        #cog_subj, cog_mean, cog_ste)

##weighted
#cog_subj_weighted, cog_mean_weighted, cog_ste_weighted =\
#cleanCogData(weighted=True)

#pResults('Cognitive load weighted means', cog_mean_weighted)
#pResults('Cognitive load weighted standard error',
        #cog_ste_weighted)
#printSignificant('Cognitive load weighted', cog_subj_weighted)

#barplot('Weighted cognitive load survey', 'Relative demand score\n'+\
#r'low $\hspace{8} \rightarrow \hspace{8}$high', 1,\
        #cog_subj_weighted, cog_mean_weighted, cog_ste_weighted)

# WWL weighted
cog_subj_weighted_WWL, cog_mean_weighted_WWL,\
    cog_ste_weighted_WWL, gen =\
    cleanCogData(weighted=True, weighting_type='WWL')

pResults('Cognitive load WWL weighted means', cog_mean_weighted_WWL)
pResults('Cognitive load WWL weighted standard error',
        cog_ste_weighted_WWL)
printSignificant('Cognitive load WWL weighted',
        cog_subj_weighted_WWL)

#already saved final figure
barplot('NASA TLX survey results', 'Relative demand score\n'+\
    r'low $\hspace{8} \rightarrow \hspace{8}$high', 2,\
    cog_subj_weighted_WWL, cog_mean_weighted_WWL, cog_ste_weighted_WWL,
    show=True, yrange=(0,11))

#Combined data
#correlation_strategies = [stats.pearsonr, stats.spearmanr]
#strategy_names = ['pearson', 'spearman']
correlation_strategies = [stats.spearmanr]
strategy_names = ['spearman']
for si, strategy in enumerate(correlation_strategies):
    strategy_name = strategy_names[si]
    measure_names = ['Accuracy', 'PS', 'Survey']
    subj_combined = np.zeros(shape=(3, N, condition_nums))
#for safety
    subj_combined[:] = np.nan

    subj_combined[0] = acc_global.subj_means
    subj_combined[1] = task_stats.ps_subj_means
    subj_combined[2] = cog_subj_weighted_WWL
    #subj_combined[2] = cog_subj
    combinedSig, combinedP = \
            combinedSigTest('Cross-measure R %s' % strategy_name,
                    subj_combined, strategy)

    subj_combined[0] = delta_acc
    subj_combined[1] = task_stats.ps_subj_means
    subj_combined[2] = cog_subj_weighted_WWL
    #subj_combined[2] = cog_subj
    combinedSig, combinedP = \
            combinedSigTest('Cross-measure R %s delta accuracy' % strategy_name,
                    subj_combined, strategy)

    subj_combined[0] = acc_start.subj_means
    subj_combined[1] = start_stats.ps_subj_means
    combinedSig, combinedP = \
            combinedSigTest('Cross-measure R %s start trials'
                    % strategy_name, subj_combined, strategy)

    subj_combined[0] = acc_end.subj_means
    subj_combined[1] = end_stats.ps_subj_means
    combinedSig, combinedP = \
            combinedSigTest('Cross-measure R %s end trials' %
                    strategy_name, subj_combined, strategy)

resultstxt.close()
print 'Finished... results text file closed\n'
