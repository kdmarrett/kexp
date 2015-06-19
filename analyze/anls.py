# author: Karl Marrett
# driver for processing pupillometry data

#import os
import glob
from os import path as op
#import time
from scipy import io as sio 
from scipy import stats
import pyeparse as pp 
import numpy as np
import matplotlib.pyplot as plt
import pickle as pck
#from pyeparse.utils import pupil_kernel
#from expyfun import binary_to_decimals, decimals_to_binary

# if something was changed in the file saving process
force_reprocess = False
#HD misunderstood directions ommitted as subject
#HP substantially corrupted
subjects = ['HL', 'GH', 'GG', 'GN', 'GI', 'HT', 'HI', 'HN', 'HK', 'HJ', 'GR', 'GU'] 


#shorten for debugging
#subjects = ['HD', 'HP']

if force_reprocess:
    reprocess_list = subjects
else:
    reprocess_list = []
N = len(subjects)

#assert version code
stim_version_code = 8010
# asserted fs
fs = 1000.0  
#TODO use remove_blink_artifacts to nullify certain target
#windows use position of eye to clean the results later
#TODO accuracy in first three blocks vs last three blocks
#TODO plot around target windows
#TODO think about degrees of freedom
#TODO units of pupil size

#data_dir = os.path.abspath(os.path.join(os.pardir, 'Data'))
data_dir = '/home/kdmarrett/lab/FilesScript/Data'

def cleanCogData(weighted=False):
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
            score = 0
            for k in range(gen_qnum):
                response =  cog[i][j]['gen_' + para[j] + '_qnum_' + \
                    str(k)][0,0]
                gen[i, j, k] = response
                if weighted:
                    weight = float(rel[i, j, k]) / total_relative_responses
                    response *= weight
                if k in (3, 5):
                    score -= response
                else:
                    score += response
            cog_subj[i, j] = score
    cog_mean = np.nanmean(cog_subj, axis=0)
    cog_ste = stats.sem(cog_subj, axis=0)
    return cog_subj, cog_mean, cog_ste


def simpleaxis(ax):
    """Taken from timday on S.O.,
     Remove the plot frame lines."""
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    #ax.get_xaxis().tick_bottom()
    #ax.get_yaxis().tick_left()

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
    sig_thresh = .05
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
    sig_thresh = .05
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
    replacement_time = trial_vars['replacement_time']
    target_time = trial_vars['target_time']
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
    return id_list

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
    return global_mean, global_ste, subj_means, subj_stds

def subj_ps_stats(global_base_correct=True, type='correct',\
        window_seconds='default'):
    """ ps[subject,cond_ind,block, trial, sample] 
    Params:
        global_base_correct : whether to mean all baseline ps data
        across the whole experiment or to baseline correct based
        individually by trial  
        type : if correct only consider ps data from trials where
        the subject entered correct num of targets"""

    if window_seconds is 'default':
        #include all of task as window
        window_samp = trial_samp - end_primer_samp
    else:
        #only analyze trimmed window
        window_samp = window_seconds * fs
    # dat holds each time trace
    mean_dat = np.zeros((N, condition_nums, trial_samp))
    mean_dat[:] = np.nan
    std_dat = np.zeros((N, condition_nums, trial_samp))
    std_dat[:] = np.nan
    bc_mean_dat = np.zeros((N, condition_nums, trial_samp))
    bc_mean_dat[:] = np.nan

    if global_base_correct:
        base_mean = np.zeros(shape=(N, 1))
    else:
        base_mean = np.zeros(shape=(N, trials_exp))

    base_mean[:] = np.nan
    for s_ind, subj_ps in enumerate(ps):
        for c_ind in range(condition_nums):
            # raw mean stack for each subject and condition
            mean_dat[s_ind, c_ind] = np.nanmean(
                    subj_ps[c_ind].reshape(trials_exp,
                        trial_samp), axis=0)
            #std_dat[s_ind, c_ind] = np.nanstd(
                    #subj_ps[c_ind].reshape(trials_exp,
                        #trial_samp), axis=0)
            #for each subject baseline correct by the pretrial
            #dark period
            #TODO consider the visual primer time as a baseline correction
            if global_base_correct:
                # mean across all trials then subtract that value
                base_mean[s_ind] = np.nanmean(np.nanmean(
                    ps_base[s_ind, c_ind].reshape(block_len
                        * s2_blocks[0][0], base_samp), axis=0))
                bc_mean_dat[s_ind, c_ind] = mean_dat[s_ind,
                        c_ind] - base_mean[s_ind]
            else:
                #for each subject for each trial find the corresponding
                    #baseline to subtract
                base_mean[s_ind, :] = np.nanmean(ps_base[s_ind,
                        c_ind].reshape(block_len
                        * s2_blocks[0][0], base_samp), axis=0)
                bc_mean_dat[s_ind, c_ind] = np.nanmean(
                        subj_ps[c_ind].reshape(trials_exp,
                            trial_samp) - base_mean, axis=0)
                #std_dat[s_ind, c_ind] = np.nanstd(
                        #subj_ps[c_ind].reshape(trials_exp,
                            #trial_samp), axis=0)
                  
    #trim all data from end of the visual primer on
    #std_dat_trim = std_dat[:,:,end_primer_samp:(end_primer_samp+ window_samp)]
    #TODO make a way to include end sample for the trimmed data
    mean_dat_trim = mean_dat[:,:,end_primer_samp:(end_primer_samp+ window_samp)]
    bc_mean_dat_trim = bc_mean_dat[:,:,end_primer_samp:(end_primer_samp+ window_samp)]
    #subject means for sig testing and plotting
    ps_subj_means = np.nanmean(mean_dat_trim, axis=2)
    ps_subj_std = np.nanstd(mean_dat_trim, axis=2)
    ps_subj_bc_means = np.nanmean(bc_mean_dat_trim, axis=2)
    ps_subj_bc_std = np.nanstd(bc_mean_dat_trim, axis=2)
    assert(ps_subj_means.shape == (N, condition_nums))
    # means across all subjects
    #TODO make sure that you have a mean trace for
    #each subject then stats.sem stats.s for each
    #point along that trace across the subjects
    full_mean_trace = np.nanmean(mean_dat, axis=0) 
    full_mean_bc_trace = np.nanmean(bc_mean_dat, axis=0) 
    full_ste_trace = stats.sem(mean_dat, axis=0) 
    full_ste_bc_trace = stats.sem(bc_mean_dat, axis=0) 
    mean_trace = np.nanmean(mean_dat_trim, axis=0) 
    bc_mean_trace = np.nanmean(bc_mean_dat_trim, axis=0)
    #FIXME this may be disrupted by np.nan values
    ste_trace = stats.sem(mean_dat_trim, axis=0)
    bc_ste_trace = stats.sem(bc_mean_dat_trim, axis=0)
    full_ste_test = np.zeros(shape=(condition_nums, trial_samp))
    #testing
    #for c_ind in range(condition_nums):
        #for sample in range(trial_samp):
            #full_ste_test[c_ind, sample] =\
            #stats.sem(mean_dat[:, c_ind, sample])

    #find peak for each subject (abandoned)
    #TODO clean peaks by subject average peak heights each cond
    #TODO get peaks by subject average peak heights each cond
    #temp = np.nanmax(bc_mean_dat_trim, axis=2)
    ##then average those peaks
    #global_bc_peak_mean = np.nanmean(temp)
    #global_bc_peak_std = np.nanstd(temp)
    #temp = np.nanmax(mean_dat_trim, axis=2)
    ##then average peaks across subjects
    #global_peak_mean = np.nanmean(temp, axis=0)
    #global_peak_std = np.nanstd(temp, axis=0)

    #single number for global average for each condition
    global_mean = np.nanmean(mean_trace, axis=1)
    global_ste = stats.sem(mean_trace, axis=1)
    global_bc_mean = np.nanmean(bc_mean_trace, axis=1)
    global_bc_ste = stats.sem(bc_mean_trace, axis=1)

    assert(len(mean_trace) == condition_nums)
    assert(len(global_mean) == condition_nums)
    assert(len(global_bc_mean) == condition_nums)
    assert(len(global_ste) == condition_nums)
    assert(mean_trace.shape[1] == window_samp);
    assert(bc_mean_trace.shape[1] == window_samp)
    assert(ste_trace.shape[1] == window_samp)
    return full_mean_trace, full_mean_bc_trace, full_ste_trace,\
            full_ste_bc_trace, mean_trace,\
            bc_mean_trace, ste_trace, bc_ste_trace, global_mean,\
            global_ste, global_bc_mean, global_bc_ste,\
            ps_subj_means, ps_subj_std,\
            ps_subj_bc_means, ps_subj_bc_std,\
            window_samp

def roundToIncrement(y, y_increment):
    return round(float(y) / y_increment) * y_increment

def barplot(title, ylabel, y_increment, subject_data, global_subj_mean,\
        global_subj_ste, yrange='default'):

    #fig, ax = plt.subplots(figsize=(12, 14)) 
    fig, ax = plt.subplots() 
    # Remove the plot frame lines.
    simpleaxis(ax)
    x = [.5, 1.0, 1.5]
    ax.set_xlim((.3, 2))
    bar_width = .25
    opacity = .4
    #find range
    if yrange is 'default':
        lim_buffer = y_increment
        yrange = np.zeros((2,1))
        yrange[0] = roundToIncrement(np.nanmin(subject_data) -\
            lim_buffer, y_increment)
        yrange[1] = roundToIncrement(np.nanmax(subject_data) +\
                lim_buffer, y_increment)
        #yrange[0] = roundToIncrement(np.nanmin(global_subj_mean) -\
        #np.nanmax(global_subj_ste) - lim_buffer, y_increment)
        #yrange[1] = roundToIncrement(np.nanmax(global_subj_mean) +\
                #np.nanmax(global_subj_ste) + lim_buffer, y_increment)

    #plot yaxis lines
    for y in range(yrange[0], yrange[1], y_increment):  
        ax.plot(range(0,3), [y] * len(range(0,3)), "--",
                lw=0.5, color="black", alpha=0.3, zorder=1) 

    #plot global data
    error_config = {'ecolor': 'k', 'elinewidth': 3, 'ezorder': 5}
    ax.bar(x, global_subj_mean, bar_width, color='w', zorder=3,
            yerr=global_subj_ste, error_kw=error_config, lw=1)
    #shift x to be in center of bars
    x = x + np.tile(bar_width / 2, condition_nums)

    #plot individual subjects
    for subj_mean in subject_data:
        ax.plot(x, subj_mean, color='k', alpha=opacity, lw=1,
                marker='o', zorder=10)

    ax.set_ylabel(ylabel)
    ax.set_ylim(yrange)
    ax.set_title('%s by condition' % title)
    plt.xticks(x, ('Alphabetic', 'Fixed-order', 'Random'))
    #plt.tight_layout()
    plt.show()
    fn = title.replace(" ", "") + '_barplot.pdf'
    print 'Saving figure:\n%s' % fn
    fig.savefig(fn)
    plt.close()

def plot_ps(trace, ste_trace, name):
    """plot a stack of subject mean ps data of all conditions
    Params:   type : can either by a raw mean or a base 
        corrected mean
          length: either trimmed around some window or full
          meaning including the entire trial_samp"""

    fig, ax = plt.subplots()
    #clean the ticks
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    opacity = .10
    local_samp_len = trace.shape[1]
    x = np.linspace(0, local_samp_len / fs, local_samp_len)
    colors = ('r', 'g', 'b')
    for c_num in range(condition_nums):
        ax.plot(x, trace[c_num], color=colors[c_num],
                linewidth=1, label=names[c_num], alpha=1)
        ax.fill_between(x, trace[c_num] - ste_trace[c_num],\
                trace[c_num] + ste_trace[c_num],\
                color=colors[c_num], alpha=opacity)  

    #mark visual_primer if length is full trial
    if local_samp_len == trial_samp:
        ax.text(end_primer / 3, 600, 'Visual\nprimer',\
                size=10) 
        ax.axvspan(0, end_primer, color='k', alpha=.12)
        ax.text(end_primer + fix_dot_time / 4,\
                610, 'Fixation\n   dot',\
                size=10) 
        ax.axvspan(end_primer, preblock_prime_sec,\
                color='k', alpha=.05)
        ax.text(24, 700, 'Task', size=10) 
        #indicate with arrow
        #ax.annotate('End visual primer', xy=(end_primer, 
            #global_mean[c_num]), xytext=(5, 2000),
            #arrowprops=dict(facecolor='black', shrink=0.02))

    ax.legend(loc=4, prop={'size':12})
    ax.set_ylabel('Pupil Size')
    #Render stats to plot
    #info = r'$\mu$=%.1f, $\sigma$=%.3f, N=%d' % \
            #(global_mean[c_num],\
            #global_std[c_num], N)
    #plt.text(20, global_mean[c_num] + 500, info)
    ax.set_xlabel('Trial Time (s)')
    ax.set_xlim((0, local_samp_len / fs))
    ax.set_title(name + ' trial pupil size')
    plt.show()
    name = name.replace(" ", "")
    fn = name + '_trace.pdf'
    print 'Saving figure:\n%s' % fn
    fig.savefig(fn)
    plt.close()

names = ['Alphabetic', 'Fixed-Order', 'Random']
# read in global stimuli parameters
param_data_dir = op.join(data_dir, 'Params')
global_vars = sio.loadmat(op.join(param_data_dir, 'global_vars.mat'))
preblock = global_vars['preblock_prime_sec'][0]
trial_len = 36.000
trial_samp = np.floor(trial_len*fs).astype(int)
# time of visual primer (s)
vPrimerLen = global_vars['vPrimerLen'] 
end_primer = vPrimerLen
end_primer_samp = int(end_primer * fs)
end_stim = int(trial_len * fs)
#seconds until letters start playing
preblock_prime_sec = 13 
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

#date structures
#ps_dict = dict()
#accuracy_dict = dict()
#ps = []
#accuracy = []
#cog = []
#take three
ps = np.ndarray(shape=(N, condition_nums, s2_blocks, block_len,
    trial_samp))
ps[:] = np.nan
ps_incorrect = np.ndarray(shape=(N, condition_nums, s2_blocks,
    block_len, trial_samp))
ps_incorrect[:] = np.nan
accuracy = np.ndarray(shape=(N, condition_nums,
    s2_blocks * block_len / condition_nums))
accuracy[:] = np.nan
base_slice_time = 2
base_samp = np.ceil(base_slice_time * fs)
ps_base = np.ndarray(shape=(N, condition_nums, s2_blocks,
    block_len, base_samp))
ps_base[:] = np.nan
usable_trials = np.zeros(shape=(N, 1))
usable_trials[:] = np.nan

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
                subj_usable_trials) = pck.load(fsubj)
        accuracy[s_ind] = subj_accuracy
        ps[s_ind] = subj_ps
        ps_incorrect[s_ind] = subj_ps_incorrect
        ps_base[s_ind] = subj_ps_base
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
        print '\tProcessing bnum: ' + str(bnum)
        raw = pp.Raw(fname)
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
            #id_list = getGeneralInfo(block_ind, bnum, advance=False)
            event_id = s_ind
            remove_flag = False

            #check for a mat file for the trial
            try:
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
            #TODO target window
            #TODO get area around targets discard if eyeblink
            for pattern in condition_pattern:
                if trial_id.startswith(pattern):
                    cond = pattern
            c_ind = con_2_ind(cond)

            #remove invalid trials
            if remove_flag:
                remove_c_ind.append(c_ind)
                remove_trial.append(tnum)
                subj_accuracy[c_ind, cond_acc_ind[c_ind]] = np.nan
                for i in range(trial_samp):
                    subj_ps[c_ind, b_ind, tnum, i] = np.nan
                continue

            #save base ps data for each subject
            base_epoch = pp.Epochs(raw, events=events, 
                event_id=event_id, tmin=-base_slice_time, tmax=0)
            ctemp = base_epoch.get_data('ps')[0]
            for i in range(int(base_samp)):
                subj_ps_base[c_ind, b_ind, tnum, i] = ctemp[i] 

            #check for blink criteria
            base_mean = np.nanmean(subj_ps_base[c_ind, b_ind, tnum])
            base_std = np.nanstd(subj_ps_base[c_ind, b_ind, tnum])

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
            remove_flag = False or remove_flag

            #remove invalid trials
            if remove_flag:
                remove_c_ind.append(c_ind)
                remove_trial.append(tnum)
                subj_accuracy[c_ind, cond_acc_ind[c_ind]] = np.nan
                for i in range(trial_samp):
                    subj_ps[c_ind, b_ind, tnum, i] = np.nan
                continue

            #otherwise process date normally
            #parse ps data and add to matrices
            trial_epoch = pp.Epochs(raw, events=events, 
                event_id = event_id, tmin=tmin, tmax=tmax)
            temp = trial_epoch.get_data('ps')[0]

            #save all other ps and accuracy info
            if correct:
                subj_accuracy[c_ind, cond_acc_ind[c_ind]] = 1
                for i in range(trial_samp):
                    subj_ps[c_ind, b_ind, tnum, i] = temp[i] 
            else:
                subj_accuracy[c_ind, cond_acc_ind[c_ind]] = 0
                for i in range(trial_samp):
                    subj_ps_incorrect[c_ind, b_ind, tnum, i] = temp[i] 
            #update indexer
            cond_acc_ind[c_ind] += 1
            subj_usable_trials += 1
    accuracy[s_ind] = subj_accuracy
    usable_trials[s_ind] = subj_usable_trials
    ps[s_ind] = subj_ps
    ps_incorrect[s_ind] = subj_ps_incorrect
    ps_base[s_ind] = subj_ps_base
    subj_tuple = (subj_accuracy, subj_ps, subj_ps_incorrect,
            subj_ps_base, subj_usable_trials)
    fsubj = open(processed_file, 'w')
    pck.dump(subj_tuple, fsubj) # overwrites
    fsubj.close()
    
##

resultstxt = open('results.txt', 'w')
#remove past data
resultstxt.truncate()
resultstxt.write('Text file for KEXP stats\n')
resultstxt.write('# of Subjects: %d\n \n' % N)

#Accuracy
acc_global_mean, acc_global_ste,\
    acc_subj_means, acc_subj_stds = subj_accuracy_stats(accuracy)

#compare initial trials to end trials
accuracy_start = accuracy[:,:,:8]
acc_global_mean_start, acc_global_ste_start, acc_subj_means_start,\
acc_subj_stds_start =  subj_accuracy_stats(accuracy_start)
accuracy_end = accuracy[:,:, 18:]
acc_global_mean_end, acc_global_ste_end,\
        acc_subj_means_end,\
acc_subj_stds_end = subj_accuracy_stats(accuracy_end)

pResults('Accuracy global mean', acc_global_mean)
pResults('Accuracy global standard error', acc_global_ste)
printSignificant('Accuracy', acc_subj_means)
#start
pResults('Accuracy global mean start', acc_global_mean_start)
pResults('Accuracy global standard error start',
        acc_global_ste_start)
#end
pResults('Accuracy global mean end', acc_global_mean_end)
pResults('Accuracy global standard error end', acc_global_ste_end)
#stats on differences between start and end
printSignificantInter('Accuracy between start and end',
        acc_subj_means_start, acc_subj_means_end)

#convert to percent
global_mean_pc = acc_global_mean * np.tile(100, len(acc_global_mean))
global_ste_pc = acc_global_ste * np.tile(100, len(acc_global_ste))
acc_subj_means_pc = acc_subj_means * 100

#start
global_mean_start_pc = acc_global_mean_start * np.tile(100,
        len(acc_global_mean_start))
global_ste_start_pc = acc_global_ste_start * np.tile(100, 
        condition_nums)
acc_subj_means_start_pc = acc_subj_means_start * 100

#end
global_mean_end_pc = acc_global_mean_end * np.tile(100,
        condition_nums)
global_ste_end_pc = acc_global_ste_end * np.tile(100,
        condition_nums)
acc_subj_means_end_pc = acc_subj_means_end * 100


#plot
barplot('Accuracy', 'Accuracy (%)', 5, acc_subj_means_pc,
        global_mean_pc, global_ste_pc, yrange=(50, 105))

#TODO write this
#double_barplot('Accuracy', 'Accuracy (%)', 5,
        #acc_subj_means_start_pc, global_mean_start_pc,
        #global_ste_start_pc, acc_subj_means_end_pc,
        #global_mean_end_pc, global_ste_end_pc, yrange=(50, 105))

#PS
full_mean_trace, full_mean_bc_trace, full_ste_trace,\
        full_ste_bc_trace, mean_trace,\
        bc_mean_trace, ste_trace, bc_ste_trace, global_mean,\
        global_ste, global_bc_mean, global_bc_ste,\
        ps_subj_means, ps_subj_std,\
        ps_subj_bc_means, ps_subj_bc_std,\
        window_samp = subj_ps_stats();

pResults('Pupil global means', global_mean)
pResults('Pupil global standard error', global_ste)
pResults('Pupil global bc means', global_bc_mean)
pResults('Pupil global bc standard error', global_bc_ste)
#printSignificant('PS', ps_subj_means)
printSignificant('PS baseline corrected', ps_subj_bc_means)

#plot ps data for all conditions
plot_ps(full_mean_bc_trace, full_ste_bc_trace, 'Base corrected')

#barplot('Mean pupil size', 'Pupil Size', 50,\
        #ps_subj_means, global_mean, global_ste)

#FIXME what are the units of pupil size?
#baseline corrected
barplot('Mean base corrected pupil size', 'Relative pupil size',
        100, ps_subj_bc_means, global_bc_mean, global_bc_ste)

#Survey
#TODO check that high is high demanding
cog_subj, cog_mean, cog_ste = cleanCogData(weighted=False)

pResults('Cognitive load unweighted means', cog_mean)
pResults('Cognitive load unweighted standard error', cog_ste)
printSignificant('Cognitive load unweighted', cog_subj)

barplot('Unweighted cognitive load survey', 'Relative demand\n'+\
r'low $\hspace{8} \rightarrow \hspace{8}$high', 5,\
        cog_subj, cog_mean, cog_ste)

#weighted
cog_subj_weighted, cog_mean_weighted, cog_ste_weighted = cleanCogData(weighted=True)

pResults('Cognitive load weighted means', cog_mean_weighted)
pResults('Cognitive load weighted standard error',
        cog_ste_weighted)
printSignificant('Cognitive load weighted', cog_subj_weighted)

barplot('Weighted cognitive load survey', 'Relative demand\n'+\
r'low $\hspace{8} \rightarrow \hspace{8}$high', 5,\
        cog_subj_weighted, cog_mean_weighted, cog_ste_weighted)

resultstxt.close()
print 'results text file closed\n'

########
## ABANDONED APPROACHES

##plot ps for each condition individually
#status = ['correct'] #only consider correct trials
#for c_ind, pattern in enumerate(condition_pattern):
    ##print 'plotting pattern' + pattern
    #for stat in status:
        #ps_type = ps_dict[pattern + stat]
        #cond_num = patternToCond[pattern]
        ##name = 'Condition ' + str(cond_num) + ' ' + stat
        #name = names[cond_num - 1]
        ##import pdb; pdb.set_trace()
        #plot_ps_condition(c_ind, name)

#peaks

#pResults('Pupil global peak means', ps_global_peak_mean)
#pResults('Pupil global bc peak means', ps_global_bc_peak_mean)
#pResults('Pupil global bc peak std', ps_global_bc_peak_std)
#pResults('Pupil global peak std', ps_global_peak_std)
#plot_ps_peaks(type='mean')
#plot_ps_peaks(type='bc_mean')

#def plot_ps_peaks(type='mean', name=''):
    #"""Creates a bar graph showing the peak ps size for each
    #condition with standard deviations """

    ##Common sizes: (10, 7.5) and (12, 9)  
    #plt.figure(figsize=(12, 14))  
      
    ## Remove the plot frame lines.
    ##fig = plt.figure()
    ##ax = plt.gca()
    ##fig.spines["top"].set_visible(False)  
    ##fig.spines["bottom"].set_visible(False)  
    ##fig.spines["right"].set_visible(False)  
    ##fig.spines["left"].set_visible(False) 
    #x = [.5, 1.0, 1.5]
    #bar_width = .25
    #error_config = {'ecolor': 'k', 'elinewidth': 3, 'ezorder': 5}
    #if type is 'mean':
        #plt.bar(x, ps_global_peak_mean, bar_width, color='w',
                #yerr=ps_global_peak_std,
                #error_kw=error_config, lw=2)
        #name = 'Mean'
    #elif type is 'bc_mean':
        #plt.bar(x, ps_global_bc_peak_mean, bar_width,
                #color='w', yerr=ps_global_bc_peak_std,
                #error_kw=error_config, lw=2)
        #name = 'bc mean'
    #else:
        #print 'Undefined type'
        #return

    ##yrange = (50, 103)
    ##plt.ylim(yrange)
    ##for y in range(50, 103, 5):  
        ##plt.plot(range(0,3), [y] * len(range(0,3)), "--",
                ##lw=0.5, color="black", alpha=0.3) 
    #plt.title(name + 'peak pupil size')
    #plt.xticks(x, ('Alphabetic', 'Fixed-order', 'Random'))
    #plt.tight_layout()
    ##plt.show()
    #plt.ylabel('Pupil Size')
    ##info = r'$\mu$=%.1f, $\sigma$=%.3f, N=%d' % (global_mean[c_num],\
            ##global_std[c_num], N)
    ##plt.text(20, global_mean[c_num] + 500, info)
    ##plt.show()
    #name = name.replace(" ", "")
    #fn = name + 'peakPS.pdf'
    #print 'Saving figure: %s' % fn
    #plt.savefig(fn)
    #plt.close()

