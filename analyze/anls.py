# author: Karl Marrett
# processes pupillometry data
# modified from pupil_deconv.py on voc_meg experiment by Eric Larson

import os
import glob
from os import path as op
import time
import scipy 
import pyeparse as pp 
import numpy as np
import pdb
import matplotlib.pyplot as plt

from pyeparse.utils import pupil_kernel
from expyfun import binary_to_decimals  # ,decimals_to_binary

#subjects = ['Karl_2014-12-18 19', 'Karl_2014-12-18 13', 'Karl_2014-11-30']
subjects = ['Karl_2014-12-18 19', 'Karl_2014-12-18 13']
data_dir = op.join(os.pardir, 'Data')
fs = 1000.0  
postblock = 5  # time after each trial to record pupil
session = 3
blocks = 3
datadir = os.path.abspath(os.path.join(os.pardir, 'Data'))


def getTrialInfo(block_ind, bnum):
    # LOAD IN TRIAL DATA/STIMS
	data_file_name = 'b' + str(bnum) + '_tr' + block_ind[bnum] + 'final'
    trial_data_path = op.join(final_datadir, data_file_name)
    trial_vars = scipy.io.loadmat(trial_data_path)
    condition_no = trial_vars['paradigm'][0]
    target_time = trial_vars['target_time']
    target_cycles = trial_vars['target_cycles']
    target_letter = trial_vars['target_letter'][0][0][0].encode('ascii')
    location_code = trial_vars['location_code'][0][0][0].encode('ascii')
    possible_letters = trial_vars['possible_letters'][0]
    #id_ = condition_asc[condition_no]
    # id_ = decimals_to_binary(id_, np.ones(1, len(id_)))
    return condition_no, target_time, location_code, correct 

# READ IN PARTICPANT SESSION VARIABLES FROM MAT FILE
# Reads in 'condition_bin', 'wheel_matrix_info', 'preblock_prime_sec'
ps = []
for subj in subjects:
    print('  Subject %s...' % subj)
    final_datadir = op.join(datadir,'Params', 'Karl', str(session))
    final_data_dir = op.join(data_dir, 'Params', 'Karl', str(session))
    global_vars = scipy.io.loadmat(op.join(final_data_dir, 'global_vars.mat'))
    condition_uni = global_vars['condition_bin']  # Unicode by default
    order = global_vars['order'][0]
    #trials = global_vars['trials_per_condition'][0]
    trials = 3; # temp
    condition_asc = []  # ASCII
    for i in range(len(condition_uni)):
        condition_asc.append(condition_uni[i].encode('ascii'))
    condition_nums = len(condition_asc)
    wheel_matrix_info = global_vars['wheel_matrix_info'][0]
    # keep track of which new wav file to use
    block_ind = dict(
        zip(range(len(order)), np.zeros(len(order), dtype=int)))
    wav_indices = dict(
        zip(condition_asc, np.zeros(len(condition_asc), dtype=int)))
    preblock = global_vars['preblock_prime_sec'][0]
    # build String condition ids for raw 'messages'
    base = 'TRIALID'
    condition_pattern = [''] * condition_nums;
    for i in range(condition_nums):
        condition_pattern[i] = base
        for j in range(len(condition_asc[i])):
            condition_pattern[i] += ' ' + condition_asc[i][j]

    # return all edfs files for a given subject
    fnames = sorted(glob.glob(op.join(data_dir, '%s_*' % subj, '*.edf')))

    # Add assertions to check MATLAB trial parameters with
    # assert len(fnames) == len(params['block_trials'])
    # subj_tab = glob.glob(op.join(data_dir, '%s_*.tab' % subj))
    # assert len(subj_tab) == 1
    # subj_tab = read_tab(subj_tab[0])

    #iterates over each edf file for specified session of subject
    #each edf represents one block from runKexp
    epochs = []
    for ri, fname in enumerate(fnames):
        raw = pp.Raw(fname)
        assert raw.info['sfreq'] == fs
        test_events = raw.find_events('SYNCTIME', 1)
        if (len(test_events) == 0):
            print '\tInvalid edf found, continuing...\n'
            continue

        raw.remove_blink_artifacts()
        # extract by condition 
        for i in range(condition_nums):
            id_ = condition_asc[i]
            event_id = 1  # categorize event by
            block_events = raw.find_events(condition_pattern[i],
                event_id)
            #assert block_events == trials, """events found and trials doing
                #match for condition: """ + str(i + 1)
            for j in range(len(block_events)):
                #events = block_events[j];
                events = block_events
                # identify paradigm and trial number
                # retrieve wav_indices
                wav_num = wav_indices[id_]
                wav_indices[id_] += 1
                data_file_name = id_ + '_tr' + str(wav_num)
                trial_data_path = op.join(final_datadir, data_file_name)

                # UNLOAD TRIAL VARS (AVOIDS AWKWARD INDEXING) Creates dict
                # of vars: 'target_letter', 'target_time', 'tot_wav_time',
                # 'paradigm', 'possible_letters' 'tot_cyc'
                trial_vars = scipy.io.loadmat(trial_data_path)
                target_time = trial_vars['target_time']
                trial_len = trial_vars['tot_wav_time'][0][0]
                target_letter = \
                    trial_vars['target_letter'][0][0][0].encode('ascii')
                # check loading of correct mat file
                assert (trial_vars['paradigm'][0].encode('ascii') == \
                        id_), \
                "Error: id_ and paradigm from mat file do not match"
                print('    Epoching: ' + condition_pattern[i])
                tmin = 0.0
                tmax = trial_len + postblock
                if j == 0:
                    trial_epoch = pp.Epochs(raw, events=events, 
                        event_id=event_id, tmin=tmin, tmax=tmax)

            epochs.append(trial_epoch)
            ps.append(trial_epoch.get_data('ps')[0])

# arrange it into a 2 dimensional array
# get mean and variance for array
# keep in mind that the pupil data goes longer than the trial length
# draw the arrow object for the start of primer, end of primer, start of
# stimulus end of stimulus
# create a way to distinguish sessions and subjects for batch processing

min_len = 9999999;
for i in range(len(ps)):
    temp = len(ps[i])
    if temp < min_len:
        min_len = temp
    
dat = np.array(np.zeros(min_len*len(ps)))
dat.shape = (len(ps), min_len)
for j in range(len(ps)):
    dat[j,:] = ps[j]

stdev = np.array(np.zeros(min_len))
mean = np.array(np.zeros(min_len))
for i in range(min_len):
    stdev[i] = np.std(dat[:,i])
    mean[i] = np.mean(dat[:,i])

#min_len = len(ps[0])
fig = plt.figure()
x = np.linspace(0, min_len / fs, min_len)
plt.plot(x, mean, 'r--', linewidth=4, label='mean')
plt.plot(x, stdev, 'b--', linewidth=2, label='inter-trial std')
for i in range(len(ps)):
    plt.plot(x, ps[i], 'k', linewidth=.1)

end_primer = preblock / 3;
end_primer_samp = int(end_primer * fs)
end_stim = int(trial_len * fs)
plt.annotate('switch primer to dot', xy=(end_primer, 
    mean[end_primer_samp]), xytext=(5, 6000),
    arrowprops=dict(facecolor='black', shrink=0.05))
plt.annotate('end of stimuli', xy=(trial_len, 
    mean[end_stim]), xytext=(trial_len, 6000),
    arrowprops=dict(facecolor='black', shrink=0.05))
plt.legend(loc=9)    
plt.ylabel('Pupil Size')
plt.xlabel('Trial Time (s)')
plt.title('Trial Pupil Size N = ' + str(len(ps)))
plt.show()
plt.savefig('trialPS.pdf')
