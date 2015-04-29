# author: Karl Marrett
# processes pupillometry data
# modified from pupil_deconv.py on voc_meg experiment by Eric Larson

import os
import glob
from os import path as op
import time
from scipy import io as sio 
import pyeparse as pp 
import numpy as np
import matplotlib.pyplot as plt
import pickle as pck

from pyeparse.utils import pupil_kernel
from expyfun import binary_to_decimals, decimals_to_binary

subjects = ['Karl']

stim_version_code = 8010
# asserted fs
fs = 1000.0  
data_dir = os.path.abspath(os.path.join(os.pardir, 'Data'))

# LOAD IN TRIAL DATA/STIMS
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

def getCogResponses(subj_data_dir):
    # retrieve wav_indices
    id_ = condition_asc[i]
    wav_num = wav_indices[id_]
    wav_indices[id_] += 1
    data_file_name = id_ + '_tr' + str(wav_num)
    trial_data_path = op.join(param_data_dir, data_file_name)
    trial_vars = sio.loadmat(trial_data_path)

# read in global stimuli parameters
param_data_dir = op.join(data_dir, 'Params')
global_vars = sio.loadmat(op.join(param_data_dir, 'global_vars.mat'))
preblock = global_vars['preblock_prime_sec'][0]
# time of visual primer
vPrimerLen = global_vars['vPrimerLen'] 
# time after each trial to record pupil
postblock = global_vars['postblock'] 
trial_len = global_vars['tot_wav_time'] 
order = global_vars['order'][0]
s2_blocks = global_vars['s2_blocks']
# trials_per_block
block_len = len(order[1])
wheel_matrix_info = global_vars['wheel_matrix_info'][0]

# build condition strings
condition_uni = global_vars['condition_bin']  # Unicode by default
condition_asc = []  # ASCII
for i in range(len(condition_uni)):
    condition_asc.append(''.join(map(str, condition_uni[i].tolist())))
#condition_asc.append(condition_uni[i].encode('ascii'))
condition_nums = len(condition_asc)
# keep track of which new wav file to use
block_ind = dict(
    zip(range(len(order)), np.zeros(len(order), dtype=int)))
wav_indices = dict(
    zip(condition_asc, np.zeros(len(condition_asc), dtype=int)))
# build String condition ids for raw 'messages'
base = 'TRIALID'
condition_pattern = [''] * condition_nums;
for i in range(condition_nums):
    condition_pattern[i] = base
    for j in range(len(condition_asc[i])):
        condition_pattern[i] += ' ' + condition_asc[i][j]

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


ps = []
for subj in subjects:
    print('  Subject %s...' % subj)

    # return all edfs files for a given subject
    fnames = sorted(glob.glob(op.join(data_dir, '%s_*' % subj, '*.edf')))
    subj_files = sorted(glob.glob(op.join(data_dir, '%s_*' % subj)))
    for file in subj_files:
        if op.isdir(file):
            subj_data_dir = file

    # check if two dirs for this subject

    # check MATLAB trial parameters with
    # assert (len(fnames) + 1) == len(params['s2_blocks'])
    # subj_tab = glob.glob(op.join(data_dir, '%s_*.tab' % subj))
    # assert len(subj_tab) == 1
    # subj_tab = read_tab(subj_tab[0])

    #iterates over each edf file for each subject session
    #each edf represents one block from runKexp
    epochs = []
    # check that fnames are sorted
    for bnum, fname in enumerate(fnames):
        # hack
        bnum += 1
        raw = pp.Raw(fname)
        assert raw.info['sfreq'] == fs
        #test_events = raw.find_events('SYNCTIME', 1)
        # find first edf file by session time and discard 
        #if (len(test_events) < 4):
            #print '\tInvalid edf found, continuing...\n'
            #continue

        raw.remove_blink_artifacts()
        #import pdb; pdb.set_trace()
        trial_mats = sorted(glob.glob(op.join(data_dir, '%s_*' % subj,
            'b'+ str(bnum) + '.*final.mat')))
        # for each trial in this block
        for tnum in range(block_len):
            #paradigm = getTrialInfo(block_ind, bnum)
            id_list = getGeneralInfo(block_ind, bnum, advance=True)
            #id_list, correct = getFinalInfo(block_ind, bnum,
                    #subj_data_dir)
            trial_id = base + ' '.join(map(str, id_list))
            event_id = 0  # categorize event must be int?
            #find the event for this trial
            event = raw.find_events(trial_id, event_id)
            #parse ps data and add to dicts/matrices
            tmin = 0.0
            tmax = trial_len + postblock
            trial_epoch = pp.Epochs(raw, events=events, 
                event_id=event_id, tmin=tmin, tmax=tmax)
            epochs.append(trial_epoch)
            ps.append(trial_epoch.get_data('ps')[0])


# arrange it into a 2 dimensional array
# get mean and variance for array
# create a way to distinguish sessions and subjects for batch processing

min_len = np.inf;
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

fig = plt.figure()
x = np.linspace(0, min_len / fs, min_len)
plt.plot(x, mean, 'r--', linewidth=4, label='mean')
plt.plot(x, stdev, 'b--', linewidth=2, label='inter-trial std')
for i in range(len(ps)):
    plt.plot(x, ps[i], 'k', linewidth=.1)

# !!
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
