# author: Karl Marrett
# driver for processing pupillometry data

import os
import glob
from os import path as op
import time
from scipy import io as sio 
from scipy import stats
import pyeparse as pp 
import numpy as np
import matplotlib.pyplot as plt
import pickle as pck
from pyeparse.utils import pupil_kernel
from expyfun import binary_to_decimals, decimals_to_binary

#TODO
#double check stats section of accuracy
#get cog scores settled
    #add in weighting
#finish text
#in general get the mean for each subject
#std and mean of subject means
#read statistics section
#test for significance of results and print to results
#make ps look better

subjects = ['HI', 'HN', 'HL', 'HK', 'HJ'] # HK later
subjects = ['HN', 'HL', 'HK', 'HJ'] 
N = len(subjects)

#assert version code
stim_version_code = 8010
# asserted fs
fs = 1000.0  
#data_dir = os.path.abspath(os.path.join(os.pardir, 'Data'))
data_dir = '/home/kdmarrett/lab/FilesScript/Data'

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

def subj_accuracy_stats():
    #subj_means = [[],[],[]]
    #subj_stds = [[],[],[]]
    subj_means = np.zeros(N, condition_nums)
    subj_stds = np.zeros(N, condition_nums)
    for i, subj_accuracy in enumerate(accuracy):
        for cond in condition_pattern:
            # change binary string into index of the condition type
            num = int(cond[-3:].replace(" ", ""), 2) - 1
            #subj_means[num].append(np.mean(subj_accuracy[cond]))
            #subj_means[num].append(np.mean(subj_accuracy[cond]))
            subj_means[i][num] = np.std(subj_accuracy[cond])
            subj_stds[i][num] = np.std(subj_accuracy[cond])

    #import pdb; pdb.set_trace()
    global_mean = []
    global_std = []
    global_ste = []
    for i in range(condition_nums):
        global_mean.append(np.mean(subj_means[i]))
        global_std.append(np.std(subj_means[i]))
        global_ste.append(global_std[-1] / np.sqrt(N))
    return global_mean, global_std, global_ste, subj_means, subj_stds

def subj_ps_stats(type='correct'):
    pass
    #if type is 'both':
        #types = status
    #else:
        #types = type
    #for stat in types:

def plot_accuracy():
    global_mean, global_std, global_ste, subj_means, subj_stds = \
        subj_accuracy_stats()
    #fig = plt.figure()
    index = np.arange(condition_nums)
    x = [.5, 1.0, 1.5]
    bar_width = .25
    opacity = .4
    global_mean_pc = global_mean * np.tile(100, len(global_mean))
    global_std_pc = global_std * np.tile(100, len(global_std))
    error_config = {'ecolor': 'k', 'elinewidth': 3}
    rects1 = plt.bar(x, global_mean_pc, bar_width, color='w',
            yerr=global_std_pc, error_kw=error_config, lw=2)
    x = x + np.tile(bar_width / 2, condition_nums)
    import pdb;pdb.set_trace()
    for subj_mean in subj_means:
        subj_mean_pc = subj_mean * np.tile(100, len(subj_mean))
        plt.plot(x, subj_mean_pc, color='k', alpha=opacity, marker='o')
        

    #plt.xlabel('Condition')
    plt.ylabel('Accuracy (%)')
    plt.ylim([70, 103])
    #plt.grid(True)
    #plt.grid(b=True, which='minor', color='k', alpha=.9)
    #plt.grid(b=False, which='major', color='k', alpha=.9)
    plt.title('Accuracy by condition')
    plt.xticks(x, ('Alphabetic', 'Fixed-order', 'Random'))
    plt.tight_layout()
    plt.show()
    #change facecolor
    plt.savefig('conditionAccuracy.png', transparent=True,
    edgecolor='none')

def plot_ps(ps_type, name=''):
    min_len = np.inf;
    for i in range(len(ps_type)):
        temp = len(ps_type[i])
        if temp < min_len:
            min_len = temp
        
    #import pdb; pdb.set_trace()
    dat = np.array(np.zeros(min_len*len(ps_type)))
    dat.shape = (len(ps_type), min_len)
    for j in range(len(ps_type)):
        dat[j,:] = ps_type[j]

    stdev = np.array(np.zeros(min_len))
    mean = np.array(np.zeros(min_len))
    for i in range(min_len):
        stdev[i] = np.std(dat[:,i])
        mean[i] = np.mean(dat[:,i])

    fig = plt.figure()
    x = np.linspace(0, min_len / fs, min_len)
    plt.plot(x, mean, 'r', linewidth=3, label='mean', alpha=1)
    plt.plot(x, stdev, 'b--', linewidth=2, label='inter-trial std')
    end_primer = vPrimerLen
    end_primer_samp = int(end_primer * fs)
    end_stim = int(trial_len * fs)
    trial_means = []
    for i in range(len(ps_type)):
        plt.plot(x, ps_type[i], 'k', linewidth=.1, alpha = .4)
        trial_means.append(ps_type[i][end_primer_samp:])
    global_mean = np.mean(trial_means)
    global_std = np.std(trial_means)

    plt.annotate('switch primer to dot', xy=(end_primer, 
        mean[end_primer_samp]), xytext=(5, 6000),
        arrowprops=dict(facecolor='black', shrink=0.05))
    plt.legend(loc=9)    
    plt.ylabel('Pupil Size')
    plt.text(2, global_mean, 'trial means = ' + \
            str(global_mean) + ', std = ' + \
            str(global_std))
    plt.xlabel('Trial Time (s)')
    plt.title(name + ' trial pupil size N = ' + str(len(ps_type)))
    #plt.show()
    name = name.replace(" ", "")
    fn = name + 'ps.png'
    plt.savefig(fn)


# read in global stimuli parameters
param_data_dir = op.join(data_dir, 'Params')
global_vars = sio.loadmat(op.join(param_data_dir, 'global_vars.mat'))
preblock = global_vars['preblock_prime_sec'][0]
# time of visual primer
vPrimerLen = global_vars['vPrimerLen'] 
# time after each trial to record pupil
postblock = global_vars['postblock'] 
#trial_len = global_vars['tot_wav_time'] 
trial_len = 36.0001
order = global_vars['order'][0]
s2_blocks = global_vars['s2_blocks']
# trials_per_block in middle section
block_len = len(order[1][0])
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
ps = []
accuracy = []
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

    cog = getCogResponses(cog, subj_data_dir)

    try:
        fsubj_accuracy = open(subj + '_accuracy.obj', 'r')
        fsubj_ps = open(subj + '_ps.obj', 'r')
        subj_accuracy = pck.load(fsubj_accuracy)
        subj_ps = pck.load(fsubj_ps)
        accuracy.append(subj_accuracy)
        ps.append(subj_ps)
        fsubj_accuracy.close()
        fsubj_ps.close()
        continue
    except:
        pass

    #new data structs
    subj_accuracy = dict()
    subj_ps = dict()
    for pattern in (condition_pattern):
        subj_accuracy[pattern] = []
        for stat in (status):
            subj_ps[pattern + stat] = []
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
        raw.remove_blink_artifacts()
        #raws[bnum] = raw
        assert raw.info['sfreq'] == fs
        if (len(fnames) != s2_blocks):
            print '\t Warning: incorrect edf file count ' + \
            str(len(fnames))

        #block_final_mats = sorted(glob.glob(op.join(data_dir, '%s_*' %
            #subj, 'b'+ str(bnum) + '.*final.mat')))

        # for each trial in this block
        for tnum in range(block_len):
            #paradigm = getTrialInfo(block_ind, bnum)
            #id_list = getGeneralInfo(block_ind, bnum, advance=False)
            event_id = s_ind
            try:
                id_list, correct = getFinalInfo(block_ind, bnum,
                        subj_data_dir, advance=True)
                trial_id = base + ' ' + ' '.join(map(str, id_list))
            except:
                print '\tWarning: trial ' + str(tnum) + \
                        ' not found for bnum ' + \
                        str(bnum) + ' in mat files'
                #advance indexer dict
                block_ind[bnum] += 1
                continue
            #find the event for this trial
            events = raw.find_events(trial_id, event_id)
            if (len(events) == 0):
                print '\tWarning: trial ' + str(tnum) + \
                        ' not found for bnum ' + \
                        str(bnum) + ' in edf files'
                continue
            else:
                print '\t\tProcessing trial ' + str(tnum)
            #parse ps data and add to dicts/matrices
            tmin = 0.0
            tmax = trial_len + postblock
            trial_epoch = pp.Epochs(raw, events=events, 
                event_id=event_id, tmin=tmin, tmax=tmax)
            for pattern in condition_pattern:
                if trial_id.startswith(pattern):
                    cond = pattern
            if correct:
                key = cond + 'correct'
            else:
                key = cond + 'incorrect'
            ps_dict[key].append(trial_epoch.get_data('ps')[0])
            accuracy_dict[cond].append(correct)
            subj_ps[key].append(trial_epoch.get_data('ps')[0])
            subj_accuracy[cond].append(correct)
    accuracy.append(subj_accuracy)
    ps.append(subj_ps)
    fsubj_accuracy = open(subj + '_accuracy.obj', 'w')
    fsubj_ps = open(subj + '_ps.obj', 'w')
    pck.dump(subj_ps, fsubj_ps) # overwrites
    pck.dump(subj_accuracy, fsubj_accuracy) # overwrites
    fsubj_accuracy.close()
    fsubj_ps.close()
    
#process cog load data
gen_qnum = 6
rel_qnum = 15
#assert there are no more zeros at the end
gen = np.zeros((N, condition_nums, gen_qnum))
rel = np.zeros((N, condition_nums, rel_qnum))
load_score = np.zeros((N, condition_nums))
for i in range(N):
    for j in range(condition_nums):
        # wait category responses by rel survey
        for k in range(rel_qnum):
            rel[i, j, k] = \
                cog[i][j]['rel_' + para[j] + '_qnum_' + str(k)][0,0]
        score = 0
        for k in range(gen_qnum):
            response =  cog[i][j]['gen_' + para[j] + '_qnum_' + \
                str(k)][0,0]
            gen[i, j, k] = response
            if k in (3, 5):
                score -= response
            else:
                score += response

plot_accuracy()
#status = ['correct'] #only consider correct trials
#for pattern in condition_pattern:
    #print 'plotting pattern' + pattern
    #for stat in status:
        #ps_type = ps_dict[pattern + stat]
        #cond_num = patternToCond[pattern]
        #name = 'Condition ' + str(cond_num) + ' ' + stat
        #plot_ps(ps_type, name)

