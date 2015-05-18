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
# resegregate correct and incorrect trials
# take only sections of trial after vprimerlen
#double check stats section of accuracy
#get cog scores settled
    #add in weighting
#finish text
#in general get the mean for each subject
#std and mean of subject means
#test for significance of results and print to results

subjects = ['HL', 'HI', 'HN', 'HK', 'HJ', 'GR'] # HK later
#reprocess_list = ['HI', 'HN', 'HL', 'HK', 'HJ', 'GR'] # HK later
reprocess_list = []
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
    subj_means = np.zeros((N, condition_nums), dtype=np.double)
    subj_stds = np.zeros((N, condition_nums), dtype=np.double)
    for i, subj_accuracy in enumerate(accuracy):
        for c_ind in range(condition_nums):
            # change binary string into index of the condition type
            #num = int(cond[-3:].replace(" ", ""), 2) - 1
            subj_means[i][c_ind] = np.nanmean(subj_accuracy[c_ind])
            subj_stds[i][c_ind] = np.nanstd(subj_accuracy[c_ind])

    global_mean = []
    global_std = []
    global_ste = []
    for i in range(condition_nums):
        #across all subjects for each condition 
        global_mean.append(np.nanmean(subj_means[:, i]))
        global_std.append(np.nanstd(subj_means[:, i]))
        global_ste.append(global_std[-1] / np.sqrt(N))
    assert(len(global_mean) == condition_nums)
    #import pdb;pdb.set_trace()
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

    #Common sizes: (10, 7.5) and (12, 9)  
    plt.figure(figsize=(12, 14))  
      
    # Remove the plot frame lines.
    #fig = plt.figure()
    #ax = plt.gca()
    #fig.spines["top"].set_visible(False)  
    #fig.spines["bottom"].set_visible(False)  
    #fig.spines["right"].set_visible(False)  
    #fig.spines["left"].set_visible(False) 
    x = [.5, 1.0, 1.5]
    bar_width = .25
    opacity = .4
    global_mean_pc = global_mean * np.tile(100, len(global_mean))
    global_std_pc = global_std * np.tile(100, len(global_std))
    error_config = {'ecolor': 'k', 'elinewidth': 3, 'ezorder': 5}
    rects1 = plt.bar(x, global_mean_pc, bar_width, color='w',
            yerr=global_std_pc, error_kw=error_config, lw=2)
    x = x + np.tile(bar_width / 2, condition_nums)
    for subj_mean in subj_means:
        subj_mean_pc = subj_mean * np.tile(100, len(subj_mean))
        plt.plot(x, subj_mean_pc, color='k', alpha=opacity, 
                marker='o')

    #plt.xlabel('Condition')
    plt.ylabel('Accuracy (%)')
    yrange = (50, 103)
    plt.ylim(yrange)
    for y in range(50, 103, 5):  
        plt.plot(range(0,3), [y] * len(range(0,3)), "--",
                lw=0.5, color="black", alpha=0.3) 
    plt.title('Accuracy by condition')
    plt.xticks(x, ('Alphabetic', 'Fixed-order', 'Random'))
    plt.tight_layout()
    #plt.show()
    #change facecolor
    #change facecolor
    plt.savefig('conditionAccuracy.png')
    plt.close()
    #plt.savefig('conditionAccuracy.png', transparent=True,
    #edgecolor='none')

def plot_ps(c_num, name=''):
    """ ps[subject,cond_ind,block, trial, sample] """

    mean_dat = np.zeros((N, condition_nums, trial_samp))
    std_dat = np.zeros((N, condition_nums, trial_samp))
    for s_ind, subj_ps in enumerate(ps):
        for c_ind in range(condition_nums):
            mean_dat[s_ind, c_ind] = np.nanmean(
                    subj_ps[c_ind].reshape(block_len*s2_blocks[0][0],
                        trial_samp), axis=0)
            std_dat[s_ind, c_ind] = np.nanstd(
                    subj_ps[c_ind].reshape(block_len*s2_blocks[0][0],
                        trial_samp), axis=0)
    mean = np.nanmean(mean_dat, axis=0)
    std = np.nanstd(std_dat, axis=0)
    assert(len(std) == condition_nums)
    global_mean = np.nanmean(mean[c_num])
    global_std = np.nanstd(std[c_num])

    #import pdb;pdb.set_trace()
    #fig = plt.figure()
    window_len = len(mean_dat[0][0]) 
    x = np.linspace(0, window_len / fs, window_len)

    plt.fill_between(x, mean[c_num] - std[c_num],  
                            mean[c_num] + std[c_num],
                            color="#3F5D7D", alpha=.5)  
    plt.plot(x, mean[c_num], color='k', linewidth=3, label='mean',
            alpha=1)
    plt.annotate('End visual primer', xy=(end_primer, 
        mean[c_num, end_primer_samp]), xytext=(5, 2000),
        arrowprops=dict(facecolor='black', shrink=0.02))
    #plt.legend(loc=9)    
    plt.ylabel('Pupil Size')
    info = r'$\mu$=%.1f, $\sigma$=%.3f, N=%d' % (global_mean,
            global_std, N)
    plt.text(20,global_mean+500,info)
    plt.xlabel('Trial Time (s)')
    plt.title(name + ' trial pupil size')
    #plt.show()
    name = name.replace(" ", "")
    fn = name + 'ps.png'
    plt.savefig(fn)
    plt.close()


# read in global stimuli parameters
param_data_dir = op.join(data_dir, 'Params')
global_vars = sio.loadmat(op.join(param_data_dir, 'global_vars.mat'))
preblock = global_vars['preblock_prime_sec'][0]
trial_len = 36.000
trial_samp = np.floor(trial_len*fs).astype(int)
# time of visual primer
vPrimerLen = global_vars['vPrimerLen'] 
end_primer = vPrimerLen
end_primer_samp = int(end_primer * fs)
end_stim = int(trial_len * fs)
# time after each trial to record pupil
postblock = global_vars['postblock'] 
#trial_len = global_vars['tot_wav_time'] 
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

def con_2_ind(pattern):
    return int(pattern[-3:].replace(" ", ""), 2) - 1

#date structures
#ps_dict = dict()
#accuracy_dict = dict()
#ps = []
#accuracy = []
#cog = []
#take three
ps = np.ndarray(shape=(N, condition_nums, s2_blocks, block_len,
    trial_samp))
accuracy = np.ndarray(shape=(N, condition_nums, s2_blocks * block_len / \
        condition_nums))
#process cog load data
gen_qnum = 6
rel_qnum = 15
#assert there are no more zeros at the end
gen = np.ndarray(shape=(N, condition_nums, gen_qnum))
rel = np.ndarray(shape=(N, condition_nums, rel_qnum))
load_score = np.ndarray(shape=(N, condition_nums))

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
    #assert it is not empty

    cog = getCogResponses(cog, subj_data_dir)

    try:
        assert(subj not in reprocess_list)
        fsubj_accuracy = open(subj + '_accuracy.obj', 'r')
        fsubj_ps = open(subj + '_ps.obj', 'r')
        subj_accuracy = pck.load(fsubj_accuracy)
        subj_ps = pck.load(fsubj_ps)
        accuracy[s_ind] = subj_accuracy
        ps[s_ind] = subj_ps
        fsubj_accuracy.close()
        fsubj_ps.close()
        continue
    except:
        #(re)process subject
        pass

    #new data structs
    subj_accuracy = accuracy[s_ind]
    subj_ps = ps[s_ind]
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
        raw.remove_blink_artifacts()
        #raws[bnum] = raw
        assert raw.info['sfreq'] == fs
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
            #find the event for this trial
            events = raw.find_events(trial_id, event_id)
            if (len(events) == 0):
                print '\tWarning: trial ' + str(tnum) + \
                        ' not found for bnum ' + \
                        str(bnum) + ' in edf files'
                remove_flag = True
                block_ind[bnum] += 1
            else:
                print '\t\tProcessing trial ' + str(tnum)
            #get c_ind
            for pattern in condition_pattern:
                if trial_id.startswith(pattern):
                    cond = pattern
            c_ind = con_2_ind(cond)
            if remove_flag:
                remove_c_ind.append(c_ind)
                remove_trial.append(tnum)
                subj_accuracy[c_ind, cond_acc_ind[c_ind]] = np.nan
                for i in range(trial_samp):
                    subj_ps[c_ind, b_ind, tnum, i] = np.nan
                continue
            #parse ps data and add to matrices
            tmin = 0.0
            tmax = trial_len + postblock
            trial_epoch = pp.Epochs(raw, events=events, 
                event_id=event_id, tmin=tmin, tmax=tmax)
            temp = trial_epoch.get_data('ps')[0]
            for i in range(trial_samp):
                subj_ps[c_ind, b_ind, tnum, i] = temp[i] 
            #import pdb; pdb.set_trace()
            if correct:
                subj_accuracy[c_ind, cond_acc_ind[c_ind]] = 1
            else:
                subj_accuracy[c_ind, cond_acc_ind[c_ind]] = 0
            cond_acc_ind[c_ind] += 1

        #must be removed from highest ind to lowest for prop indexing
        #remove_c_ind.sort(reverse=True)
        #remove_trial.sort(reverse=True)
            ##remove null trials from np array
        #for i in range(len(remove_trial)):
            #block = subj_ps[remove_c_ind[i], b_ind]
            #subj_ps[remove_c_ind[i], b_ind]= np.delete(block,
                    #block[remove_trial[i]], axis=0)
            #subj_accuracy = np.delete(subj_accuracy, subj_accuracy[
                #remove_c_ind[i], remove_trial[i]], axis=1)

    accuracy[s_ind] = subj_accuracy
    ps[s_ind] = subj_ps
    fsubj_accuracy = open(subj + '_accuracy.obj', 'w')
    fsubj_ps = open(subj + '_ps.obj', 'w')
    pck.dump(subj_ps, fsubj_ps) # overwrites
    pck.dump(subj_accuracy, fsubj_accuracy) # overwrites
    fsubj_accuracy.close()
    fsubj_ps.close()
    
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
status = ['correct'] #only consider correct trials
names = ['Alphabetic', 'Fixed-Order', 'Random']
for c_ind, pattern in enumerate(condition_pattern):
    #print 'plotting pattern' + pattern
    for stat in status:
        ps_type = ps_dict[pattern + stat]
        cond_num = patternToCond[pattern]
        #name = 'Condition ' + str(cond_num) + ' ' + stat
        name = names[cond_num - 1]
        plot_ps(c_ind, name)

