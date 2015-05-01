"""
================================
Script ' Karl's Experiment (Kexp)'
================================
    
This script runs an experiment with spatially distributed letter streams.

Author: Karl Marrett <kdmarret@uw.edu>, <kdmarrett@gmail.com>
"""

#import timeit
from scipy import io as sio
import sys
import pyglet
from math import pi
import pickle as pck
import numpy as np
import math
import os.path as op
import expyfun as ef
from expyfun.io import read_wav
from expyfun import (EyelinkController, visual, get_keyboard_input,
        assert_version, decimals_to_binary)
from CircularLayout import CircularLayout
import os

assert ef.__version__ == '2.0.0.dev'
# assert version of stimuli to use
stim_version_code = 1691

PATH = os.path.abspath(os.pardir)
datadir = op.join(PATH, 'Data')

# GLOBAL VARIABLES
debug = True
skipTrain = False
wait_brief = .2
wait_long = 2
msg_dur = 3.0
postblock = 0  # time after each trial to record pupil
vPrimerLen = 7

# EC PARAMETERS
total_btns = 10
gen_survey_btn = range(1, total_btns)
rel_survey_btn = [1, 2]
#rel_survey_btn = ['num_1', 'num_2']
#gen_survey_btn = ['num_1', 'num_2', 'num_3', 
    #'num_4', 'num_5', 'num_6', 'num_7', 'num_8', 'num_9']
#resp_dict = dict()
#for i in range(1, total_btns):
    #resp_dict[gen_survey_btn[i]] = i

std_args = ['kexp']
std_kwargs = dict(screen_num=0, window_size=[800, 600], check_rms=None,
    full_screen=True, stim_db=65, noise_db=40, stim_rms=0.01,
    #session='1',participant='new',  
    suppress_resamp=False, response_device='keyboard',
    output_dir=datadir, stim_fs=24414)  

def drawPrimer(wheel_matrix_info, target_letter, possible_letters):
    """ Creates the primer letters and draws to visual buffer """

    # get center locations of each wheel
    wheel_loc = CircularLayout(3, radius=.50, angles=[- pi, (pi / 2), 0])
    # get individual letter locations
    letter_loc = []
    for i in range(len(wheel_matrix_info)):
        letters_wheel = wheel_matrix_info[i]
        temp = CircularLayout( letters_wheel, radius=.20,
                relative_center=wheel_loc.positions[i])
        letter_loc.extend(temp.positions)
    # draw selections to visual buffer
    for i in range(len(possible_letters)):
        letter = possible_letters[i][0].encode('ascii')
        if (letter == target_letter):
            color = 'Lime'
        else:
            color = 'LightGray'
        ec.screen_text( letter, pos=letter_loc[i], color=color,
                font_size=30, wrap=False)

def cogLoadSurvey(gen_survey, mid_survey, rel_survey, paradigm,
        edf_outputdir, tnum, ec):
    """ Prompt users with cognitive load questions for each condition.
    Saves a separate mat file for each paradigm with keys also
    identified by paradigm and qnum."""
    
    resp_dict = dict()
    ec.write_data_line('cogLoadSurvey', paradigm)
    ec.write_data_line('gen_survey', paradigm)
    if (tnum == 0):
        ec.screen_prompt(instr['cog_orient'])

    for qnum in dict.keys(gen_survey):
        #key by paradigm for safety
        key = 'gen_' + paradigm + '_qnum_' + str(qnum)
        response = getInput(gen_survey_btn, ec, double_check=True,
                text=gen_survey[qnum])
        ec.write_data_line(key, response)
        resp_dict[key] = response

    if (tnum == 0):
        for qnum in dict.keys(mid_survey):
            ec.screen_prompt(mid_survey[qnum])
    else:
        ec.screen_prompt(instr['cog_mid_abbrev'])

    ec.write_data_line('rel_survey', paradigm)
    for qnum in dict.keys(rel_survey):
        key = 'rel_' + paradigm + '_qnum_' + str(qnum)
        response = getInput(rel_survey_btn, ec, double_check=True, 
                text=rel_survey[qnum])
        ec.write_data_line(key, response)
        resp_dict[key] = response
    identifier = paradigm + '_cog_resp.mat'
    matname = op.join(edf_outputdir, identifier )
    #save with rest of the EDF files
    sio.savemat(matname, resp_dict)

def getInput(response_btns, ec, text, double_check=False,
        cogLoad=True):
    """Handles and directs user input for survey section."""  
    
    while True:
        if cogLoad:
            response = ec.screen_prompt(text, timestamp=False)
        else:
            response = ec.wait_one_press(max_wait=np.inf, min_wait=0,
                    timestamp=False)
        try:
            assert(int(response) in response_btns) # check input
        except ValueError:
            ec.screen_text(instr['ValueError'])
            ec.flip()
            ec.wait_secs(wait_long)
            response = ''
            continue
        except AssertionError:
            ec.screen_text('Please enter a digit value in the range of '
                    + str( min(response_btns)) + ' to ' +
                    str(max(response_btns)))
            ec.flip()
            ec.wait_secs(wait_long)
            response = ''
            continue
        except:  # if any other exception arises start over
            response = ''
            continue
        if (double_check):
            check_response = ec.screen_prompt('You pressed ' + str(
                response) + ', if this is the number you want, press it' +
                ' again key to continue, otherwise press any other key to' +
                ' redo.', timestamp=False)
            if (str(check_response) != str(response)):
                response = ''  # clear past response and loop again
        if response:
            break
    ec.flip()
    return int(response)

def getId_list(paradigm):
    """ Converts the string binary 'paradigm' to an int list for trial
    stamping to TDT and el"""
    return map(int, list(paradigm))

def recordTrial(wheel_matrix_info, preblock, block_ind, bnum, instr, ec,
        stimdir, final_datadir, record_pupil, record_correct,
        save_correct):

    """ Takes the indice of all current condition types and the binary
    name of the condition to find the trial wav.  Displays instructions
    according to instr() and plays stimuli while recording and logging
    pupillometry data.  """

    # identify paradigm and trial number
    #start_time = timeit.default_timer()
    trial_vars = getTrialInfo(block_ind, bnum)
    #elapsed = timeit.default_timer() - start_time
    #print 'load mat ' + str(elapsed) + '\n'
    data_file_name = 'b' + str(bnum) + '_tr' + str(block_ind[bnum]) 
    stim_file_name = 'b' + str(bnum) + '_tr' + str(block_ind[bnum]) + '.wav'
    trial_stim_path = op.join(stimdir, stim_file_name)

    # UNLOAD TRIAL VARS (AVOIDS AWKWARD INDEXING)
    # Creates dict of vars: 'target_letter', 'target_time',
    # 'tot_wav_time', 'paradigm', 'possible_letters' 'tot_cyc'
    target_cycles = trial_vars['target_cycles']
    id_list = trial_vars['trial_id'][0].tolist()
    target_letter = trial_vars['target_letter'][0][0][0].encode('ascii')
    possible_letters = trial_vars['possible_letters'][0]
    final_vars = dict()
    final_vars['trial_id'] = id_list
    # load WAVs for this block
    #start_time = timeit.default_timer()
    stims = []
    stims.append(read_wav(trial_stim_path)[0])  # ignore fs
    stim_dur = stims[0].shape[-1] / ec.stim_fs
    #elapsed = timeit.default_timer() - start_time
    ec.clear_buffer()
    ec.load_buffer(stims[0])
    #elapsed = timeit.default_timer() - start_time
    #print 'load stimuli ' + str(elapsed) + '\n'
    # draw visual primer
    drawPrimer(wheel_matrix_info, target_letter, possible_letters)

    # make screenshots
    #screenshot = ec.screenshot()
    #screen_obj = op.join(edf_outputdir, str(block_ind[bnum]) +
            #'screenshot.obj')
    #screenshot_file = open(screen_obj, 'w')
    #pck.dump(screenshot, screenshot_file)

    # edf stamped for epoch starts
    ec.identify_trial(ec_id=id_list, el_id=id_list, ttl_id=id_list)
    #if record_pupil:
        #ec.identify_trial(ec_id=id_list, el_id=id_list, ttl_id=id_list)
    #else:
        #try:
            #ec.identify_trial(ec_id=id_list, ttl_id=id_list)
        #except:
            #ec.identify_trial(ec_id=id_list, el_id=id_list, ttl_id=id_list)
    ec.start_stimulus(flip=True)  # the visual primer is displayed
    if debug:
        ec.wait_secs(2)
    else:
        ec.wait_secs(vPrimerLen)
    # Draw fixation dot to visual buffer
    fix = visual.FixationDot(ec, colors=['whitesmoke', 'whitesmoke'])
    fix.draw()
    ec.flip()  # the fixation dot is displayed
    if debug:
        ec.wait_secs(1)  
    else:
        # wait until stim has finished
        ec.wait_secs(stim_dur - vPrimerLen)  
        ec.wait_secs(postblock)
    # clear screen
    ec.flip()
    #write all pertinent data as a safety
    if record_pupil:
        ec.write_data_line('target_letter', target_letter)
        ec.write_data_line('target_cycles', target_cycles)

    ec.stop()
    ec.trial_ok()
    # update indexer
    block_ind[bnum] += 1
    if (record_correct):
        response = promptResponse(ec)
        if (response == target_cycles):
            correct = 1
        else:
            correct = 0
        ec.write_data_line('correct: ', correct)
        if save_correct:
            final_dfn = data_file_name + 'final'
            # save with other edf files for exp
            out_path = op.join(edf_outputdir, final_dfn)
            # add in an extra 'correct' data field
            final_vars['correct'] = correct
            sio.savemat(out_path, final_vars)

        return correct
    else:
        return 0

def correctFeedback(ec):
        fix = visual.FixationDot(ec, colors=['Lime', 'Lime'])
        fix.draw()
        ec.flip()  # the fixation dot is displayed
        ec.wait_secs(1.2)
        return

def failFeedback(ec):
        fix = visual.FixationDot(ec, colors=['red', 'red'])
        fix.draw()
        ec.flip()  # the fixation dot is displayed
        ec.wait_secs(1.2)
        return

def promptResponse(ec):
    fix = visual.FixationDot(ec, colors=['yellow', 'yellow'])
    fix.draw()
    ec.flip()  # the fixation dot is displayed
    return getInput(rel_survey_btn, ec, double_check=False, 
            text=instr['bad_input'], cogLoad=False)

def train(order, wheel_matrix_info, preblock, block_ind, instr, ec, 
        stimdir, final_datadir, record_pupil=False,
        record_correct=True, save_correct=False):

    """ Run record trial for each conditions type until subject
    gets two in a row.  If subject doesn't get two in a row after
    specified number of tries, experiment exits"""

    tot_train_blocks = 3
    train_num = 5
    trials_per_cond = len(order[train_num][0])
    # instructions
    # for each condition type
    ec.screen_prompt(instr['start_train'])
    ec.screen_prompt(instr['more_train'])
    for i in range(tot_train_blocks):
        counter = 0;
        status_passed = False;
        oldCorrect = 0
        lastTry = False
        while True:
            if (counter == (trials_per_cond - 1)):
                if lastTry:
                    ec.screen_prompt('You did not pass the'
                            ' training for this condition.'
                            ' Unfortunately you can not continue'
                            ' with the experiment. You may exit the'
                            ' booth now.')
                    sys.exit()
                else:
                    ec.screen_prompt('You did not pass the'
                            ' training for this condition.'
                            ' You can try again one more time'
                            ' by pressing 1.')
                    block_ind[train_num] = 0 # reset ind to reloop
                    counter = 0
                    lastTry = True
            correct = 0
            if (counter == 0):
                condition_no = i
                trial_key = 's' + str(0) + '_' + 'start_trial_' \
                + str(condition_no)
                ec.screen_prompt( instr[(trial_key)])
            correct = recordTrial( wheel_matrix_info, preblock,
                    block_ind, train_num, instr, ec, stimdir,
                    final_datadir, record_pupil, record_correct,
                    save_correct)
            if (correct):
                correctFeedback(ec)
                if (oldCorrect):
                    # must get two in a row to continue
                    ec.screen_prompt(instr['cond_pass'],
                            timestamp=False,
                            max_wait=2);
                    status_passed = True
                    break
            else:
                failFeedback(ec)
            oldCorrect = correct
            counter += 1
        if (status_passed):
            train_num += 1
        else:
            ec.screen_prompt(instr['train_fail']);
    ec.screen_prompt(instr['end_train'])
    return

def getTrialInfo(block_ind, bnum):
    # load in trial data/stims
    data_file_name = 'b' + str(bnum) + '_tr' + str(block_ind[bnum]) + \
            'trial'
    trial_data_path = op.join(final_datadir, data_file_name)
    trial_vars = sio.loadmat(trial_data_path)
    return trial_vars

# run experiment
inputSection = input('Start from section (0,1,2)? ')
if (inputSection == 1):
    inputBlock = input('Start from block (0,1, ...8) ')
else:
    inputBlock = 0
if (inputSection == 2):
    inputCondition = input('Start from condition (0,1,2)? ')
else:
    inputCondition = 0
startInfo = dict()
startInfo['inputSection'] = inputSection
startInfo['inputBlock'] = inputBlock
with ef.ExperimentController(*std_args, **std_kwargs) as ec:
    el = EyelinkController(ec)  # create el instance
    stimdir = op.join(PATH, 'Stims', str(ec.stim_fs))

    assert os.path.isdir(stimdir), 'Can not find Stim directory. \
    Have you runCreateStims.m yet?'
    final_datadir = op.join( datadir, 'Params')
    assert os.path.isdir(final_datadir), 'Can not find Data directory. \
    Have you runCreateStims.m yet?'
    # read in particpant session variables from mat file
    global_vars = sio.loadmat(op.join(final_datadir, 'global_vars.mat'))
    global_vars['vPrimerLen'] = vPrimerLen
    global_vars['postblock'] = postblock
    sio.savemat(op.join(final_datadir, 'global_vars.mat'), global_vars)
    version_code = global_vars['version_code'][0][0]
    print "Using stimuli of version_code: " + str(version_code) + "\n"
    assert version_code == stim_version_code, """Version code specified 
            does not match that of the loaded stimuli.  Did you
            recreate the stimuli?  If so update \'stim_version_code\'
            in \'runKexp.py\'"""
    wheel_matrix_info = global_vars['wheel_matrix_info'][0]
    order = global_vars['order'][0]
    preblock = global_vars['preblock_prime_sec'][0]
    s2_blocks = global_vars['s2_blocks'][0][0]

    # make condition ordering
    # keep the same block ordering for the same subject
    np.random.seed(np.abs(hash(ec._exp_info['participant'])))
    # ordering of the 9 blocks in section 2
    mid_block_order = np.random.permutation(range(1,(1 +
        s2_blocks))).tolist()
    # all other randomness is determined in runcreatestims.m
    section = []
    section.append([0]) # Make section 1
    section.append(mid_block_order) # Make section 2
    section.append([s2_blocks + 1]) # Make section 3
    assert (s2_blocks + 1 == 10)
    folder = ec._exp_info['participant'] + '_' + \
        ec._exp_info['date']
    startInfo['session'] = ec._exp_info['session']
    edf_outputdir = op.join(datadir, folder)
    startInfo['mid_block_order'] = mid_block_order
    ec.write_data_line('mid_block_order', mid_block_order);
    startmat = op.join(edf_outputdir, 'startInfo.mat')
    
    #save start information to data file
    sio.savemat(startmat, startInfo)#save with rest of the EDF files
    ec.set_visible(True)
    ec.set_background_color([0.1] * 3)
    ec.flip()
    ec.start_noise()

    # adjust instruction languages according to runcreatestims
    if global_vars['English']:
        from text_kexp import *
    else:
        from text_kexp_GER import *

    # keep track which new trial to use with array of ind for each block
    block_ind = dict(
        zip(range(len(order)), np.zeros(len(order), dtype=int).tolist()))

    ec.screen_prompt(instr['start_exp'])
    for snum in range(len(section)):
        if (snum < inputSection):
            continue
        ec.write_data_line('Section: ', snum)
        # Initialize section vars
        if (snum == 1):
            # record only in middle section
            record_pupil = True
            record_correct = True
            save_correct = True
        else:
            record_pupil = False
            record_correct = False
            save_correct = False
        section_key = 's' + str(snum) + '_' + 'start_sect'
        ec.screen_prompt(
            instr[(section_key)], 
            max_wait=wait_keys[section_key])

        # run block
        for b_ind in range(len(section[snum])):
            bnum = section[snum][b_ind]
            if (snum == 1):
                if (b_ind < inputBlock):
                    continue
            ec.write_data_line('Block: ', str(bnum))
            # show instructions
            if (snum != 2):
                block_key = 's' + str(snum) + '_' + 'start_block_' + str(b_ind)
                ec.screen_prompt( instr[(block_key)])
            #start a new EDF file only in the middle section 
            if ((snum != 2) and debug):
                el.calibrate(prompt=True)
                assert el.recording 
            for tnum in range(len(order[section[snum][b_ind]][0])):
                if (tnum < inputCondition):
                    block_ind[bnum] += 1
                    continue
                ec.write_data_line('Trial: ', tnum)

                # introduce the conditions in first/last section
                if (snum != 1):
                    trial_vars = getTrialInfo(block_ind, bnum)
                    paradigm = trial_vars['paradigm'][0]
                    condition_no = tnum
                    trial_key = 's' + str(0) + '_' + 'start_trial_' \
                    + str(condition_no)
                    ec.screen_prompt( instr[(trial_key)])

                # start trial
                recordTrial( wheel_matrix_info, preblock, block_ind,
                        bnum, instr, ec, stimdir, final_datadir,
                        record_pupil, record_correct, save_correct )
                # update block_ind
                start_time = timeit.default_timer()
                blockmat = op.join(edf_outputdir, 'block_ind.obj')
                blockfile = open(blockmat, 'w')
                pck.dump(block_ind, blockfile) # overwrites
                elapsed_time = timeit.default_timer() - start_time
                print 'save blockmat for each trial' + str(elapsed_time) + '\n'
                if (snum == 2):
                    cogLoadSurvey(gen_survey, mid_survey, rel_survey,
                            paradigm, edf_outputdir, tnum, ec)
                if (snum != 1):
                    # End trial text feedback
                    trial_end_key = 's' + str(snum) + '_' + 'end_trial'
                    ec.screen_prompt( instr[(trial_end_key)],
                            max_wait=wait_keys[trial_end_key])
            # train for the first section 
            if not skipTrain:
                if (snum == 0):
                    train(order, wheel_matrix_info, preblock, block_ind,
                            instr, ec, stimdir, final_datadir )

            # End block
            block_end_key = 's' + str(snum) + '_' + 'end_block'
            if (snum == 1):
                el.stop() # close edf file for each block in mid section
            if (snum != 2):
                ec.screen_prompt( instr[(block_end_key)],
                        max_wait=wait_keys[block_end_key])

        # End section
        section_end_key = 's' + str(snum) + '_' + 'end_sect'
        ec.screen_prompt( instr[(section_end_key)],
                max_wait=wait_keys[section_end_key])
    ec.stop_noise()
