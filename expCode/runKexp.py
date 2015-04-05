"""
================================
Script ' Karl's Experiment (Kexp)'
================================
    
This script runs an experiment with spatially distributed letter streams.
"""
# Author: Karl Marrett <kdmarret@uw.edu>, <kdmarrett@gmail.com>

#TODO
#how to import functions from and mat files from different dirs
#is there a point to writing to the log file
#locking in a version of ef
#simplify the response buttons
#deep review of all the instr

import scipy
import pyglet
from math import pi
import numpy as np
import random
import math
import os.path as op
import expyfun as ef
from expyfun.io import read_wav
from expyfun import (EyelinkController, visual,
                     get_keyboard_input, assert_version, decimals_to_binary)
from CircularLayout import CircularLayout
import os
# from processPupil import *
import processPupil
processPupil._check_pyeparse()

#assert ef.__version__ == '2.0.0.DASCogLoad'

# do this in two lines
localPath = os.path.abspath(os.curdir)
os.chdir("..")
PATH = os.path.abspath(os.curdir)
os.chdir(localPath)
datadir = op.join(PATH, 'Data')

# EC PARAMETERS
total_btns = 10
gen_survey_btn = range(total_btns)
rel_survey_btn = [1, 2]
pretrial_wait = 2.5
std_args = ['kexp']
std_kwargs = dict(screen_num=0, window_size=[800, 600], check_rms=None,
		full_screen=True, stim_db=65, noise_db=40, stim_rms=0.01,
		#session='1',participant='foo',  
		suppress_resamp=False, response_device='keyboard',
		output_dir=datadir, stim_fs=16000)  # 44100.0

# GLOBAL VARIABLES
wait_brief = .2
wait_long = 2
msg_dur = 3.0
postblock = 1  # time after each trial to record pupil
# these should be hardcoded and read in from the matlab stimuli script
extra_wait = 10

# MAKE CONDITION ORDERING
section = []
section.append(0) # Make section 1
section.append([1, 2, 3]) # Make section 2
section.append(4) # Make section 3


def drawPrimer(wheel_matrix_info, target_letter, possible_letters):
    """ Creates the primer letters and draws to visual buffer """

    # get center locations of each wheel
    wheel_loc = CircularLayout(3, radius=.45, angles=[- pi, (pi / 2), 0])
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


def cogLoadSurvey(gen_survey, mid_survey, rel_survey, id_, ec):
    """ Prompt users with cognitive load questions for each condition"""

    ec.write_data_line('cogLoadSurvey')
    ec.write_data_line(id_)
    # import ipdb; ipdb.set_trace()
    ec.write_data_line('gen_survey')
    for qnum in dict.keys(gen_survey):
        ec.write_data_line(id_ + 'Q_no_' + str(qnum))
        response = surveyInput(gen_survey[qnum], gen_survey_btn, ec)
        ec.write_data_line(response)

    for qnum in dict.keys(mid_survey):
        ec.screen_prompt(mid_survey[qnum])

    ec.write_data_line('rel_survey')
    for qnum in dict.keys(rel_survey):
        ec.write_data_line(id_ + 'Q_no_' + str(qnum))
        response = surveyInput(rel_survey[qnum], rel_survey_btn, ec)
        ec.write_data_line(response)


def surveyInput(text, response_btns, ec):
    """Handles and directs user input for survey section.  +++ Need to
    separate TDT continue button input from the keyboard input for
    subjects"""

    response = ''
    while not response:
        response = ec.screen_prompt(text, timestamp=False)
        # check input
        try:
            assert(int(response) in response_btns)
        except ValueError:
            ec.screen_text(gen_survey['ValueError'])
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
        check_response = ec.screen_prompt('You pressed ' + str(
            response) + ', if this is the number you want press "{}" to' +
            ' continue otherwise press any other key to' +
            ' redo'.format(cont_btn_label), timestamp=False)
        if check_response != str(cont_btn):
            response = ''  # clear past response and loop again
    return response

def recordTrial(wheel_matrix_info, preblock, block_ind, bnum, instr, ec,
        el, stimdir, final_datadir, record_pupil, record_correct):

    """ Takes the indice of all current condition types and the binary
    name of the condition to find the trial wav.  Displays instructions
    according to instr() and plays stimuli while recording and logging
    pupillometry data.  """

    # identify paradigm and trial number
    data_file_name = 'b' + str(bnum) + '_tr' + block_ind[bnum]
    stim_file_name = data_file_name + '.wav'
    trial_data_path = op.join(final_datadir, data_file_name)
    trial_stim_path = op.join(stimdir, stim_file_name)

    # UNLOAD TRIAL VARS (AVOIDS AWKWARD INDEXING)
    # Creates dict of vars: 'target_letter', 'target_time',
    # 'tot_wav_time', 'paradigm', 'possible_letters' 'tot_cyc'
    trial_vars = scipy.io.loadmat(trial_data_path)
    target_time = trial_vars['target_time']
    target_cycles = trial_vars['target_cycles']
    target_letter = trial_vars['target_letter'][0][0][0].encode('ascii')
    possible_letters = trial_vars['possible_letters'][0]
    # check loading of correct mat file
    # load WAVs for this block
    id_list = map(int, list(id_))
    stims = []
    stims.append(read_wav(trial_stim_path)[0])  # ignore fs
    stim_dur = stims[0].shape[-1] / ec.stim_fs
    ec.clear_buffer()
    ec.load_buffer(stims[0])
    # draw visual primer
    drawPrimer(wheel_matrix_info, target_letter, possible_letters)
	# edf stamped for epoch starts
    ec.identify_trial(ec_id=id_list, el_id=id_list, ttl_id=id_list)
    ec.start_stimulus(flip=True)  # the visual primer is displayed
    ec.wait_secs(preblock / 3)
    # Draw fixation dot to visual buffer
    fix = visual.FixationDot(ec, colors=['whitesmoke', 'whitesmoke'])
    fix.draw()
    ec.flip()  # the fixation dot is displayed
    ec.wait_secs(stim_dur - preblock / 3)  # wait until stim has finished
    ec.wait_secs(postblock)
    # clear screen
    ec.flip()
    # write out data
    if record_pupil:
		#write all pertinent data as a safety
        ec.write_data_line('target_time', target_time)
        ec.write_data_line('target_letter', target_letter)
        ec.write_data_line('target_cycles', target_cycles)

	# do these fail silently if there is no current edf?
    ec.stop()
    ec.trial_ok()
    # update indexer
    block_ind[bnum] += 1
	response = promptResponse(ec)
    if (response == target_cycles):
        correct = 1
    else:
        correct = 0
    if (record_correct):
        final_df = data_file_name + 'final'
        trial_vars['correct'] = correct
        scipy.savemat(final_df, trial_vars)
    return correct

def correctFeedback(ec):
	fix = visual.FixationDot(ec, colors=['Lime', 'Lime'])
	fix.draw()
	ec.flip()  # the fixation dot is displayed
	ec.wait_sec(2)
	return

def failFeedback(ec):
	fix = visual.FixationDot(ec, colors=['magenta', 'magenta'])
	fix.draw()
	ec.flip()  # the fixation dot is displayed
	ec.wait_sec(2)
	return

def promptResponse(ec):
	fix = visual.FixationDot(ec, colors=['yellow', 'yellow'])
	fix.draw()
	ec.flip()  # the fixation dot is displayed
    raw_resp = ec.screen_prompt('', timestamp=False, flip=False)
    return int(raw_resp)
	

def train(order, wheel_matrix_info, preblock, block_ind, instr, ec, el,
		stimdir, final_datadir, record_pupil=False, record_correct=False
		):

    """ Run record trial for each conditions type until subject
    gets two in a row.  If subject doesn't get two in a row after
    specified number of tries, experiment exits"""

    tries = 2
    tot_train_blocks = 3
    train_num = 5
    # instructions
    # for each condition type
	ec.screen_prompt(instr['start_train'], live_keys=button_keys['start_exp'])
	ec.screen_prompt(instr['more_train'], live_keys=button_keys['start_exp'])
    for i in range(tot_train_blocks):
		counter = 0;
		status_passed = False;
		oldCorrect = 0
        while (counter < (tries* (len(order[train_num][0])))):
            counter += 1
			snum = 0
			correct = 0
			condition_no = getTrialCondition(block_ind, train_num)
			# change this to the instructions given in the introduction
			ec.screen_prompt(instr['start_train'], live_keys=button_keys['start_exp'])
            correct = recordTrial( wheel_matrix_info, preblock,
                    block_ind, train_num, instr, ec, el, stimdir,
                    final_datadir, record_pupil, record_correct )
            if (correct):
				correctFeedback(ec)
				if (oldCorrect):
					# must get two in a row to continue
					ec.screen_prompt(instr['cond_pass'], timestamp=False,
							live_keys=button_keys['start_exp'], max_wait=2);
					status_passed = True
					break
			else:
				failFeedback(ec)
			oldCorrect = correct

        if (!status_passed):
            ec.screen_text(instr['train_fail'], timestamp=False);
        else:
            train_num += 1

	ec.screen_prompt(instr['end_train'], live_keys=button_keys['start_exp'])
	return

def getTrialCondition(block_ind, bnum):
    # LOAD IN TRIAL DATA/STIMS
    wav_num = block_ind[bnum]
    data_file_name = 'b' + str(bnum) + '_tr' + wav_num
    stim_file_name = data_file_name + '.wav'
    trial_data_path = op.join(final_datadir, data_file_name)
    trial_vars = scipy.io.loadmat(trial_data_path)
    condition_no = trial_vars['paradigm'][0]
    #condition_no = section[snum][bnum][tnum]
    #id_ = condition_asc[condition_no]
    # id_ = decimals_to_binary(id_, np.ones(1, len(id_)))
    return condition_no

# RUN EXPERIMENT
with ef.ExperimentController(*std_args, **std_kwargs) as ec:
    stimdir = op.join(PATH, 'Stims', ec._exp_info[
                      'participant'], ec._exp_info['session'])
    el = EyelinkController(ec)  # create el instance
    ec.set_visible(True)
    ec.set_background_color([0.1] * 3)
    ec.flip()
    ec.start_noise()

    assert(os.path.isdir(stimdir), 'Can not find Stim directory. \
    Have you runCreateStims.m yet?')
    assert(os.path.isdir(final_datadir), 'Can not find Data directory. \
    Have you runCreateStims.m yet?')

    # read in particpant session variables from mat file
    final_datadir = op.join( datadir, 'Params')
    global_vars = scipy.io.loadmat(op.join(final_datadir, 'global_vars.mat'))
    condition_uni = global_vars['condition_bin']  # Unicode by default
    condition_asc = []  # ASCII
    for i in range(len(condition_uni)):
        condition_asc.append(condition_uni[i].encode('ascii'))
    condition_nums = len(condition_asc)
    wheel_matrix_info = global_vars['wheel_matrix_info'][0]
    order = global_vars['order'][0]
    preblock = global_vars['preblock_prime_sec'][0]

    # adjust instruction languages according to runcreatestims
    if global_vars['English']:
        from text_kexp import *
    else:
        from text_kexp_GER import *

    # keep track which new trial to use with array of ind for each block
    block_ind = dict(
        zip(range(len(order)), np.zeros(len(order), dtype=int)))

    ec.screen_prompt(instr['start_exp'], live_keys=button_keys['start_exp'])
    for snum in range(len(section)):
        ec.write_data_line('Section: ', snum)
        # Initialize section vars
        if (snum == 1):
            # record only in middle section
            record_pupil = True
            record_correct = True
        else:
            record_pupil = False
            record_correct = False
        section_key = 's' + str(snum) + '_' + 'start_sect'
        ec.screen_prompt(
            instr[(section_key)], live_keys=button_keys[(section_key)],
            max_wait=wait_keys[section_key])

        # run block
        for bnum in range(len(section[snum])):
            ec.write_data_line('Block: ', str(bnum))
            # show instructions
            block_key = 's' + str(snum) + '_' + 'start_block_' + str(bnum)
            ec.screen_prompt( instr[(block_key)],
                    live_keys=button_keys[(block_key)],
                    max_wait=wait_keys[block_key])
            if (snum == 1):
				#start a new EDF file only in the middle section 
                el.calibrate(prompt=True)
            for tnum in range(len(order[section[snum][bnum]][0])):
                ec.write_data_line('Trial: ', tnum)

                if (snum == 0):
                    condition_no = getTrialCondition(block_ind, bnum)
                #remove instructions from section 3
                if snum == 2:
                    # use instructions from section 1
                    trial_key = 's' + \
                        str(0) + '_' + 'start_trial_' + str(condition_no)
                else:
                    trial_key = 's' + \
                        str(snum) + '_' + 'start_trial_' + str(condition_no)

                # instructions if the first section
                if (snum == 0):
                    ec.screen_prompt( instr[(trial_key)],
                            live_keys=button_keys[(trial_key)],
                            max_wait=wait_keys[trial_key])

                # start trial
                recordTrial( wheel_matrix_info, preblock, block_ind,
                        bnum, instr, ec, el, stimdir, final_datadir,
                        record_pupil, record_correct )
                if snum == 2:
                    cogLoadSurvey(gen_survey, mid_survey, rel_survey, id_, ec)
                # End trial
                trial_end_key = 's' + str(snum) + '_' + 'end_trial'
                ec.screen_prompt(
                    instr[(trial_end_key)], live_keys=button_keys[
                        (trial_end_key)], max_wait=wait_keys[trial_end_key])

            # train for the first section 
            if (snum == 0):
				train(order, wheel_matrix_info, preblock, block_ind,
						instr, ec, el, stimdir, final_datadir )

            # End block
            block_end_key = 's' + str(snum) + '_' + 'end_block'
            if (snum == 1):
                el.stop() # close edf file for each block in 2 section
            ec.screen_prompt( instr[(block_end_key)],
                    live_keys=button_keys[ (block_end_key)],
                    max_wait=wait_keys[block_end_key])

        # End section
        section_end_key = 's' + str(snum) + '_' + 'end_sect'
        ec.screen_prompt( instr[(section_end_key)],
                live_keys=button_keys[(section_end_key)],
                max_wait=wait_keys[section_end_key])
    ec.stop_noise()
