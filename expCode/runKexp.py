"""
================================
Script ' Karl's Experiment (Kexp)'
================================

This script runs an experiment with spatially distributed letter streams.
"""
# Author: Karl Marrett <kdmarret@uw.edu>, <kdmarrett@gmail.com>

# TO DO
# add in pupil script
# el instance can not be made in mint
# learn identifity trial and trial ok
# getting input responses for NASA all logged
# what is np.random.RandomState(0)

import scipy
import pyglet
from math import pi
import numpy as np
import random
import math
import os.path as op
import expyfun as ef
from expyfun import decimals_to_binary
from expyfun.io import read_wav
from expyfun import (EyelinkController, visual,
                     get_keyboard_input, assert_version, decimals_to_binary)
from CircularLayout import CircularLayout
from text_kexp import *

#assert ef.__version__ == '2.0.0.DASCogLoad'

# Temporary Hack
participant = 'foo'
session = '1'

PATH = '/home/kdmarrett/git/kexp'
stimdir = op.join(PATH, 'Stims', participant, session)
# datadir = op.join(PATH, 'Data', participant, session)
datadir = op.join(PATH, 'Data')


# EC PARAMETERS
cont_btn = 8
cont_btn_label = 'Next'
pretrial_wait = 2.5
std_args = ['kexp']
std_kwargs = dict(screen_num=0, window_size=[800, 600], full_screen=True,
                  stim_db=65, noise_db=40,   session='1', participant='foo',
                  stim_rms=0.01, check_rms=None, suppress_resamp=False,
                  output_dir=datadir, stim_fs=16000)  # 44100.0

# RANDOM NUMBER GENERATOR
rng = np.random.RandomState(0)
# GLOBAL VARIABLES
block_in_sections = [1, 5, 8]
trial_in_block = [8, 9, 1]
controls_in_block = 2 # only applies to section 2
wait_brief = .2
wait_long = 2
msg_dur = 3.0
postblock = 5  # time after each trial to record pupil

def ExperimentOrdering(block_in_sections, trial_in_block, condition_nums, controls_in_block):
    """Creates the random ordering of the conditions for each trial
    by section, block, and trial.  Refer to block_in_sections and 
    trial_in_block for final shape of section.

    Returns:
    section -- which gives the unique condition number for each trial
    indexed by section[section_num][block_num][trial_num]
    """

    section = []
    # Make section 1
    section.append([random.sample(range(0, condition_nums), condition_nums)])

    # Make section 2
    control_wraps = False
    while not control_wraps:
        block = []
        for i in range(block_in_sections[1]):
            repeats = True
            bicontrol = False
            while repeats:
            # while (repeats or not bicontrol):
                # generate trial
                # trial = np.random.randint(
                #     0, condition_nums, condition_nums).tolist()
                # require at least two controls
                # bicontrol = len(np.where(trial == 0)) >= 2

                # alternate way of generating trial
                trial = random.sample(range(0, condition_nums), 
                            condition_nums) #creates a shuffled range(8)
                # add an extra control to every trial for more comparisons
                control_ind = trial.index(0)
                # import ipdb; ipdb.set_trace()
                # control_ind = np.where(np.array(trial) == 0)[0]
                second_range = range(condition_nums)
                del second_range[control_ind]
                second_ind = random.sample(second_range, 1)[0]
                trial[second_ind] = 0

                # check no condition has consecutive trials 
                repeats = checkRepeats(trial)
            block.append(trial)
        control_wraps = controlProceedsFollows(block)
    section.append(block)

    # Make section 3
    block = []
    condition_ordering = random.sample(
        range(0, condition_nums), condition_nums)
    for i in range(condition_nums):
        block.append(condition_ordering[i])
    section.append(block)
    return section

def checkRepeats(trial):
    """ Checks for any repeats in a given list """
    repeats = False
    for i in range(len(trial) - 1):
        if (trial[i] == trial[i + 1]):
            repeats = True
    return repeats

def controlProceedsFollows(block):
    """ For section 2 checks whether the block as
    a whole has at least one occurrence of the control
    condition before and after every other condition"""

    # Create arrays with elements counting which condition
    # has proceeded/followed a control
    # control element compared to all other conditions
    proceeds_condition = np.zeros((condition_nums - 1), int).tolist()
    follows_condition = np.zeros((condition_nums - 1), int).tolist()
    for i in range(len(block)):
        trial = np.array(block[i])
        control_ind = np.where(trial == 0)[0]
        for j in range(len(control_ind)):
            ind = control_ind[j]
            if ind != 0:  # avoids index out of bounds error
                proceeds_condition[trial[ind - 1] - 1] += 1
            if ind != (condition_nums - 1):
                follows_condition[trial[ind + 1] - 1] += 1
    control_wraps = False
    # check that all conditions proceeded and followed by control
    if (len(np.nonzero(proceeds_condition)[0]) == (condition_nums - 1)):
        if (len(np.nonzero(follows_condition)[0]) == (condition_nums - 1)):
            control_wraps = True
    return control_wraps

def recordTrial(wheel_matrix_info, preblock, id_, wav_indices, instr, ec, el, stimdir, final_datadir, record_pupil=True):
    """ Takes the indice of all current condition types and the binary name
    of the condition to find the trial wav.  Displays instructions according
    to instr() and plays stimuli while recording and logging
    pupillometry data.  """

    # identify paradigm and trial number
    wav_num = wav_indices[id_]
    data_file_name = id_ + '_tr' + str(wav_num)
    stim_file_name = id_ + '_tr' + str(wav_num) + '.wav'
    trial_data_path = op.join(final_datadir, data_file_name)
    trial_stim_path = op.join(stimdir, stim_file_name)

    # UNLOAD TRIAL VARS (AVOIDS AWKWARD INDEXING)
    # Creates dict of vars: 'target_letter', 'target_time', 'tot_wav_time',
    # 'paradigm', 'possible_letters'
    trial_vars = scipy.io.loadmat(trial_data_path)
    target_time = trial_vars['target_time']
    target_letter = trial_vars['target_letter'][0][0][0].encode('ascii')
    possible_letters = trial_vars['possible_letters'][0]
    # check loading of correct mat file
    assert (trial_vars['paradigm'][0].encode('ascii')
            == id_), "Error: id_ and paradigm from mat file do not match"

    # get center locations of each wheel
    wheel_loc = CircularLayout(3, radius=.6, angles=[- pi, (pi / 2), 0])
    # get individual letter locations
    letter_loc = []
    for i in range(len(wheel_matrix_info)):
        letters_wheel = wheel_matrix_info[i]
        temp = CircularLayout(
            letters_wheel, radius=.30, relative_center=wheel_loc.positions[i])
        # letter_loc.append(temp.positions)
        letter_loc.extend(temp.positions)
    # draw selections to visual buffer
    for i in range(len(possible_letters)):
        letter = possible_letters[i][0].encode('ascii')
        if (letter == target_letter):
            color = 'Lime'
        else:
            color = 'LightGray'
        ec.screen_text(
            letter, pos=letter_loc[i], color=color, font_size=55, wrap=False)
    id_list = map(int, list(id_))
    ec.identify_trial(ec_id=id_list, el_id=id_list, ttl_id=id_list)

    # load WAVs for this block
    stims = []
    stims.append(read_wav(trial_stim_path)[0])  # ignore fs
    ec.flip()
    stim_dur = stims[0].shape[-1] / ec.stim_fs
    ec.load_buffer(stims[0])
    # play stim
    ec.start_stimulus(flip=True)  # the visual primer is displayed
    ec.wait_secs(preblock)
    # Draw fixation dot to visual buffer
    ec.flip()  # the fixation dot is displayed
    ec.wait_secs(stim_dur - preblock)  # wait until stim has finished
    ec.stop()
    ec.clear_buffer()
    ec.wait_secs(postblock)
    # clear screen
    ec.flip()
    # write out data
    ec.write_data_line('target_time', target_time)
    ec.write_data_line('target_letter', target_letter)
    ec.write_data_line('target_time', target_time)

    ec.trial_ok()
    wav_indices[id_] += 1
    return wav_indices


def cogLoadSurvey(gen_survey, mid_survey, rel_survey, id_, ec):
    ec.write_data_line('cogLoadSurvey')
    ec.write_data_line(id_)
    for qnum in len(gen_survey):
        ec.write_data_line('gen_survey')
        ec.write_data_line(id_ + 'Q_no_' + str(qnum))
        response = get_keyboard_input(gen_survey([qnum]), out_type=int)
        ec.write_data_line(response)
    for qnum in len(mid_survey):
        # +++ live_keys?
        screen_prompt(mid_survey([qnum]), live_keys=[cont_btn])
    for qnum in len(rel_survey):
        ec.write_data_line('rel_survey')
        ec.write_data_line(id_ + 'Q_no_' + str(qnum))
        response = get_keyboard_input(rel_survey([qnum]), out_type=int)
        ec.write_data_line(response)


# RUN EXPERIMENT
with ef.ExperimentController(*std_args, **std_kwargs) as ec:
    # el = EyelinkController(ec)  # create el instance
    el = ''

    # Assert the subject id and session number exist
    # assert(os.path.isdir(stimdir), 'Can not find Stim directory.  Have you runCreateStims.m yet?')
    # assert(os.path.isdir(op.join(stimdir, participant, session)), 'Can not find Stim directory.  Have you runCreateStims.m yet?')

    # READ IN PARTICPANT SESSION VARIABLES FROM MAT FILE
    # Reads in 'condition_bin', 'wheel_matrix_info', 'preblock_prime_sec'
    final_datadir = op.join(datadir, participant, session)
    global_vars = scipy.io.loadmat(op.join(final_datadir, 'global_vars.mat'))
    condition_uni = global_vars['condition_bin']
    condition_asc = []
    for i in range(len(condition_uni)):
        condition_asc.append(condition_uni[i].encode('ascii'))
    condition_nums = len(condition_asc)
    wheel_matrix_info = global_vars['wheel_matrix_info'][0]
    wav_indices = dict(
        zip(condition_asc, np.zeros(len(condition_asc), dtype=int)))
    preblock = global_vars['preblock_prime_sec'][0]

    # MAKE CONDITION ORDERING
    section = ExperimentOrdering(
        block_in_sections, trial_in_block, condition_nums, controls_in_block)

    for snum in range(len(block_in_sections)):
        ec.write_data_line('Section: ', snum)
        ec.start_noise()
        # Initialize section vars
        section_base = 's' + str(snum) + '_'
        ec.screen_prompt(
            instr[(section_base + 'start_sect')], live_keys=[cont_btn])

        # run block
        for bnum in range(block_in_sections[snum]):
            ec.write_data_line('Block: ', bnum)
            # show instructions
            ec.screen_prompt(
                instr[(section_base + 'start_block_' + str(bnum))], live_keys=[cont_btn])

            # log block name
            ec.write_data_line('block', str(bnum))

            for tnum in range(trial_in_block[snum]):
                ec.write_data_line('Trial: ', tnum)

                # LOAD IN TRIAL DATA/STIMS
                condition_no = section[snum][bnum][tnum]
                id_ = condition_asc[condition_no]
                # id_ = decimals_to_binary(id_, np.ones(1, len(id_)))
                # start trial
                ec.screen_prompt(
                    instr[(section_base + 'start_trial_' + str(tnum))], live_keys=[cont_btn])
                if snum == 2:
                    record_pupil = False
                else:
                    record_pupil = True
                wav_indices = recordTrial(
                    wheel_matrix_info, preblock, id_, wav_indices, instr, ec, el, stimdir, final_datadir, record_pupil)
                if snum == 2:
                    cogLoadSurvey(gen_survey, mid_survey, rel_survey, id_, ec)
                # End trial
                ec.screen_prompt(
                    instr[(section_base + 'end_trial')], live_keys=[], max_wait=wait_long)
            # End block
            ec.screen_prompt(
                instr[(section_base + 'end_block')], live_keys=[cont_btn])
        # End section
        ec.screen_prompt(
            instr[(section_base + 'end_sect')], max_wait=0.0, live_keys=[cont_btn])
ec.stop_noise()
