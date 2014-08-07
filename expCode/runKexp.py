"""
================================
Script ' Karl's Experiment (Kexp)'
================================

This script runs an experiment with spatially distributed letter streams.
"""
# Author: Karl Marrett <kdmarret@uw.edu>, <kdmarrett@gmail.com>

# TO DO
# add in pupil script from voc_meg and pupil...py
# reading/writing from Stims, Data
# learn identifity trial and trial ok
# clean all instructions
# draw tri o speller
# getting input responses for NASA

import pdb # +++
import scipy
import pyglet
from math import pi
# from glob import glob
import numpy as np
import math
import os.path as op
import expyfun as ef
from expyfun import decimals_to_binary
from expyfun.io import read_wav
from expyfun import (EyelinkController, visual, get_keyboard_input, assert_version, decimals_to_binary)
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
exp_datadir = 'exp_Data'



# EXPERIMENT PARAMETERS
cont_btn = 8
cont_btn_label = 'Next'
pretrial_wait = 2.5
feedback_dur = 1.5
std_args = ['kexp']
std_kwargs = dict(screen_num=0, window_size=[800, 600], full_screen=False,
				  stim_db=65, noise_db=40,   session='1', participant='foo',
				  stim_rms=0.01, check_rms=None, suppress_resamp=False,
				  output_dir=datadir, stim_fs=16000)  # 44100.0


# RANDOM NUMBER GENERATOR
rng = np.random.RandomState(0)

# DEFINE TRIAL RECORDING FUNCTION
def	recordTrial(wheel_matrix_info, preblock, id_, stim_trial_indices, instr, ec, stimdir, final_datadir, record_pupil = True):
	""" Takes the indice of all current condition types and the binary name
	of the condition to find the trial wav.  Displays instructions according
	to intr_key and plays stimuli while recording and logging 
	pupillometry data.  Pylink functionality to be added last"""

	# UNLOAD TRIAL VARS (INDEXING IS AWKWARD)
	# identify paradigm and trial number
	trial_wav_no = stim_trial_indices['id_']
	file_name = str(id_) + '_tr' + str(trial_wav_no)
	trial_data_path = op.join(final_datadir, file_name)
	# Creates dict of vars: 'target_letter', 'target_time', 'tot_wav_time', 'paradigm', 'possible_letters' 
	trial_vars = scipy.io.loadmat(trial_data_path)
	trial_stim_path = op.join(stimdir, file_name)	

	target_letter = trial_vars['target_letter']
	target_time = trial_vars['target_time']
	paradigm = trial_vars['paradigm']
	target_letter = trial_vars['target_letter']
	possible_letters = trial_vars['possible_letters']


	# DRAW SELECTIONS ON SCREEN
	# get the center locations of each wheel
	wheel_loc = CircularLayout(3, radius = .6, angles = [- pi, (pi / 2), 0]) # must be three wheels
	# get the individual letter locations
	letter_loc = [];
	for i in range(len(wheel_matrix_info)):
		letters_wheel = wheel_matrix_info[i]
		temp = CircularLayout(letters_wheel, radius = .15, relative_center = wheel_loc.positions[i])
		letter_loc.append(temp.positions)

	pdb.set_break()
	

	for i in range(len(trial_vars['possible_letters'])):
		letter = trial_vars['possible_letters']	[i][0][0]
		print letter[0]
		print trial_vars['target_letter'][0]	
		if (letter == target_letter):
			color = 'Lime'
		else :
			color = 'LightGray'
		# ec.screen_text(letter, letter_loc[i], )
		ec.screen_text(letter, letter_loc[0][i], color=color, font_size=18)
	ec.wait_until(preblock)	
	# Refer to voc_meg.py and pupillometry.py and PylinkAPI.pdf for referenc on recording pupillometry
	ec.identify_trial(ec_id=id_, el_id=id_, ttl_id = id_)    

	# load WAVs for this block
	# Creates dict of vars: 'target_letter', 'target_time', 'tot_wav_time', 'paradigm', 'possible_letters' 
	stims = []
	stims.append(read_wav(trial_stim_path)[0])  # ignore fs
	ec.flip()
	# get ready
	stim_dur = stim.shape[-1] / ec.stim_fs
	ec.load_buffer(stim)

	# play stim
	ec.start_stimulus(flip=False)
	ec.wait_until(stim_dur)
	ec.stop()
	ec.trial_ok()
	stim_trial_indices['id_'] += 1

	# clear screen
	ec.flip()

	# write out data
	# ec.write_data_line('target_time', target_time)

	return stim_trial_indices

def cogLoadSurvey(paradigm, gen_survey, mid_survey, rel_survey, id_, ec):
	ec.write_data_line('cogLoadSurvey')
	ec.write_data_line(id_)
	for qnum in len(gen_survey):
		ec.write_data_line('gen_survey')
		ec.write_data_line(str(paradigm) + 'Q_no_' + str(qnum))
		response = get_keyboard_input(gen_survey([qnum]), out_type = int)
		ec.write_data_line(response)
	for qnum in len(mid_survey):
		screen_prompt(mid_survey([qnum]), live_keys = [cont_btn]) # +++ live_keys?
	for qnum in len(rel_survey):
		ec.write_data_line('rel_survey')
		ec.write_data_line(str(paradigm) + 'Q_no_' + str(qnum))
		response = get_keyboard_input(rel_survey([qnum]), out_type = int)
		ec.write_data_line(response)


# MAKE BLOCK DESIGN
block = [];
section = [];

#  Make sections 1 and 3 structure
reps = (1, 1)
for snum in range(2):
	for bnum in range(8):
		block.extend(np.tile((bnum), (reps[snum], 1)))
	section.append(block)

section.append(section[1])
section[1] = [];

# Make section 2 structure
# This needs to be pseudo randomly generated
block.extend([1, 2, 3, 2, 1, 3, 1, 3, 2])
block.extend([3, 2, 1, 5, 4, 1, 4, 5, 1])
block.extend([1, 4, 5, 1, 5, 4, 6, 5, 1])
block.extend([5, 6, 1, 5, 6, 5, 1, 5, 1]) 
block.extend([1, 8, 7, 8, 1, 7, 1, 7, 8])
section[1] = [block[0], block[1], block[2], block[3], block[4]];

# RUN EXPERIMENT
with ef.ExperimentController(*std_args, **std_kwargs) as ec:
	# el = EyelinkController(ec) # create el instance

	# Assert the subject id and session number exist
	# assert(os.path.isdir(op.join(stimDir, participant, session)), 'Can not find Stim directory.  Have you runCreateStims.m yet?')

	# READ IN GLOBAL VARIABLES FROM FILE
	# Reads in 'condition_bin', 'wheel_matrix_info', preblock_prime_sec 
	final_datadir = op.join(datadir, participant, session)
	global_vars = scipy.io.loadmat(op.join(final_datadir, 'global_vars.mat'))

	# VARIOUS VARIABLES
	block_in_sections = [1, 5, 8]
	trial_in_blocks = [8, 9, 1]
	condition_bin = global_vars['condition_bin']
	wheel_matrix_info = global_vars['wheel_matrix_info'][0]
	stim_trial_indices = dict(zip(condition_bin, np.zeros(len(condition_bin))))
	# section_instr_counter = np.zeros(len(global_vars['wheel_matrix_info']))
	preblock = global_vars['preblock_prime_sec'][0]
	wait_brief = .2
	wait_long = 2
	msg_dur = 3.0

	for snum in range(len(section)):

		ec.start_noise()

		# Initialize section vars
		section_base = 's' + str(snum) + '_'
		ec.screen_prompt(instr[(section_base + 'start_sect')], live_keys=[cont_btn])

		# run blocks
		for bnum in range(block_in_sections[snum]):
			# show instructions
			ec.screen_prompt(instr[(section_base + 'start_block_' + str(bnum))], live_keys=[cont_btn])

			# log block name
			ec.write_data_line('block', str(bnum))

			for tnum in range(section[snum][bnum]): # +++ check this 
				pdb.set_trace()
				# LOAD IN TRIAL DATA/STIMS
				paradigm_no = section[snum][bnum][tnum]
				id_ = global_vars['condition_bin'][paradigm_no]
				# id_ = decimals_to_binary(id_, np.ones(1, len(id_)))

				# start trial
				ec.screen_prompt(instr[(section_base + 'start_trial_' + str(tnum))], live_keys=[cont_btn])
				# ec.identify_trial(ec_id=ecid, ttl_id=ttlid) # should include eyelink controller as well
				if (snum == 2):
					record_pupil = False
				else :
					record_pupil = True

				stim_trial_indices = recordTrial(wheel_matrix_info, preblock, id_, stim_trial_indices, instr, ec, stimdir, final_datadir, record_pupil)	

				if (snum == 2):
					cogLoadSurvey(paradigm, gen_survey, mid_survey, rel_survey, id_, ec)

				# End trial
				ec.screen_prompt(instr[(section_base + 'end_trial')], live_keys=[], max_wait = wait_long)

			# End block
			ec.screen_prompt(instr[(section_base + 'end_block')], live_keys=[cont_btn])

		# End section
		ec.screen_prompt(instr[(section_base + 'end_sect')], max_wait=0.0, live_keys=[cont_btn])

# DASCogLoad dr refere
# cur_cols = np.where(attn[block][tnum], 'Lime', 'LightGray').tolist()
# txt_obj = []
# for n, cat in enumerate(cur_cats):
# 	cat = '<center>' + cat + '</center>'  # hack (pyglet bug)
# 	txt_obj.append(ec.screen_text(cat, pos=[xpos[n], ypos[n]],
# 								  color=cur_cols[n], font_size=36))
# end_wait = ec.current_time + pretrial_wait
# ec.flip()