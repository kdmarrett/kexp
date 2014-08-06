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
# from expyfun import (ExperimentController, EyelinkController, visual,
					 # get_keyboard_input, assert_version, decimals_to_binary)
from CircularLayout import CircularLayout
from text_kexp import *

#assert ef.__version__ == '2.0.0.DASCogLoad'

# Temporary Hack
participant = 'foo'
session = '1'

PATH = '/home/kdmarrett/git/kexp'
# /home/kdmarrett/git/kexp/Data/foo/1

#stimDir 
# varsfile = 'expVars.mat'
stimdir = op.join(PATH, 'Stims', participant, session)
# datadir = op.join(PATH, 'Data', participant, session)
datadir = op.join(PATH, 'Data')
exp_datadir = 'exp_Data'


#READ IN WAVFILES
wait_dur = 3.0
msg_dur = 3.0

# EXPERIMENT PARAMETERS
cont_btn = 8
cont_btn_label = 'Next'
pretrial_wait = 2.5
feedback_dur = 1.5
std_args = ['kexp']
std_kwargs = dict(screen_num=0, window_size=[800, 600], full_screen=True,
				  stim_db=65, noise_db=40,   session='1', participant='foo',
				  stim_rms=0.01, check_rms=None, suppress_resamp=False,
				  output_dir=datadir, stim_fs=16000)  # 44100.0


# RANDOM NUMBER GENERATOR
rng = np.random.RandomState(0)

# DEFINE TRIAL RECORDING FUNCTION
def	recordTrial(trial_vars, global_vars, trial_stims, id_, stim_trial_indices, instr, ec, stimdir, datadir, record_pupil = True):
	""" Takes the indice of all current condition types and the binary name
	of the condition to find the trial wav.  Displays instructions according
	to intr_key and plays stimuli while recording and logging 
	pupillometry data.  Pylink functionality to be added last"""

	# DRAW SELECTIONS ON SCREEN
	wheel_loc = CircularLayout(3, radius = .6, angles = [- pi, (pi / 2), 0]) # must be three wheels
	letter_loc = [];
	for i in range(len(global_vars['wheel_matrix_info'])):
		temp = CircularLayout(wheel_matrix_info[i], radius = .15, relative_center = wheel_loc.positions[i])
		letter_loc.append(temp.positions)

	for i in range(len(trial_vars['possible_letters'])):
		letter = trial_vars['possible_letters']	[i]
		if (letter == trial_vars['target_letter']):
			color = 'Lime'
		else :
			color = 'LightGray'
		# ec.screen_text(letter, letter_loc[i], )
		ec.screen_text(letter, letter_loc[i], color=color, font_size=18)

	# Refer to voc_meg.py and pupillometry.py and PylinkAPI.pdf for referenc on recording pupillometry
	ec.identify_trial(ec_id=id_, el_id=id_)    

	# load WAVs for this block
	ec.screen_text('loading...')
	stims = []
	stims.append(read_wav(trial_stims)[0])  # ignore fs
	ec.flip()

	# get ready
	stim_dur = stim.shape[-1] / ec.stim_fs
	ec.load_buffer(stim)

	# play stim
	ec.start_stimulus(flip=False)
	ec.wait_until(stim_dur)
	ec.stop()
	ec.trial_ok()

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
	# el = EyelinkController(ec)

	# Assert the subject id and session number exist
	# assert(os.path.isdir(op.join(stimDir, participant, session)), 'Can not find Stim directory.  Have you runCreateStims.m yet?')

	# READ IN GLOBAL VARIABLES FROM FILE
	# Reads in 'condition_bin', 'wheel_matrix_info', preblock_prime_sec 
	final_datadir = op.join(datadir, participant, session)
	global_vars = scipy.io.loadmat(op.join(final_datadir, 'global_vars.mat'))

	# VARIOUS VARIABLES
	block_in_sections = [1, 5, 8]
	trial_in_blocks = [8, 9, 1]
	stim_trial_indices = np.zeros(len(global_vars['condition_bin']))
	section_instr_counter = np.zeros(len(global_vars['wheel_matrix_info']))
	preblock = global_vars['preblock_prime_sec'][0]
	wait_brief = .2
	wait_long = 2

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

				# LOAD IN TRIAL DATA/STIMS
				paradigm_no = section[snum][bnum][tnum]
				trial_paradigm = global_vars['condition_bin'][paradigm_no]
				# id_ = decimals_to_binary(trial_paradigm, np.ones(1, len(trial_paradigm)))
				id_ = trial_paradigm 
				file_name = str(trial_paradigm) + '_tr' + str(tnum)
				trial_data = op.join(final_datadir, file_name)
				print trial_data
				# Creates dict of vars: 'target_letter', 'target_time', 'tot_wav_time', 'preblock_prime_sec', 'paradigm', 'possible_letters', 'preblock_prime_sec'
				trial_vars = scipy.io.loadmat(trial_data)
				trial_stims = op.join(stimdir, file_name)	

				# start trial
				ec.screen_prompt(instr[(section_base + 'start_trial_' + str(tnum))], live_keys=[cont_btn])
				# ec.identify_trial(ec_id=ecid, ttl_id=ttlid) # should include eyelink controller as well
				if (snum == 1):
					record_pupil = False
				recordTrial(trial_vars, global_vars, trial_stims, id_, stim_trial_indices, instr, ec, stimdir, datadir, record_pupil)	
				if (snum == 1):
					cogLoadSurvey(paradigm, gen_survey, mid_survey, rel_survey, id_, ec)
				# ec.trial_ok					

				# End trial
				ec.screen_prompt(instr[(section_base + 'end_trial')], live_keys=[], max_wait = wait_long)

			# End block
			ec.screen_prompt(instr[(section_base + 'end_block')], live_keys=[cont_btn])

		# End section
		ec.screen_prompt(instr[(section_base + 'end_sect')], max_wait=6.0, live_keys=[])

	# DASCogLoad drawing for reference
	# cur_cols = np.where(attn[block][tnum], 'Lime', 'LightGray').tolist()
	# txt_obj = []
	# for n, cat in enumerate(cur_cats):
	# 	cat = '<center>' + cat + '</center>'  # hack (pyglet bug)
	# 	txt_obj.append(ec.screen_text(cat, pos=[xpos[n], ypos[n]],
	# 								  color=cur_cols[n], font_size=36))
	# end_wait = ec.current_time + pretrial_wait
	# ec.flip()