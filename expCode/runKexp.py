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

#mainDir = /Users/nancygrulke/git/kexp
#stimDir 
# varsfile = 'expVars.mat'
stimdir = 'Stims'
datadir = 'data'

import scipy
import pyglet
# from glob import glob
import numpy as np
import math
import os.path as op
import expyfun as ef
from expyfun.io import read_wav
# from expyfun import (ExperimentController, EyelinkController, visual,
                     # get_keyboard_input, assert_version, decimals_to_binary)
from CircularLayout import CircularLayout
from text_kexp import *

#assert ef.__version__ == '2.0.0.DASCogLoad'

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
				  stim_db=65, noise_db=40,  # session='1', participant='foo',
				  stim_rms=0.01, check_rms=None, suppress_resamp=False,
				  output_dir=datadir, stim_fs=16000)  # 44100.0


# RANDOM NUMBER GENERATOR
rng = np.random.RandomState(0)

# # GROUP FILENAMES OF STIMULI BY PARADIGM BINARY STRING
# stim_blocks['t_one'] = sorted(glob(op.join(stimdir, 'train-one*')))

# VARIOUS VARIABLES
block_in_sections = [1, 5, 8]
stim_trial_indices = np.zeros(len(condition_binary))
section_instr_counter = np.zeros(len(wheel_matrix_info))

# DEFINE TRIAL RECORDING FUNCTION
def recordTrial(stim_trial_indices, instr_key, condition_binary, trial_target_times, possible_letters, wheel_matrix_info, ec, trial_path, record_pupil = True):
	""" Takes the indice of all current condition types and the binary name
	of the condition to find the trial wav.  Displays instructions according
	to intr_key and plays stimuli while recording and logging 
	pupillometry data.  Pylink functionality to be added last"""

	# DRAW SELECTIONS ON SCREEN
	wheel_loc = CircularLayout(3, radius = .6, angles = [- phi, (phi / 2), 0]) # must be three wheels
	letter_loc = [];
	for i in len(wheel_matrix_info):
		temp = CircularLayout(wheel_matrix_info[i], radius = .15, relative_center = wheel_loc.positions[i])
		letter_loc.append(temp.positions)

	# letterloc1 = CircularLayout(wheel_matrix_info[0], relative_center = wheel_loc.positions[0])
	# letterloc1 = CircularLayout(wheel_matrix_info[1], relative_center = wheel_loc.positions[1])
	# letterloc1 = CircularLayout(wheel_matrix_info[2], relative_center = wheel_loc.positions[2])

	# DASCogLoad drawing for reference
	# cur_cols = np.where(attn[block][tnum], 'Lime', 'LightGray').tolist()
	# txt_obj = []
	# for n, cat in enumerate(cur_cats):
	# 	cat = '<center>' + cat + '</center>'  # hack (pyglet bug)
	# 	txt_obj.append(ec.screen_text(cat, pos=[xpos[n], ypos[n]],
	# 								  color=cur_cols[n], font_size=36))
	# end_wait = ec.current_time + pretrial_wait
	# ec.flip()

	# Refer to voc_meg.py and pupillometry.py and PylinkAPI.pdf for referenc on recording pupillometry

	# load WAVs for this block
	ec.screen_text('loading...')
	stims = []
	stims.append(read_wav(path)[0])  # ignore fs
	ec.flip()


	# get ready
	stim_dur = stim.shape[-1] / ec.stim_fs
	ec.load_buffer(stim)
	ec.wait_until(end_wait)

	# play stim
	ec.start_stimulus(flip=False)
	ec.stop()

	# clear screen
	ec.flip()

	# write out data
	ec.write_data_line('target_time', target_time)
	ec.trial_ok()

	return stim_trial_indices

def cogLoadSurvey(paradigm, gen_survey, mid_survey, rel_survey, ec):
	ec.write_data_line('cogLoadSurvey')
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
section = (); 

#  Make sections 1 and 3 structure
reps = (1, 1)
for snum in range(2):
	for bnum in range(8):
		block[bnum] = tile((bnum), (reps[snum], 1));
	section[snum] = block;

section[2] = section[1];
section[1] = ();

# Make section 2 structure
# This needs to be pseudo randomly generated
block[0] = (1, 2, 3, 2, 1, 3, 1)
block[1] = (3, 2, 1, 5, 4, 1, 4, 5, 1)
block[2] = (1,4,5, 1,5, 4, 6,5, 1 )
block[3] =(5,6,1, 6,5,1,5) 
block[4] = (1,8,7,8,1,7, 1,7, 8)
section[1] = (block[0], block[1], block[2], block[3], block[4]);

# RUN EXPERIMENT
with ef.ExperimentController(*std_args, **std_kwargs) as ec:
    el = EyelinkController(ec)

	# Assert the subject id and session number exist
	assert(os.path.isdir(op.join(stimDir, participant, session)), 'Can not find Stim directory.  Have you runCreateStims.m yet?')

	# READ IN GLOBAL VARIABLES FROM FILE
	global_vars = scipy.io.loadmat(op.join(datadir, participant, session, 'global_vars.mat'))
	ec.start_noise()

	for snum in len(section):

		# Initialize section vars
		section_base = str('s' + snum + '_')
		ec.screen_prompt(instr[(section_base + 'start_sect')], live_keys=[cont_btn])

		# run blocks
		for bnum, block in section[snum]:
			# show instructions
			ec.screen_prompt(instr[(section_base + 'start_block_' + str(bnum))], live_keys=[cont_btn])

			# log block name
			ec.write_data_line('block', block)

			for tnum in section[snum][bnum]: # +++ check this 
				# LOAD IN TRIAL DATA
				# Creates dict of vars: 'target_letter', 'target_time', 'token_rate_modulation', 'tot_wav_time', 'condition_no', 'possible_letters'
				trial_data = scipy.io.loadmat(trial_path)

				# start trial
				ec.screen_prompt(instr[(section_base + 'start_trial_' + str(tnum))], live_keys=[cont_btn])
				ec.identify_trial(ec_id=ecid, ttl_id=ttlid) # should include eyelink controller as well
				if (snum == 2):
					record_pupil = False
				recordTrial(stim_trial_indices, instr_key, condition_binary, trial_target_times, possible_letters, wheel_matrix_info, ec, trial_path, record_pupil)	
				if (snum == 2):
					cogLoadSurvey(paradigm, gen_survey, mid_survey, rel_survey, ec)


				# End trial
				ec.screen_prompt(instr[(section_base + 'end_trial')], live_keys=[cont_btn])

			# End block
			ec.screen_prompt(instr[(section_base + 'end_block')], live_keys=[cont_btn])

		# End section
		ec.screen_prompt(instr[(section_base + 'end_sect')], max_wait=6.0, live_keys=[])

	# JUNK
				# iterate
				# cnum += 1
				# if block not in tr_blocks:
				# 	cum_trial += 1
				# if block in tr_blocks and tnum == len(stims) - 1:
				# 	tnum = 0
				# else:
				# 	tnum += 1

			# # loop through trials in this block
			# tnum = 0
			# cnum = 0  # cumulative training trial number
			# while tnum < len(stims):
			# 	# training warning?
			# 	if cnum == 10:
			# 		if block in ('t_one', 't_two', 't_fa_'):
			# 			ec.screen_prompt(instr['twrna'])
			# 		elif block in ('t_faa', 't_fab'):
			# 			ec.screen_prompt(instr['twrnb'])
			# 	elif cnum == 20 and block in tr_blocks:
			# 		ec.screen_prompt(instr['tfail'], max_wait=20.0, live_keys=[])
			# 		ec.close()

			# 	stim = stims[tnum]

				# logging
				# if block in tr_blocks:
				# 	ecid = '{}-{}-{}-{}-{}'.format('training block {} trial {}'.format(bnum, tnum),
				# 								   ''.join(np.char.asarray(attn[block][tnum]).ravel()),
				# 								   ''.join(np.char.asarray(targ_locs[block][tnum]).ravel()),
				# 								   ' '.join(words[block][tnum].ravel()),
				# 								   ' '.join(np.char.asarray(onset_sec[block][tnum]).ravel()))
				# else:
				# 	ecid = np.binary_repr(cum_trial, width=8)

				# ttlid = np.array(list(np.binary_repr(cum_trial, width=8)), int)  # what does this do??

	# counterbalance experiment order across subjects
	# if int(ec._exp_info['session']) % 2 == 0:
	#     order = tr_blocks + sm_blocks + md_blocks
	# else:
	#     order = tr_blocks + md_blocks + sm_blocks

