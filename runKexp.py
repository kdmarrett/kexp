"""
================================
Script ' Karl's Experiment (Kexp)'
================================

This script runs an experiment with spatially distributed letter streams.
"""
# Author: Karl Marrett <kdmarret@uw.edu>

# TO DO
# add in pupil script from voc_meg and pupil...py
# catch vars.mat
# reading from 
# check the subject id and session number exist

#mainDir = /Users/nancygrulke/git/
#stimDir 
# varsfile = 'expVars.mat'
stimdir = 'Stims'
datadir = 'data'

import scipy
import pyglet
from glob import glob
import numpy as np
import os.path as op
import expyfun as ef
from expyfun import ExperimentController
import expyfun.analyze as ea

#assert ef.__version__ == '2.0.0.DASCogLoad'

#READ IN WAVFILES
isi = 0.5
wait_dur = 3.0
msg_dur = 3.0

# EXPERIMENT PARAMETERS
cont_btn = 8
cont_btn_label = 'Next'
resp_btn = 1
min_rt = 0.1
max_rt = 1.25
pretrial_wait = 2.5
feedback_dur = 1.5
isi = 0.2
std_args = ['kexp']
std_kwargs = dict(screen_num=0, window_size=[800, 600], full_screen=True,
				  stim_db=65, noise_db=40,  # session='1', participant='foo',
				  stim_rms=0.01, check_rms=None, suppress_resamp=False,
				  output_dir=datadir, stim_fs=24414)  # 44100.0

# RANDOM NUMBER GENERATOR
rng = np.random.RandomState(0)

# # GROUP FILENAMES OF STIMULI BY BLOCK
# stim_blocks['t_one'] = sorted(glob(op.join(stimdir, 'train-one*')))
# stim_blocks['t_two'] = sorted(glob(op.join(stimdir, 'train-two*')))
# stim_blocks['t_fa_'] = sorted(glob(op.join(stimdir, 'train-four-a-*')))
# stim_blocks['t_faa'] = sorted(glob(op.join(stimdir, 'train-four-aa*')))
# stim_blocks['t_fab'] = sorted(glob(op.join(stimdir, 'train-four-ab*')))
# stim_blocks['e_one'] = sorted(glob(op.join(stimdir, 'trial-0[01]*')))  # 0-19
# stim_blocks['e_two'] = sorted(glob(op.join(stimdir, 'trial-0[23]*')))  # 20-39
# stim_blocks['e_thr'] = sorted(glob(op.join(stimdir, 'trial-0[45]*')))  # etc
# stim_blocks['e_fou'] = sorted(glob(op.join(stimdir, 'trial-0[67]*')))
# stim_blocks['e_fiv'] = sorted(glob(op.join(stimdir, 'trial-0[89]*')))
# stim_blocks['e_six'] = sorted(glob(op.join(stimdir, 'trial-1*')))

# INSTRUCTIONS AND FEEDBACK
#instr_args = [cont_btn_label, resp_btn, streams, bnum]  # , 2 * blocks]
instr = dict()
instr[0] = ('In this experiment you will be listening to letter streams '
				  'in different spatial locations. The letters will contain'
				  'the alphabet with the extra commands '
				  'Most trials will have letters grouped consecutively.'
						  'Your job is to press the "{} button when you '
				  'hear that the target letter that does not match the target letter '
				  'spatial location where the word occurred. There will be a '
				  'little bit of background noise to make the task more '
				  'challenging. Push {} to continue.').format(resp_btn, cont_btn_label)

# TRAINING INSTRUCTIONS
# instr['start_train'] = ('Practice first with just one letter stream. Push "{}" when '
#                   'you\'re ready to start, then push the {} button when you '
#                   'hear a word that does not match the category.'
#                   ''.format(cont_btn_label, resp_btn))
# instr['t_two'] = ('Now there will be two word streams. Listen to BOTH streams '
#                   'and press the {} button for words in EITHER stream that '
#                   'don\'t match the category of the stream they occur in. '
#                   'Push "{}" to start.'.format(resp_btn, cont_btn_label))
# instr['t_fa_'] = ('Good job. This time there will be four streams. Ignore the '
#                   'streams whose category names are grey, and attend to the '
#                   'stream whose category name is green. When you hear words '
#                   'in that stream that don\'t match the category, press {}. '
#                   'Push "{}" to start.'.format(resp_btn, cont_btn_label))
# instr['t_faa'] = ('Good job. This time, you will have to attend to streams in '
#                   'two different locations at the same time, but the '
#                   'category will be the same across the two locations. Push '
#                   '"{}" to start, then when you hear words in either stream '
#                   'that don\'t match the category, press {}.'
#                   ''.format(cont_btn_label, resp_btn))
# instr['t_fab'] = ('Good job. Time for the last training block. Like the last '
#                   'round, there are four streams and you have to ignore the '
#                   'ones whose categories are grey. Only this time, the two '
#                   'green categories will be different from each other. Press '
#                   '"{}" to start, then when you hear a word in either one of '
#                   'the green streams that doesn\'t match the category of that '
#                   'stream, press {}.'
#                   ''.format(cont_btn_label, resp_btn))
# instr['tpass'] = ('Good job! You passed the training. Press "{}" to move on '
#                   'to the experiment.'.format(cont_btn_label))
# instr['twrna'] = ('It seems like you\'re struggling with this part of the '
#                   'training. To pass, you will need to get all targets '
#                   'correct with no extra button presses, on two trials in a '
#                   'row. If you want to keep trying, press "{}". If you want '
#                   'to stop the experiment now, you can just get up and leave '
#                   'the booth (you will still be paid for the time you spent '
#                   'so far).'.format(cont_btn_label))
# instr['twrnb'] = ('It seems like you\'re struggling with this part of the '
#                   'training. To pass, you will need to do two trials in a '
#                   'row with no more than one mistake in each. If you want to '
#                   'keep trying, press "{}". If you want '
#                   'to stop the experiment now, you can just get up and leave '
#                   'the booth (you will still be paid for the time you spent '
#                   'so far).'.format(cont_btn_label))
# instr['tfail'] = ('You have not passed this part of the training after 20 '
#                   'attempts, so unfortunately we cannot let you continue '
#                   'with the experiment (please don\'t feel bad; it is a hard '
#                   'task and not everyone can do it well). Thank you for '
#                   'participating; you will still be paid for the time you '
#                   'spent so far.')

instr['e_one'] = ('In this half of the experiment the categories have three '
				  'words each (just like in the training). There are 20 '
				  'trials in this block; you won\'t get any feedback whether '
				  'each trial was correct, and new trials will begin '
				  'automatically shortly after the previous trial finishes. '
				  'There will be three blocks like this, with breaks in '
				  'between blocks. When you\'re ready, to begin this block, '
				  'press "{}" to begin, then use the {} button to respond.'
				  ''.format(cont_btn_label, resp_btn))
instr['e_two'] = ('You\'re about to begin the second block in this half of '
				  'the experiment. Reminder: there is no feedback, and new '
				  'trials start automatically shortly after the previous '
				  'trial ends. There are 20 trials in this block and three '
				  'words in each category. Press "{}" to begin and then press '
				  '{} to respond during the '
				  'trials.'.format(cont_btn_label, resp_btn))
instr['e_thr'] = ('You\'re about to begin the last block in this half of '
				  'the experiment. Reminder: there is no feedback, and new '
				  'trials start automatically shortly after the previous '
				  'trial ends. There are 20 trials in this block and three '
				  'words in each category. Press "{}" to begin and then press '
				  '{} to respond during the '
				  'trials.'.format(cont_btn_label, resp_btn))
instr['e_fou'] = ('In this half of the experiment the categories are '
				  'different, and they have six words each (instead of three, '
				  'like in the training). '
				  'There are 20 trials in this block; you won\'t get any '
				  'feedback whether each trial was correct, and new trials '
				  'will begin automatically shortly after the previous trial '
				  'finishes. There will be three blocks like this, with '
				  'breaks in between blocks. When you\'re ready, to begin '
				  'this block, press "{}" to begin, then use the {} button to '
				  'respond.'.format(cont_btn_label, resp_btn))
instr['e_fiv'] = ('You\'re about to begin the second block in this half of '
				  'the experiment. Reminder: there is no feedback, and new '
				  'trials start automatically shortly after the previous '
				  'trial ends. There are 20 trials in this block and six '
				  'words in each category. Press "{}" to begin and then press '
				  '{} to respond during the '
				  'trials.'.format(cont_btn_label, resp_btn))
# instr['e_six'] = ('You\'re about to begin the last block in this half of '
#                   'the experiment. Reminder: there is no feedback, and new '
#                   'trials start automatically shortly after the previous '
#                   'trial ends. There are 20 trials in this block and six '
#                   'words in each category. Press "{}" to begin and then press '
#                   '{} to respond during the '
#                   'trials.'.format(cont_btn_label, resp_btn))
instr['midpt'] = ('Good work! You\'re done with the first section of the '
				  'experiment. Take a break (you can leave the booth if you '
				  'need to). Press "{}" when you\'re ready to resume.'
				  ''.format(cont_btn_label))
instr['edone'] = ('All done! Thank you very much for participating!')
instr['break'] = ('Good job! Take a break if you like, then press "{}" when '
				  'you\'re ready for the next block.'.format(cont_btn_label))

# VARIOUS VARIABLES
cum_trial = 0  # cumulative trial counter (not including training)
xpos = [-0.75, -0.25, 0.25, 0.75]  # on-screen text locations
ypos = [-0.25, 0.25, 0.25, -0.25]
curr = False  # for passing training
prev = False
block_in_sections = [7 5 7]
stim_trial_indices = np.zeros(len(condition_binary))

# DEFINE TRIAL RECORDING FUNCTION
def recordTrial(stim_trial_indices, instr_key, condition_binary, record_pupil, trial_target_times, possible_letters, wheel_matrix_info = )
	 ''' Takes the indice of all current condition types and the binary name
	of the condition to find the trial wav.  Displays instructions according
	to intr_key and plays stimuli while recording and logging 
	pupillometry data.  Pylink functionality to be added last'''

	# draw categories on screen
	cur_cats = tuple(cats[block][tnum])
	cur_cols = np.where(attn[block][tnum], 'Lime', 'LightGray').tolist()
	txt_obj = []
	for n, cat in enumerate(cur_cats):
		cat = '<center>' + cat + '</center>'  # hack (pyglet bug)
		txt_obj.append(ec.screen_text(cat, pos=[xpos[n], ypos[n]],
									  color=cur_cols[n], font_size=36))
	end_wait = ec.current_time + pretrial_wait
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
	ec.write_data_line('target_times', t_times)
	ec.trial_ok()

	return stim_trial_indices

# Make tuples
section = (); 

#  Make sections 1 and 3 structure
reps = (1, 2);
for snum in range(2)
	for bnum in range(8)
		block[bnum] = tile((bnum), (reps[snum], 1));
	section[snum] = block;

section[2] = section[1];
section[1] = ();

# Make section 2 structure
block[0] = (1, 2, 3, 2, 1 3, 1)
block[1] = (3, 2, 1, 5, 4, 1, 4, 5, 1)
block[2] = (1,4,5 1,5 4, 6,5, 1 )
block[3] =(5,6,1, 6,5,1,5) 
block[4] = (1,8,7,8,1,7, 1,7, 8)
section[1] = (block[0], block[1], block[2], block[3], block[4]);

# RUN EXPERIMENT
with ef.ExperimentController(*std_args, **std_kwargs) as ec:
	# counterbalance experiment order across subjects
	# if int(ec._exp_info['session']) % 2 == 0:
	#     order = tr_blocks + sm_blocks + md_blocks
	# else:
	#     order = tr_blocks + md_blocks + sm_blocks
	for snum in section
		ec.screen_prompt(instr['start'], live_keys=[cont_btn])
		ec.start_noise()

		# run blocks
		for bnum, block in section[snum]:
			# training passed?
			# if bnum == len(tr_blocks):
			#     ec.screen_prompt(instr['tpass'], live_keys=[cont_btn])
			# # halfway point?
			# elif bnum == len(tr_blocks) + len(sm_blocks):
			#     ec.screen_prompt(instr['midpt'], live_keys=[cont_btn])
			# # between blocks
			# elif bnum > len(tr_blocks):
			ec.screen_prompt(instr['break'], live_keys=[cont_btn])

			# log block name
			ec.write_data_line('block', block)

			# load WAVs for this block
			ec.screen_text('loading...')
			stims = []
			for path in stim_blocks[block]:
				stims.append(ef.stimuli.read_wav(path)[0])  # ignore fs
			ec.flip()

			# show instructions
			ec.screen_prompt(instr[block], live_keys=[cont_btn])

			# loop through trials in this block
			tnum = 0
			cnum = 0  # cumulative training trial number
			while tnum < len(stims):
				# training warning?
				if cnum == 10:
					if block in ('t_one', 't_two', 't_fa_'):
						ec.screen_prompt(instr['twrna'])
					elif block in ('t_faa', 't_fab'):
						ec.screen_prompt(instr['twrnb'])
				elif cnum == 20 and block in tr_blocks:
					ec.screen_prompt(instr['tfail'], max_wait=20.0, live_keys=[])
					ec.close()

				stim = stims[tnum]

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
				ec.identify_trial(ec_id=ecid, ttl_id=ttlid)

				# iterate
				cnum += 1
				if block not in tr_blocks:
					cum_trial += 1
				if block in tr_blocks and tnum == len(stims) - 1:
					tnum = 0
				else:
					tnum += 1

	# finished!
	ec.screen_prompt(instr['edone'], max_wait=6.0, live_keys=[])
