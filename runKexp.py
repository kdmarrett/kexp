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
# assert the subject id and session number exist
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
from glob import glob
import numpy as np
import os.path as op
import expyfun as ef
from expyfun import ExperimentController, get_keyboard_input
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

# # GROUP FILENAMES OF STIMULI BY PARADIGM BINARY STRING
# stim_blocks['t_one'] = sorted(glob(op.join(stimdir, 'train-one*')))

# INSTRUCTIONS AND FEEDBACK
#instr_args = [cont_btn_label, resp_btn, streams, bnum]  # , 2 * blocks]
instr = dict()

# SECTION 1 (INTRODUCTION) INSTRUCTIONS
instr[0] = ('In this experiment you will be listening to the letters of the alphabet grouped into'
	  'three different spatial locations left, middle, and right. Together, the groups will span'
	  'the full alphabet with the extra commands \'Read\' , \'Pause\', \'Space\', and \'Delete\'. '
	  'Most trials will have letters grouped consecutively in alphabetic order.'
	  'You will also notice that other than the voice commands each spatial location has a distinct'
	  ' speaker for the letters. Your task is to hear every occurence of your target letter although '
	  'you will not be given any feedback on your performance. There will also be a little bit of background noise to make the task more '
	  'challenging. Push {} to continue.').format(cont_btn_label)

instr[] = ('In this first section you will just be introduced to the different condition types. This section has '
	' 1 block with a trial for every condition for a total of 8 trials. Ignore the letters that are colored'
	'grey, and attend to the green highlighted letter that is spoken at the beginning of each trial.'
	'Don\'t worry if hearing the letter is difficult at first.  Just try to attend every occurrence of the target letter.' )
	'for the entire trial. Push {} to start.'.format(cont_btn_label)

instr[1] = ('In this condition letters are assigned a spatial location in alphabetic order with their own'
	'individual tone.  You can use the speaker, the location, and the tone to help you listen for each occurrence of'
	'the target letter in this condition.  Try listening to this condition now by pushing {} to continue').format(cont_btn_label)

instr[2] = ('In this condition letters are assigned a spatial location in alphabetic order with their own'
	'individual tone.  In this conditon however, each location is given a different speed. You can'
	' use the speaker, the location, the tone,  and the speed to help you listen for each occurrence of'
	'the target letter in this condition.  Try listening to this condition now by pushing {} to continue').format(cont_btn_label)

instr[3] = ('In this condition letters are assigned a spatial location in alphabetic order with their own'
	'individual tone. The targets in this condition are always one of \'B\', \'C\'\'D\'\'E\'\'G\'\'P\'\'T\''
	'\'V\', or \'Z\' You can use the speaker, the location, and the tone to help you listen for each occurrence of'
	'the target letter in this condition.  Try listening to this condition now by pushing {} to continue').format(cont_btn_label)

instr[4] = ('In this condition letters are assigned a spatial location in alphabetic order with their own'
	'individual tone.  In this conditon after each cycle of letters the ordering will change. This means'
	'You can use the speaker, the location, but not the tone to help you listen for each occurrence of'
	'the target letter in this condition.  Try listening to this condition now by pushing {} to continue').format(cont_btn_label)

instr[5] = ('In this condition letters are assigned a spatial location in alphabetic order with their own'
	'individual tone.  Each letter is assigned a unique tone but these tones are assigned randomly, '
	'meaning that you can get not rely on the any particular pattern of tones to help guide you.  '
	'You can use the speaker, the location, and the unique tone to help you listen for each occurrence of'
	'the target letter in this condition.  Try listening to this condition now by pushing {} to continue').format(cont_btn_label)

instr[6] = ('In this condition letters are assigned a spatial location in alphabetic order with their own'
	'individual tone. In this condition the loudness of each speaker is oscillating at the same rate'
	' although the phase of each speaker will be unique.  This means you can use the speaker, the'
	' location, the tone,  and the differences in loudness to help you listen for each occurrence of'
	'the target letter in this condition.  Try listening to this condition now by pushing {} to continue').format(cont_btn_label)

instr[7] = ('In this condition letters are assigned a spatial location in alphabetic order with their own'
	'individual tone. In this condition the loudness of each speaker is oscillating at the different rates'
	'This means you can use the speaker, the location, the tone,  and the differences in oscillation' 
	' rate to help you listen for each occurrence of the target letter in this condition.  Try listening'
	' to this condition now by pushing {} to continue').format(cont_btn_label)

instr[8] = ('In this condition letters are assigned a spatial location in random order with their own'
	'individual tone. Despite the random ordering, each letter has the same tone throughout the'
	'trial and the tones are ascended in a regular ascending or descending pattern. You can'
	' use the speaker, the location, the tone,  and the speed to help you listen for each occurrence of'
	'the target letter in this condition.  Try listening to this condition now by pushing {} to continue').format(cont_btn_label)

instr[9] = ('Good work! You\'re done with the first section of the '
	  'experiment. Take a break (you can leave the booth if you '
	  'need to). Press "{}" when you\'re ready to resume.'.format(cont_btn_label))

# SECTION 2 INSTRUCTIONS
instr[] = ('You\'re about to begin the second section of the experiment. Reminder: Just as'
	' in the introduction section there is no feedback, and new trials of different conditions'
	'will start automatically shortly after the previous trial ends. Note that in this section the different condition'
	' are interspersed meaning that you must orient yourself to the particular condition by either the primer or'
	' by listening to the unique cues of the particular trial. There are about 9 trials in each'
	' block and 5 blocks in this section, with breaks in between blocks. When you\'re ready, to begin press "{}"'.format(cont_btn_label))

instr['break'] = ('Good job! Take a break if you like, then press "{}" when '
				  'you\'re ready for the next block.'.format(cont_btn_label))

instr[] = ('You\'re about to begin the second block in this section of '
	  'the experiment. Reminder: there is no feedback, and new '
	  'trials start automatically shortly after the previous '
	  'trial ends. There are 20 trials in this block and three '
	  'words in each category. Press "{}" to begin.'.format(cont_btn_label))

instr[] = ('You\'re about to begin the third block in this section of '
	  'the experiment. Reminder: there is no feedback, and new '
	  'trials start automatically shortly after the previous '
	  'trial ends. There are 20 trials in this block and three '
	  'words in each category. Press "{}" to begin.'.format(cont_btn_label))

instr[] = ('You\'re about to begin the fourth block in this section of '
	  'the experiment. Reminder: there is no feedback, and new '
	  'trials start automatically shortly after the previous '
	  'trial ends. There are 20 trials in this block and three '
	  'words in each category. Press "{}" to begin.'.format(cont_btn_label))

instr[] = ('You\'re about to begin the last block in this section of '
	  'the experiment. Reminder: there is no feedback, and new '
	  'trials start automatically shortly after the previous '
	  'trial ends. There are 20 trials in this block and three '
	  'words in each category. Press "{}" to begin.'.format(cont_btn_label))

instr[] = ('Good work! You\'re done with the second section of the '
	  'experiment. Take a break (you can leave the booth if you '
	  'need to). Press "{}" when you\'re ready to resume.'.format(cont_btn_label))

# SECTION 3 COGNTIVE LOAD ASSESSMENT INSTRUCTIONS
survey_text = dict();
instr[] = ('You\'re about to begin the last section of the experiment. Reminder: Just as'
	' in the introduction section there is no feedback, but in this section you be presented with 2'
	' trials of the same condition then asked to answer several questions about the relative'
	'difficulty of each condition.  In this secton there are eight blocks for each section.'
	' Press "{}" to begin'.format(cont_btn_label))

survey_text[] = ('We are interested in the experiences you had during the experiment. In the most general sense we'
		'are examining the “workload” you experienced. The factors that influence you experience of'
		'workload may come from the task itself, your feelings about your own performance, how much effort'
		'you put in, or the stress and frustration you felt. Because workload may be caused by many different'
		'factors, we would like you to evaluate several of them.  '
		'Please read the descriptions of the set of six scales carefully. If you have a question about any of the'
		'scales in the table, please ask me about it. It is important that they be clear to you. You may keep the'
		'descriptions with you for reference during the study.').format(cont_btn_label)

survey_text[] = ('After performing the experiment you will be given a sheet of rating scales. Each line has two endpoint'
		' descriptors that describe the scale. Please consider each rating scale and condition type separately'
		' when answering the questions.').format(cont_btn_label)

survey_text[] = ('How mentally demanding was the task? How much mental and perceptual activity was required (e.g.'
		'thinking, deciding, remembering, looking, searching)? Was the task easy or demanding, simple or '
		'complex, exacting or forgiving?  When you are finished entering in a value, press {} to continue.').format(cont_btn_label)

survey_text[] = ('How much physical activity was required (e.g. pushing, pulling, turning, controlling, activating)?'
		' Was the task easy or demanding, slow or brisk, slack or strenuous, restful or laborious?When'
		' you are finished entering in a value, press {} to continue.').format(cont_btn_label)

survey_text[] = ('How much time pressure did you feel due to the rate or pace at which the task or task elements'
		' occurred? Was the pace slow and leisurely or rapid and frantic? When you are finished entering'
		' in a value, press {} to continue.').format(cont_btn_label)

survey_text[] = ('How successful do you think you were in accomplishing the goals of the task set by the'
		' experimenter (or yourself)? How satisfied were you with your performance in accomplishing'
		' these goals?  When you are finished entering in a value, press {} to continue.').format(cont_btn_label)

survey_text[] = ('How hard did you have to work (mentally and physically) to accomplish your level of performance?'
		' When you are finished entering in a value, press {} to continue.').format(cont_btn_label)

survey_text[] = ('How insecure, discouraged, irritated, stressed and annoyed versus secure, gratified, content,'
		 'relaxed and complacent did you feel during the task? When you are finished entering in a value,'
		 ' press {} to continue.').format(cont_btn_label)

survey_text[] = ('Throughout this experiment the rating scales are used to assess your experiences in the different task'
		'conditions. People interpret these scales in individual ways. For example, some people feel that'
		'mental or temporal demands are the essential aspects of workload regardless of the effort they'
		'expended on a given task or the level of performance they achieved.'
		'The evaluation you are about to perform is a technique to assess the relative importance of six factors'
		'in determining how much workload you experienced.'.format(cont_btn_label)

survey_text[] = 'The procedure is simple: You will be presented with a series of pairs of rating scale titles (for'
		'example Effort versus Mental Demands) and asked to choose which of the items was more'
		'important to your experience of workload in the task that you just performed.'
		'Indicate the Scale Title that represents the more important contributor to workload for the specific'
		'task you performed in the experiment.'
		'Please consider your choices carefully and make them consistent with how you used the rating scales'
		'during the particular task you were asked to evaluate. Do not think there is any correct pattern: we'
		'are only interested in your opinions.')

survey_text[1] = '1. Physical Demand or 2. Temporal Demand. Press the number corresponding to the aspect that you consider the larger factor contributing to the workload.'
survey_text[1] = '1. Effort or 2. Mental Demand. Press the number corresponding to the aspect that you consider the larger factor contributing to the workload.'
survey_text[1] = '1. Frustration or 2. Physical Demand. Press the number corresponding to the aspect that you consider the larger factor contributing to the workload.'
survey_text[1] = '1. Effort or 2. Frustration. Press the number corresponding to the aspect that you consider the larger factor contributing to the workload.'
survey_text[1] = '1. Mental Demand or 2. Temporal Demand. Press the number corresponding to the aspect that you consider the larger factor contributing to the workload.'
survey_text[1] = '1. Physical Demand or 2. Effort. Press the number corresponding to the aspect that you consider the larger factor contributing to the workload.'
survey_text[1] = '1. Temporal Demand or 2. Performance. Press the number corresponding to the aspect that you consider the larger factor contributing to the workload.'
survey_text[1] = '1. Frustration or 2. Mental Demand. Press the number corresponding to the aspect that you consider the larger factor contributing to the workload.'
survey_text[1] = '1. Temporal Demand or 2. Frustration. Press the number corresponding to the aspect that you consider the larger factor contributing to the workload.'
survey_text[1] = '1. Performance or 2. Effort. Press the number corresponding to the aspect that you consider the larger factor contributing to the workload.'
survey_text[1] = '1. Effort or 2. Temporal Demand. Press the number corresponding to the aspect that you consider the larger factor contributing to the workload.'
survey_text[1] = '1. Frustration or 2. Performance. Press the number corresponding to the aspect that you consider the larger factor contributing to the workload.'
survey_text[1] = '1. Performance or 2. Physical Demand. Press the number corresponding to the aspect that you consider the larger factor contributing to the workload.'
survey_text[1] = '1. Mental Demand or 2. Performance. Press the number corresponding to the aspect that you consider the larger factor contributing to the workload.'
survey_text[1] = '1. Mental Demand or 2. Physical Demand. Press the number corresponding to the aspect that you consider the larger factor contributing to the workload.'
instr['done'] = ('All done! Thank you very much for participating!')

# VARIOUS VARIABLES
cum_trial = 0  # cumulative trial counter (not including training)
xpos = [-0.75, -0.25, 0.25, 0.75]  # on-screen text locations
ypos = [-0.25, 0.25, 0.25, -0.25]
curr = False  # for passing training
prev = False
block_in_sections = [7 5 ]
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

def cogLoadSurvey(paradigm,  survey_text, ec):
	for qnum in len(survey_text)
		ec.write_data_line(str(paradigm) + str(qnum))
		ec.write_data_line(input(survey_text([qnum])))


# MAKE BLOCK DESIGN
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
				ec.identify_trial(ec_id=ecid, ttl_id=ttlid)

				# iterate
				# cnum += 1
				# if block not in tr_blocks:
				# 	cum_trial += 1
				# if block in tr_blocks and tnum == len(stims) - 1:
				# 	tnum = 0
				# else:
				# 	tnum += 1

	# finished!
	ec.screen_prompt(instr['done'], max_wait=6.0, live_keys=[])
