# text_kexp.py
# Author: Karl Marrett
# Text and other related dictionaries for runKexp.py script

import numpy as np


# add in how letters are ordered at the beginning
# explaining pattern after the fact for first trial (ascending and descending)
cont_btn = 0
cont_btn_label = '0'

wait_brief = .2
wait_long = 2

# INSTRUCTIONS AND FEEDBACK
instr = dict()
button_keys = dict()

# SECTION 1 (INTRODUCTION) INSTRUCTIONS
instr['cond_pass'] = ('Excellent!  You passed the training for this'
' condition')

instr['end_train'] = ('Congratulations! You finished all the training.')

instr['train_fail'] = ('Sorry, you did not pass the training section '
'of the experiment. Unfortunately, you can not continue with the '
'experiment but you will be paid for your time.  You may exit the booth '
'now.')

instr['start_train'] = ('In this part of the experiment, we will test'
        ' your ability to identify the correct number of target letters in a'
        ' given trial.  For each condition you must get two consecutive'
        ' trials correct in order to continue. '
        'To continue, press "{}" .'.format(cont_btn_label))

instr['more_train'] = ('Your job for this experiment is to listen to '
		'the number of times the target letter occurs.  The target '
		'letter is spoken at the location it will be presented at and '
		'is shown in green at the location it will be spoken at '
                'during the beginning of the trial.  At the end '
		'of the trial, the fixation dot will turn yellow, prompting you'
		' to enter the number of targets your heard.  During'
                ' this training '
		' you will be given feedback as to whether your response was'
		' correct. When your response was correct, the dot will'
                ' change to green you entered the incorrect amount of '
                'targets it will instead shift to red.  When you are ready '
                ' to start press "{}" to continue.'.format(cont_btn_label))

instr['start_exp'] =('In this experiment you will be listening to letters grouped into '
                      'three different spatial locations (left, middle, and right). '
                      ' Your task for each trial in this experiment is'
                      ' to count every occurrence of your target letter for the duration of'
                      ' the trial. '
                       'This experiment is split into 3 sections, each with a '
                       'varying number of blocks which contain grouped trials.  '
                       'Between each section, there will'
                      ' be an opportunity to take a break and leave the booth.'
                      '  Push "{}" to continue.'.format(cont_btn_label))

button_keys['start_exp'] = [cont_btn]

instr['s0_start_sect'] = ('This experiment will also assess pupil responses, so after the primer for each trial a small fixation dot'
                          ' will appear in the center of screen. Keep your eyes fixated on this dot for as long as it remains visible. '
                          'Push "{}" to continue.'.format(cont_btn_label))

button_keys['s0_start_sect'] = [cont_btn]

instr['s0_start_block_0'] = ( 'In this first section, you will be introduced to the different condition types.'
				' Each letter group, left, middle, and right, has a distinct speaker for the letters. To orient you to the layout'
					  ' and ordering, each trial will begin with a visual primer of the spatial location of the letters. '
					  'In these primers, you will notice that one of the letters is colored green and spoken, this is the target letter for the trial. '
					  'Together, the groups will span the full alphabet along with the extra voiced commands \'Read\' , \'Pause\', \'Space\', and \'Delete\'. '
							' Again, pay attention to the occurrences of the' 
                             ' target green highlighted letter that is '
							 'spoken at the beginning of each trial. '
                             ' There will also be background noise to make the'
							 ' task more challenging. '
                             'Don\'t worry if the letter is difficult to hear at first. Do your best to attend every occurrence of the target letter throughout the trial. '
                             'Push "{}" to start.'.format(cont_btn_label))

button_keys['s0_start_block_0'] = [cont_btn]

instr['s0_start_trial_0'] = ('In this condition, letters are assigned a spatial location in alphabetic order.'
                             ' You can use the unique speaker and location to help you listen for each occurrence of'
                             ' the target letter. Push "{}" to start.'.format(cont_btn_label))

button_keys['s0_start_trial_0'] = [cont_btn]

# instr['s0_start_trial_1'] = ('In this condition, letters are assigned a spatial location and unique tone in alphabetic order. '
#                              'Additionally, the rate in which letters are presented differs for each of the three '
#                              'spatial locations. You can use the unique speaker, location, tone, and presentation rate to help you listen for each occurrence of'
#                              ' the target letter. Push "{}" to start.'.format(cont_btn_label))

instr['s0_start_trial_1'] = ('In this condition, letters are assigned a spatial location in a random order. '
          'Throughout the trial, the ordering of the letters will remain constant.  You can use the unique speaker and location to help you listen for each occurrence of'
                             ' the target letter in this condition. Push "{}" to start.'.format(cont_btn_label))

button_keys['s0_start_trial_1'] = [cont_btn]

instr['s0_start_trial_2'] = ('In this condition, letters are assigned a spatial location in a random order. '
          'Throughout the trial, the ordering of the letters will change meaning you will not know when the target letter will occur.'
          ' You can use the unique speaker and location to help you listen for each occurrence of'
                             ' the target letter in this condition. Push "{}" to start.'.format(cont_btn_label))

button_keys['s0_start_trial_2'] = [cont_btn]

# instr['s0_start_trial_3'] = ('In this condition, letters are assigned a spatial location and unique tone in alphabetic order. '
#                              'The targets in this condition are always one of \'B\', \'C\', \'D\', \'E\', \'G\', \'P\', \'T\','
#                              ' \'V\', or \'Z\'. You can use the unique speaker, location, and tone to help you listen for each occurrence of'
#                              ' the target letter. Push "{}" to start.'.format(cont_btn_label))

#button_keys['s0_start_trial_3'] = [cont_btn]

#instr['s0_start_trial_4'] = ('In this condition, letters are assigned a spatial location in alphabetic order but not their own'
                             #' unique tone. In other words, after each cycle of letters, the ordering will change. Therefore,'
                             #' you can use the unique speaker and location but not the tone to help you listen for each occurrence of'
                             #' the target letter. Push "{}" to start.'.format(cont_btn_label))

#button_keys['s0_start_trial_4'] = [cont_btn]

#instr['s0_start_trial_5'] = ('In this condition, letters are assigned a spatial location and unique tone in alphabetic order.'
                             #' Each letter is assigned a unique tone but these tones are assigned randomly, '
                             #'so you can get not rely on the any particular pattern of tones to help guide you. '
                             #'You can use the unique speaker, location, and the unique tone to help you listen for each occurrence of'
                             #' the target letter. Push "{}" to start.'.format(cont_btn_label))

#button_keys['s0_start_trial_5'] = [cont_btn]

#instr['s0_start_trial_6'] = ('In this condition, letters are assigned a spatial location and unique tone in alphabetic order. '
                             #'In this condition the loudness of each speaker is oscillating at the different rates'
                             #' This means you can use the unique speaker, location, tone,  and the differences in oscillation'
                             #' rate to help you listen for each occurrence of the target letter. '
                             #'Push "{}" to start.'.format(cont_btn_label))

#button_keys['s0_start_trial_6'] = [cont_btn]

#instr['s0_start_trial_7'] = ('In this condition, letters are assigned a spatial location and unique tone in alphabetic order. '
                             #'The loudness of each speaker will oscillate at the same rate'
                             #' although the phase of each speaker will be unique. This means you can use the unique speaker, '
                             #' location, tone,  and the differences in loudness to help you listen for each occurrence of'
                             #' the target letter. Push "{}" to start.'.format(cont_btn_label))

#button_keys['s0_start_trial_7'] = [cont_btn]

instr['s0_end_trial'] = (
    'Good work! You\'re done with this trial.'.format(cont_btn_label))

button_keys['s0_end_trial'] = []

instr['s0_end_block'] = (
    'You\'re done with the only block for this section.'.format(cont_btn_label))

button_keys['s0_end_block'] = []

instr['s0_end_sect'] = ('Great! You\'re done with the first section of the '
                        'experiment. Take a break (you can leave the booth). Press "{}" when you\'re ready to resume.'.format(cont_btn_label))

button_keys['s0_end_sect'] = [cont_btn]

# SECTION 2 INSTRUCTIONS
instr['s1_start_sect'] = ('You\'re about to begin the second section of'
							'the experiment. There are 27 trials in each'
                          ' block and 3 blocks in this section, with breaks in between blocks. Just as'
                          ' in the training section, at the end of each'
						  ' trial the fixation dot will change yellow'
						  ' prompting you for the number of targets in'
						  ' the trial.  Unlike the training however, there'
						  ' is no feedback as to whether your responses'
						  ' were correct. In this section, a new trial of a different condition'
                          ' will start automatically shortly after the previous trial ends. Note that in this section the different conditions'
                          ' are interspersed and there is no longer a description before each trial. This means that you must orient yourself to the particular condition by either the primer or'
                          ' by listening to the unique cues of the particular trial. When you\'re ready to begin, press "{}"'.format(cont_btn_label))

button_keys['s1_start_sect'] = [cont_btn]

instr['s1_start_block_0'] = (
    'You\'re about to begin the first block in this section.  Here we go!'.format(cont_btn_label))

button_keys['s1_start_block_0'] = []

instr['s1_start_trial_0'] = ('')

instr['s1_start_trial_1'] = ('')

instr['s1_start_trial_2'] = ('')

instr['s1_start_trial_3'] = ('')

instr['s1_start_trial_4'] = ('')

instr['s1_start_trial_5'] = ('')

instr['s1_start_trial_6'] = ('')

instr['s1_start_trial_7'] = ('')

instr['s1_start_trial_8'] = ('')

button_keys['s1_start_trial_0'] = []

instr['s1_start_block_1'] = ('You\'re about to begin the second block in this section of '
                             'the experiment. Reminder: there is no feedback, and new '
                             'trials start automatically shortly after the previous '
                             'trial ends. There are 27 trials in this block. Press "{}" to begin.'.format(cont_btn_label))

button_keys['s1_start_block_1'] = [cont_btn]

instr['s1_start_block_2'] = ('You\'re about to begin the last block in this section of '
                             'the experiment. Reminder: there is no feedback, and new '
                             'trials start automatically shortly after the previous '
                             'trial ends. There are 27 trials in this block. Press "{}" to begin.'.format(cont_btn_label))

button_keys['s1_start_block_2'] = [cont_btn]

instr['s1_end_trial'] = ('')

button_keys['s1_end_trial'] = []

instr['s1_end_block'] = ('Good job! Take a break if you like, then press "{}" when '
                         'you\'re ready for the next block.'.format(cont_btn_label))

button_keys['s1_end_block'] = [cont_btn]

instr['s1_end_sect'] = ('Good work! You\'re done with the second section of the '
                        'experiment. Take a break (you can leave the booth if you '
                        'need to). Press "{}" when you\'re ready to resume.'.format(cont_btn_label))

button_keys['s1_end_block'] = [cont_btn]

# SECTION 3 COGNTIVE LOAD ASSESSMENT INSTRUCTIONS
instr['s2_start_sect'] = ('You\'re about to begin the last section of the experiment. Reminder: just as'
                          ' in the introduction section, there is no feedback and your task for each trial is to listen for every occurrence of your target letter for the duration of'
                          ' the trial. In this section you will be presented with 1'
                          ' trial from each condition type then asked to answer several questions about the relative'
                          ' difficulty of each condition.'
                          ' We are interested in your experience with the experiment. In the most general sense, we'
                          ' are examining the \'workload\' you experienced. The factors that influence you experience of'
                          ' workload may come from the task itself, your feelings about your own performance, how much effort'
                          ' you put in, or any frustration you felt. Because workload may be caused by any of these'
                          ' factors, we would like you to evaluate several of them.'
                          ' Press "{}" to continue.'.format(cont_btn_label))

button_keys['s2_start_sect'] = [cont_btn]

instr['s2_start_block_0'] = ('After performing the task for each condition,'
                             ' you will be asked to rate your experience from 0 to 9 for several factors. Please consider each rating'
                             ' scale and condition type separately when answering the questions. '
                             'Please read the descriptions of the set of six scales carefully. '
                             'If you have a question about any of the scales, please don\'t hesitate to ask someone outside the booth. Press "{}" to continue.'.format(cont_btn_label))

button_keys['s2_start_block_0'] = [cont_btn]

# trial instr taken from section 1
#continue automatically
instr['s2_end_trial'] = ('Thank you for your responses.')

button_keys['s2_end_trial'] = []

instr['s2_end_block'] = ('Thank you for your responses. Take a break if you like, then press "{}" when '
                         'you\'re ready for the next block.'.format(cont_btn_label))

button_keys['s2_end_block'] = [cont_btn]

instr['s2_end_sect'] = ('All done! Thank you very much for participating!')

button_keys['s2_end_block'] = []

gen_survey = dict()
gen_survey[0] = ('How mentally demanding was the task? How much mental and perceptual activity was required (e.g.'
                 'thinking, deciding, remembering, looking, searching)? Was the task easy or demanding, simple or '
                 'complex, exacting or forgiving? Enter a digit value from 0 to 9 on the keyboard with 9 being heavily demanding.'.format(cont_btn_label))

gen_survey[1] = ('How much physical activity was required (e.g. pushing, pulling, turning, controlling, activating)?'
                 ' Was the task easy or demanding, slow or brisk, slack or strenuous, restful or laborious? Enter a digit value from 0 to 9 on the keyboard with 9 being heavily demanding.'.format(cont_btn_label))

gen_survey[2] = ('How much time pressure did you feel to keep up with the rate or pace at which the task or task elements'
                 ' occurred? Was the pace slow and leisurely or rapid and frantic? Enter a digit value from 0 to 9 on the keyboard with 9 being heavily pressured by time.'.format(cont_btn_label))

gen_survey[3] = ('How successful do you think you were in accomplishing the goals of the task set by the'
                 ' experimenter (or yourself)? How satisfied were you with your performance in accomplishing'
                 ' these goals?  Enter a digit value from 0 to 9 on the keyboard with 9 being highly successful.'.format(cont_btn_label))

gen_survey[4] = ('How hard did you have to work overall (both mentally and physically) to accomplish your level of performance?'
                 ' Enter a digit value from 0 to 9 on the keyboard with 9 being extremely hard.'.format(cont_btn_label))

gen_survey[5] = ('How insecure, discouraged, irritated, stressed and annoyed versus secure, gratified, content, '
                 'relaxed and complacent did you feel during the task? Enter a digit value from 0 to 9 on the keyboard with 9 being highly relaxed.'.format(cont_btn_label))

gen_survey['ValueError'] = 'Please enter a single digit value on the keyboard'

mid_survey = dict()
mid_survey[0] = ('Throughout this this the rating scales are used to assess your experiences in the different task '
                 'conditions. People interpret these scales in individual ways. For example, some people feel that '
                 'mental or temporal demands are the essential aspects of workload regardless of the effort they '
                 'expended on a given task or the level of performance they achieved. '
                 'The evaluation you are about to perform is a technique to assess the relative importance of six factors '
                 'in determining how much workload you experienced. Press "{}" to continue.'.format(cont_btn_label))

mid_survey[1] = ('Now you will be presented with a series of pairs of rating scale titles (for '
                 'example Effort versus Mental Demands) and asked to choose which of the items was more '
                 'important to your experience of workload in the task that you just performed. '
                 'Indicate the Scale Title that represents the more important contributor to workload for the specific '
                 'task you performed in the experiment. '
                 'Please consider your choices carefully and make them consistent with how you used the rating scales '
                 'during the particular task you were asked to evaluate. There is no correct pattern -- we '
                 'are only interested in your opinion. Press "{}" to continue.'.format(cont_btn_label))

rel_survey = dict()
rel_survey[
    0] = '1. Physical Demand or 2. Temporal Demand. Press the key corresponding to the aspect that you consider the larger factor contributing to the workload.'
rel_survey[
    1] = '1. Effort or 2. Mental Demand. Press the key corresponding to the aspect that you consider the larger factor contributing to the workload.'
rel_survey[
    2] = '1. Frustration or 2. Physical Demand. Press the key corresponding to the aspect that you consider the larger factor contributing to the workload.'
rel_survey[
    3] = '1. Effort or 2. Frustration. Press the key corresponding to the aspect that you consider the larger factor contributing to the workload.'
rel_survey[
    4] = '1. Mental Demand or 2. Temporal Demand. Press the key corresponding to the aspect that you consider the larger factor contributing to the workload.'
rel_survey[
    5] = '1. Physical Demand or 2. Effort. Press the key corresponding to the aspect that you consider the larger factor contributing to the workload.'
rel_survey[
    6] = '1. Temporal Demand or 2. Performance. Press the key corresponding to the aspect that you consider the larger factor contributing to the workload.'
rel_survey[
    7] = '1. Frustration or 2. Mental Demand. Press the key corresponding to the aspect that you consider the larger factor contributing to the workload.'
rel_survey[
    8] = '1. Temporal Demand or 2. Frustration. Press the key corresponding to the aspect that you consider the larger factor contributing to the workload.'
rel_survey[
    9] = '1. Performance or 2. Effort. Press the key corresponding to the aspect that you consider the larger factor contributing to the workload.'
rel_survey[
    10] = '1. Effort or 2. Temporal Demand. Press the key corresponding to the aspect that you consider the larger factor contributing to the workload.'
rel_survey[
    11] = '1. Frustration or 2. Performance. Press the key corresponding to the aspect that you consider the larger factor contributing to the workload.'
rel_survey[
    12] = '1. Performance or 2. Physical Demand. Press the key corresponding to the aspect that you consider the larger factor contributing to the workload.'
rel_survey[
    13] = '1. Mental Demand or 2. Performance. Press the key corresponding to the aspect that you consider the larger factor contributing to the workload.'
rel_survey[
    14] = '1. Mental Demand or 2. Physical Demand. Press the key corresponding to the aspect that you consider the larger factor contributing to the workload.'

# ASSIGN BUTTON_KEYS AND WAIT TIMES BY GENERAL PATTERN
template = dict.fromkeys(instr)
#override template with explicit values so far
button_keys = dict(template.items() + button_keys.items())
for key in dict.keys(instr):
    value = instr.get(key)
    if (value == ('')):
        button_keys[key] = []

wait_keys = dict.fromkeys(button_keys)
for key in dict.keys(wait_keys):
    value = button_keys.get(key)
    if (value == []):
        wait_keys[key] = 2
    else:
        wait_keys[key] = np.inf
