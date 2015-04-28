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
instr['bad_input'] = ('Please enter either a 1 or 2 on the keyboard for '
        'the number of targets you heard during the trial.')

instr['start_exp'] =('In this experiment you will be listening to'
        ' letters and commands grouped into '
          'different spatial locations. '
           'This experiment is split into 3 sections, each with a '
           'varying number of blocks.  '
           'Between each section and block, there will'
          ' be an opportunity to take a break and leave the booth.'
          '  Push 1 to continue.'.format(cont_btn_label))

button_keys['start_exp'] = [cont_btn]

instr['s0_start_sect'] = ( 
        'In this section, you will be'
        ' introduced to each condition type before beginning the training.'
        ' To orient you to the layout'
      ' and ordering, each trial will begin with a visual' 
      ' representation of the spatial locations of the letters. '
      'In these visual representations, you will notice that one of the letters is colored green and spoken, this is the target letter for the trial. '
      'Push 1 to continue.'.format(cont_btn_label))

button_keys['s0_start_sect'] = [cont_btn]

instr['s0_start_block_0'] = ( 
        'The visual representations orient you to the letter and it\'s '
        'location in the trial. '
      'The target letter will always stay in the same location (left,'
      ' middle, or right) during a trial regardless of '
      'the condition. '
        'Count the occurrences of this target letter during the trial. '
        'After the visual representation, a small fixation dot'
      ' will appear in the center of screen. Keep your eyes fixated on this dot for as long as it remains visible. '
         'Push 1 to start.'.format(cont_btn_label))

button_keys['s0_start_block_0'] = [cont_btn]

instr['s0_start_trial_0'] = ('Consider the condition where letters are'
        ' in alphabetic order throughout the trial. Push 1 to start.'.format(cont_btn_label))

button_keys['s0_start_trial_0'] = [cont_btn]

# instr['s0_start_trial_1'] = ('In this condition, letters are assigned a spatial location and unique tone in alphabetic order. '
#                              'Additionally, the rate in which letters are presented differs for each of the three '
#                              'spatial locations. You can use the unique speaker, location, tone, and presentation rate to help you listen for each occurrence of'
#                              ' the target letter. Push 1 to start.'.format(cont_btn_label))

instr['s0_start_trial_1'] = ('Consider the condition where letters are assigned a random order '
          'but keep the same ordering throughout the trial. Push 1 to start.'.format(cont_btn_label))

button_keys['s0_start_trial_1'] = [cont_btn]

instr['s0_start_trial_2'] = ('Consider the condition where letters are assigned a random order '
          'and change ordering throughout the trial. Push 1 to start.'.format(cont_btn_label))

button_keys['s0_start_trial_2'] = [cont_btn]

# instr['s0_start_trial_3'] = ('In this condition, letters are assigned a spatial location and unique tone in alphabetic order. '
#                              'The targets in this condition are always one of \'B\', \'C\', \'D\', \'E\', \'G\', \'P\', \'T\','
#                              ' \'V\', or \'Z\'. You can use the unique speaker, location, and tone to help you listen for each occurrence of'
#                              ' the target letter. Push 1 to start.'.format(cont_btn_label))

#button_keys['s0_start_trial_3'] = [cont_btn]

#instr['s0_start_trial_4'] = ('In this condition, letters are assigned a spatial location in alphabetic order but not their own'
                             #' unique tone. In other words, after each cycle of letters, the ordering will change. Therefore,'
                             #' you can use the unique speaker and location but not the tone to help you listen for each occurrence of'
                             #' the target letter. Push 1 to start.'.format(cont_btn_label))

#button_keys['s0_start_trial_4'] = [cont_btn]

#instr['s0_start_trial_5'] = ('In this condition, letters are assigned a spatial location and unique tone in alphabetic order.'
                             #' Each letter is assigned a unique tone but these tones are assigned randomly, '
                             #'so you can get not rely on the any particular pattern of tones to help guide you. '
                             #'You can use the unique speaker, location, and the unique tone to help you listen for each occurrence of'
                             #' the target letter. Push 1 to start.'.format(cont_btn_label))

#button_keys['s0_start_trial_5'] = [cont_btn]

#instr['s0_start_trial_6'] = ('In this condition, letters are assigned a spatial location and unique tone in alphabetic order. '
                             #'In this condition the loudness of each speaker is oscillating at the different rates'
                             #' This means you can use the unique speaker, location, tone,  and the differences in oscillation'
                             #' rate to help you listen for each occurrence of the target letter. '
                             #'Push 1 to start.'.format(cont_btn_label))

#button_keys['s0_start_trial_6'] = [cont_btn]

#instr['s0_start_trial_7'] = ('In this condition, letters are assigned a spatial location and unique tone in alphabetic order. '
                             #'The loudness of each speaker will oscillate at the same rate'
                             #' although the phase of each speaker will be unique. This means you can use the unique speaker, '
                             #' location, tone,  and the differences in loudness to help you listen for each occurrence of'
                             #' the target letter. Push 1 to start.'.format(cont_btn_label))

#button_keys['s0_start_trial_7'] = [cont_btn]

instr['s0_end_trial'] = (
    'Good work! You\'re done with this trial.'.format(cont_btn_label))

button_keys['s0_end_trial'] = []

instr['s0_end_block'] = (
    'You\'re done with the only block for this section.'.format(cont_btn_label))

button_keys['s0_end_block'] = []

# Train instructions
instr['cond_pass'] = ('Excellent!  You passed the training for this'
' condition')

instr['end_train'] = ('Congratulations! You finished all the training.'
        '  To continue, press 1.'.format(cont_btn_label))

instr['train_fail'] = ('Sorry, you did not pass the training section '
'of the experiment. Unfortunately, you can not continue with the '
'experiment but you will be paid for your time.  You may exit the booth '
'now.')

instr['start_train'] = ('This training section tests'
        ' your ability to identify the correct number of target letters in a'
        ' given trial.  For each condition you must get two consecutive'
        ' trials correct in order to continue. '
        'To continue, press 1.'.format(cont_btn_label))

instr['more_train'] = ('During each trial, fixate on the center dot for as long as it'
          ' remains visible and count the times you hear the target letter.  When the dot changes yellow,'
          ' enter the number of targets in the trial 1 or 2.'
        ' You will be given feedback of whether your response was'
        ' correct. When correct, the dot will'
        ' change to green.  If you entered the incorrect number, '
        'it will instead shift to red.  When you are ready '
        'to start press 1 to continue.'.format(cont_btn_label))

instr['s0_end_sect'] = ('Great! You\'re done with the first section of the '
                        'experiment. Take a break (you can leave the booth). Press 1 when you\'re ready to resume.'.format(cont_btn_label))

button_keys['s0_end_sect'] = [cont_btn]

# SECTION 2 INSTRUCTIONS
instr['s1_start_sect'] = ('You\'re about to begin the second section of'
                          ' the experiment. There are breaks between'
                          ' each of the 6 blocks in this section.'
                          ' Now the different conditions'
                          ' are interspersed and you will not be given' 
                          ' feedback after each trial.  This means that you must'
                          ' orient yourself to the particular condition by either the primer or'
                          ' by listening to the unique orderings of the'
                          ' particular trial. When you\'re ready to begin, press 1.'.format(cont_btn_label))

button_keys['s1_start_sect'] = [cont_btn]

instr['s1_start_block_0'] = ('You\'re about to begin the first block in this section of '
                             'the experiment. During each trial, fixate on the center dot for as long as it'
                              ' remains visible. When it changes yellow,'
                              ' enter the number of target occurrences in'
                              ' the trial 1 or 2. Press 1 to start.')

button_keys['s1_start_block_0'] = []

instr['s1_start_block_1'] = ('You\'re about to begin the second block in this section of '
                             'the experiment. During each trial, fixate on the center dot for as long as it'
                              ' remains visible. When it changes yellow,'
                              ' enter the number of target occurrences in'
                              ' the trial 1 or 2. Press 1 to start.')

button_keys['s1_start_block_1'] = [cont_btn]

instr['s1_start_block_2'] = ('You\'re about to begin the next block in this section of '
                             'the experiment. During each trial, fixate on the center dot for as long as it'
                              ' remains visible. When it changes yellow,'
                              ' enter the number of target occurrences in'
                              ' the trial 1 or 2. Press 1 to start.')

instr['s1_start_block_3'] = ('You\'re about to begin the next block in this section of '
                             'the experiment. During each trial, fixate on the center dot for as long as it'
                              ' remains visible. When it changes yellow,'
                              ' enter the number of target occurrences in'
                              ' the trial 1 or 2. Press 1 to start.')

instr['s1_start_block_4'] = ('You\'re about to begin the next block in this section of '
                             'the experiment. During each trial, fixate on the center dot for as long as it'
                              ' remains visible. When it changes yellow,'
                              ' enter the number of target occurrences in'
                              ' the trial 1 or 2. Press 1 to start.')


instr['s1_start_block_5'] = ('You\'re about to begin the last block in this section of '
                             'the experiment. During each trial, fixate on the center dot for as long as it'
                              ' remains visible. When it changes yellow,'
                              ' enter the number of target occurrences in'
                              ' the trial 1 or 2. Press 1 to start.')

button_keys['s1_start_block_2'] = [cont_btn]

instr['s1_end_trial'] = ('')

button_keys['s1_end_trial'] = []

instr['s1_end_block'] = ('Good job! Take a break if you like, then press 1 when '
                         'you\'re ready for the next block.'.format(cont_btn_label))

button_keys['s1_end_block'] = [cont_btn]

instr['s1_end_sect'] = ('Good work, you\'re almost done!'
                        ' Please exit the booth and take a break now. '
                        ' During your break, the experimenter will give'
                        ' you the final instructions for the last section.'
                        ' Press 1 when you\'re ready to resume.'.format(cont_btn_label))

button_keys['s1_end_block'] = [cont_btn]

# SECTION 3 COGNTIVE LOAD ASSESSMENT INSTRUCTIONS
instr['s2_start_sect'] = ('In this last section, you will be presented with 1'
                          ' trial from each condition type then asked'
                          ' several questions about its difficulty.'
                          ' If you haven\'t exited the booth for a'
                          ' break and received the final instructions'
                          ' from the experimenter please do so now.'
                          ' Press 1 to continue.'.format(cont_btn_label))

button_keys['s2_start_sect'] = [cont_btn]

instr['cog_orient'] = ('Now'
                         ' you will be asked to rate your experience'
                         ' from 1 to 9 for several factors. Remember to consider each rating'
                         ' scale and this condition type independently when answering the questions. '
                         'Please read the descriptions of the set of six scales carefully. '
                         'If you have a question about any of the scales, please don\'t hesitate to ask someone outside the booth. Press 1 to continue.'.format(cont_btn_label))

button_keys['s2_start_block_0'] = [cont_btn]

# trial instr taken from section 1
#continue automatically

instr['s2_end_trial'] = ('Thank you for your responses, you are done with'
        ' this condition. Take a break if you like, then press 1 when '
                         'you\'re ready for the next block.'.format(cont_btn_label))

button_keys['s2_end_trial'] = []

button_keys['s2_end_block'] = [cont_btn]

instr['s2_end_sect'] = ('All done! Thank you very much for participating!'
        ' You may exit the booth now.')

button_keys['s2_end_block'] = []

gen_survey = dict()
gen_survey[0] = ('How mentally demanding was the task? How much mental and perceptual activity was required (e.g.'
                 'thinking, deciding, remembering, looking, searching)? Was the task easy or demanding, simple or '
                 'complex, exacting or forgiving? Enter a digit value from 1 to 9 on the keyboard with 9 being heavily demanding.'.format(cont_btn_label))

gen_survey[1] = ('How much physical activity was required (e.g. pushing, pulling, turning, controlling, activating)?'
                 ' Was the task easy or demanding, slow or brisk, slack or strenuous, restful or laborious? Enter a digit value from 1 to 9 on the keyboard with 9 being heavily demanding.'.format(cont_btn_label))

gen_survey[2] = ('How much time pressure did you feel to keep up with the rate or pace at which the task or task elements'
                 ' occurred? Was the pace slow and leisurely or rapid and frantic? Enter a digit value from 1 to 9 on the keyboard with 9 being heavily pressured by time.'.format(cont_btn_label))

gen_survey[3] = ('How successful do you think you were in accomplishing the goals of the task set by the'
                 ' experimenter (or yourself)? How satisfied were you with your performance in accomplishing'
                 ' these goals?  Enter a digit value from 1 to 9 on the keyboard with 9 being highly successful.'.format(cont_btn_label))

gen_survey[4] = ('How hard did you have to work overall (both mentally and physically) to accomplish your level of performance?'
                 ' Enter a digit value from 1 to 9 on the keyboard with 9 being extremely hard.'.format(cont_btn_label))

gen_survey[5] = ('How insecure, discouraged, irritated, stressed and annoyed versus secure, gratified, content, '
                 'relaxed and complacent did you feel during the task?'
                 ' Enter a digit value from 1 to 9 on the keyboard with 9 being highly relaxed.'.format(cont_btn_label))

instr['ValueError'] = 'Please enter a single digit value on the keyboard.'

mid_survey = dict()
mid_survey[0] = ('Throughout this section, the rating scales are used to assess your experiences in the different task '
                 'conditions. People interpret these scales in individual ways. For example, some people feel that '
                 'mental or temporal demands are the essential aspects of workload regardless of the effort they '
                 'expended on a given task or the level of performance they achieved. '
                 'The evaluation you are about to perform is a technique to assess the relative importance of six factors '
                 'in determining how much workload you experienced. Press 1 to continue.'.format(cont_btn_label))

mid_survey[1] = ('Now you will be presented with a series of pairs of rating scale titles (for '
                 'example Effort versus Mental Demands) and asked to choose which of the items was more '
                 'important to your experience of workload. '
                 'Press the number corresponding to the scale title that represents the more important contributor to workload. '  
                 'Remember, you are rating the relative difficulty '
                 'of the task only for the task of the'
                 ' condition that you just heard independently of the other conditions. '
                 'Please consider your choices carefully and make them consistent with how you used the rating scales '
                 'during the particular task you were asked to evaluate. There is no correct pattern -- we '
                 'are only interested in your opinion. Press 1 to continue.'.format(cont_btn_label))

instr['cog_mid_abbrev'] = ( 'Press the number corresponding to the scale title that represents the more important contributor to workload. '  
                 'Remember, you are rating the relative difficulty '
                 'of the task only for the task of the'
                 ' condition that you just heard independently of the other conditions. '
                 'Please consider your choices carefully and make them consistent with how you used the rating scales '
                 'during the particular task you were asked to evaluate. Press 1 to continue.'.format(cont_btn_label))

rel_survey = dict()
rel_survey[0] = ' 1. Physical Demand\n2. Temporal Demand\nWhich do you consider the larger factor contributing to the workload?'
rel_survey[
    1] = ' 1. Effort\n2. Mental Demand\nWhich do you consider the larger factor contributing to the workload?'
rel_survey[
    2] = ' 1. Frustration\n2. Physical Demand\nWhich do you consider the larger factor contributing to the workload?'
rel_survey[
    3] = ' 1. Effort\n2. Frustration\nWhich do you consider the larger factor contributing to the workload?'
rel_survey[
    4] = ' 1. Mental Demand\n2. Temporal Demand\nWhich do you consider the larger factor contributing to the workload?'
rel_survey[
    5] = ' 1. Physical Demand\n2. Effort\nWhich do you consider the larger factor contributing to the workload?'
rel_survey[
    6] = ' 1. Temporal Demand\n2. Performance\nWhich do you consider the larger factor contributing to the workload?'
rel_survey[
    7] = ' 1. Frustration\n2. Mental Demand\nWhich do you consider the larger factor contributing to the workload?'
rel_survey[
    8] = ' 1. Temporal Demand\n2. Frustration\nWhich do you consider the larger factor contributing to the workload?'
rel_survey[
    9] = ' 1. Performance\n2. Effort\nWhich do you consider the larger factor contributing to the workload?'
rel_survey[
    10] = ' 1. Effort\n2. Temporal Demand\nWhich do you consider the larger factor contributing to the workload?'
rel_survey[
    11] = ' 1. Frustration\n2. Performance\nWhich do you consider the larger factor contributing to the workload?'
rel_survey[
    12] = ' 1. Performance\n2. Physical Demand\nWhich do you consider the larger factor contributing to the workload?'
rel_survey[
    13] = ' 1. Mental Demand\n2. Performance\nWhich do you consider the larger factor contributing to the workload?'
rel_survey[
    14] = ' 1. Mental Demand\n2. Physical Demand\nWhich do you consider the larger factor contributing to the workload?'

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

wait_keys['s2_end_trial'] = np.inf

