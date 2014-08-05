# text_kexp.py
# Author: Karl Marrett
# Text dicts for kexp

cont_btn_label = 'Next'
# INSTRUCTIONS AND FEEDBACK
instr = dict()
 
# SECTION 1 (INTRODUCTION) INSTRUCTIONS
instr['s1_start_sect'] = ('In this experiment you will be listening to letters grouped into'
	  'three different spatial locations left, middle, and right. Together, the groups will span'
	  'the full alphabet with the extra voiced commands \'Read\', \'Pause\', \'Space\', and \'Delete\'. '
	  'You will also notice that other than the voiced commands, each spatial location has a distinct'
	  ' speaker for the letters.  Most trials will have letters grouped consecutively in alphabetic order, but to orient you'
	  'to the layout, each trial will begin with a primer which includes a visual representation of the spatial location of the letters.'
	  'In these primers you will notice that one of the letters is colored green and spoken, this is the'
	  'target letter for the trial.'
	  ' Your task for each trial in this experiment is to listen for every occurence of your target letter for the duration of'
	  ' the trial.  This experiment will assess pupil responses, so after the primer for each trial a small fixation dot '
	  ' will appear in the center of screen.  Fixate on the dot as quickly as possible after it appears and move your eyes as little'
	  'as possible for the remainder of the trial. Push "{}" to continue.'.format(cont_btn_label))

instr['s1_start_block'] = ('In this first section you will be introduced to the different condition types. This section contains'
		' 1 block with a trial for every condition for a total of 8 trials. Remember: ignore the letters that are colored'
		'grey, and attend to the green highlighted letter that is spoken at the beginning of each trial.'
		' There will also be a little bit of background noise to make the task more challenging.'
		'Don\'t worry if hearing the letter is difficult at first.  Just try to attend every occurrence of the target letter.' 
		'for the entire trial. Push "{}" to start.'.format(cont_btn_label))

instr['s1_start_trial_1'] = ('In this condition letters are assigned a spatial location and individual tone in alphabetic order.'
	' You can use the speaker, the location, and the tone to help you listen for each occurrence of'
	'the target letter in this condition.  Try listening to this condition now by pushing "{}" to start'.format(cont_btn_label))

instr['s1_start_trial_2'] = ('In this condition letters are assigned a spatial location and individual tone in alphabetic order.'
	' In this conditon however, the rate in which letters are presented differs for each of the three '
	'spatial locations. You can use the speaker, the location, the tone, and the presentation rate to help you listen for each occurrence of'
	'the target letter in this condition.  Try listening to this condition now by pushing "{}" to start'.format(cont_btn_label))

instr['s1_start_trial_3'] = ('In this condition letters are assigned a spatial location and individual tone in alphabetic order.'
	'The targets in this condition are always one of \'B\', \'C\', \'D\', \'E\', \'G\', \'P\', \'T\','
	'\'V\', or \'Z\'. You can use the speaker, the location, and the tone to help you listen for each occurrence of'
	'the target letter in this condition.  Try listening to this condition now by pushing "{}" to start'.format(cont_btn_label))

instr['s1_start_trial_4'] = ('In this condition letters are assigned a spatial location in alphabetic order but not their own'
	'individual tone.  In this conditon after each cycle of letters the ordering will change. This means'
	'You can use the speaker, the location, but not the tone to help you listen for each occurrence of'
	'the target letter in this condition.  Try listening to this condition now by pushing "{}" to start'.format(cont_btn_label))

instr['s1_start_trial_5'] = ('In this condition letters are assigned a spatial location and individual tone in alphabetic order.'
	' Each letter is assigned a unique tone but these tones are assigned randomly, '
	'meaning that you can get not rely on the any particular pattern of tones to help guide you.  '
	'You can use the speaker, the location, and the unique tone to help you listen for each occurrence of'
	'the target letter in this condition.  Try listening to this condition now by pushing "{}" to start'.format(cont_btn_label))

instr['s1_start_trial_6'] = ('In this condition letters are assigned a spatial location and individual tone in alphabetic order.'
	'In this condition the loudness of each speaker is oscillating at the same rate'
	' although the phase of each speaker will be unique.  This means you can use the speaker, the'
	' location, the tone,  and the differences in loudness to help you listen for each occurrence of'
	'the target letter in this condition.  Try listening to this condition now by pushing "{}" to start'.format(cont_btn_label))

instr['s1_start_trial_7'] = ('In this condition letters are assigned a spatial location and individual tone in alphabetic order.'
	'In this condition the loudness of each speaker is oscillating at the different rates'
	'This means you can use the speaker, the location, the tone,  and the differences in oscillation' 
	' rate to help you listen for each occurrence of the target letter in this condition.  Try listening'
	' to this condition now by pushing "{}" to start'.format(cont_btn_label))

instr['s1_start_trial_8'] = ('In this condition letters are assigned a spatial location and individual tone in random order.'
	'Despite the random ordering, each letter has the same tone throughout the'
	'trial and the tones are ascended in a regular ascending or descending pattern. You can'
	' use the speaker, the location, the tone,  and the speed to help you listen for each occurrence of'
	'the target letter in this condition.  Try listening to this condition now by pushing "{}" to start'.format(cont_btn_label))
 
instr['s1_end_trial'] = ('Good work! You\'re done with this trial.'.format(cont_btn_label))

instr['s1_end_block'] = ('You are also done with the entire block.'.format(cont_btn_label))

instr['s1_end_sect'] = ('Good work! You\'re done with the first section of the '
	  'experiment. Take a break (you can leave the booth if you '
	  'need to). Press "{}" when you\'re ready to resume.'.format(cont_btn_label))

# SECTION 2 INSTRUCTIONS
instr['s2_start_sect'] = ('You\'re about to begin the second section of the experiment. There are about 9 trials in each'
' block and 5 blocks in this section, with breaks in between blocks. Just as'
	' in the introduction section there is no feedback, and in this section a new trial of a different condition'
	'will start automatically shortly after the previous trial ends. Note that in this section the different condition'
	' are interspersed and there is no text between trials in a blocks.  This means that you must orient yourself to the particular condition by either the primer or'
	' by listening to the unique cues of the particular trial. '.format(cont_btn_label))

instr['s2_start_block_1'] = ('When you\'re ready to begin the first block press "{}"'.format(cont_btn_label))

instr['s2_start_block_2'] = ('You\'re about to begin the second block in this section of '
	  'the experiment. Reminder: there is no feedback, and new '
	  'trials start automatically shortly after the previous '
	  'trial ends. There are 20 trials in this block and three '
	  'words in each category. Press "{}" to begin.'.format(cont_btn_label))

instr['s2_start_block_3'] = ('You\'re about to begin the third block in this section of '
	  'the experiment. Reminder: there is no feedback, and new '
	  'trials start automatically shortly after the previous '
	  'trial ends. There are 20 trials in this block and three '
	  'words in each category. Press "{}" to begin.'.format(cont_btn_label))

instr['s2_start_block_4'] = ('You\'re about to begin the fourth block in this section of '
	  'the experiment. Reminder: there is no feedback, and new '
	  'trials start automatically shortly after the previous '
	  'trial ends. There are 20 trials in this block and three '
	  'words in each category. Press "{}" to begin.'.format(cont_btn_label))

instr['s2_start_block_5'] = ('You\'re about to begin the last block in this section of '
	  'the experiment. Reminder: there is no feedback, and new '
	  'trials start automatically shortly after the previous '
	  'trial ends. There are 20 trials in this block and three '
	  'words in each category. Press "{}" to begin.'.format(cont_btn_label))

instr['s2_end_block'] = ('Good job! Take a break if you like, then press "{}" when '
				  'you\'re ready for the next block.'.format(cont_btn_label))

instr['s2_end_sect'] = ('Good work! You\'re done with the second section of the '
	  'experiment. Take a break (you can leave the booth if you '
	  'need to). Press "{}" when you\'re ready to resume.'.format(cont_btn_label))

# SECTION 3 COGNTIVE LOAD ASSESSMENT INSTRUCTIONS
instr['s3_start_sect'] = ('You\'re about to begin the last section of the experiment. Reminder: Just as'
		' in the introduction section there is no feedback, but in this section you be presented with 1'
		' trial of the same condition then asked to answer several questions about the relative'
		'difficulty of each condition.  In this secton there are eight blocks, one for each condition out_type. '
		'We are interested in the experiences you had during the experiment. In the most general sense we'
		'are examining the \'workload\' you experienced. The factors that influence you experience of'
		'workload may come from the task itself, your feelings about your own performance, how much effort'
		'you put in, or the stress and frustration you felt. Because workload may be caused by many different'
		'factors, we would like you to evaluate several of them.'
		' Press "{}" to begin'.format(cont_btn_label))

instr['s3_start_block_1'] = ( 'After performing the task for each condition,'
		' you will be asked to rate your experience from 1 to 10 for several factors. Please consider each rating'
		' scale and condition type separately when answering the questions.'
		'Please read the descriptions of the set of six scales carefully. It is important that they be clear to you, '
		'if you have a question about any of the scales in the table, please ask the experiment facilitator about it.'.format(cont_btn_label))

instr['s3_end_trial'] = ('Thank you for your responses.') #continue automatically

instr['s3_end_block'] = ('Good job! Take a break if you like, then press "{}" when '
				  'you\'re ready for the next block.'.format(cont_btn_label)) 

instr['s3_end_sect'] = ('All done! Thank you very much for participating!')

gen_survey = dict()
gen_survey[0] = ('How mentally demanding was the task? How much mental and perceptual activity was required (e.g.'
		'thinking, deciding, remembering, looking, searching)? Was the task easy or demanding, simple or '
		'complex, exacting or forgiving?  When you are finished entering in a value, press "{}" to continue.'.format(cont_btn_label))

gen_survey[1] = ('How much physical activity was required (e.g. pushing, pulling, turning, controlling, activating)?'
		' Was the task easy or demanding, slow or brisk, slack or strenuous, restful or laborious?When'
		' you are finished entering in a value, press "{}" to continue.'.format(cont_btn_label))

gen_survey[2] = ('How much time pressure did you feel due to the rate or pace at which the task or task elements'
		' occurred? Was the pace slow and leisurely or rapid and frantic? When you are finished entering'
		' in a value, press "{}" to continue.'.format(cont_btn_label))

gen_survey[3] = ('How successful do you think you were in accomplishing the goals of the task set by the'
		' experimenter (or yourself)? How satisfied were you with your performance in accomplishing'
		' these goals?  When you are finished entering in a value, press "{}" to continue.'.format(cont_btn_label))

gen_survey[4] = ('How hard did you have to work (mentally and physically) to accomplish your level of performance?'
		' When you are finished entering in a value, press "{}" to continue.'.format(cont_btn_label))

gen_survey[5] = ('How insecure, discouraged, irritated, stressed and annoyed versus secure, gratified, content,'
		 'relaxed and complacent did you feel during the task? When you are finished entering in a value,'
		 ' press "{}" to continue.'.format(cont_btn_label))

mid_survey = dict()
mid_survey[0] = ('Throughout this experiment the rating scales are used to assess your experiences in the different task'
		'conditions. People interpret these scales in individual ways. For example, some people feel that'
		'mental or temporal demands are the essential aspects of workload regardless of the effort they'
		'expended on a given task or the level of performance they achieved.'
		'The evaluation you are about to perform is a technique to assess the relative importance of six factors'
		'in determining how much workload you experienced.'.format(cont_btn_label))

mid_survey[1] = ('Now you will be presented with a series of pairs of rating scale titles (for'
		'example Effort versus Mental Demands) and asked to choose which of the items was more'
		'important to your experience of workload in the task that you just performed.'
		'Indicate the Scale Title that represents the more important contributor to workload for the specific'
		'task you performed in the experiment.'
		'Please consider your choices carefully and make them consistent with how you used the rating scales'
		'during the particular task you were asked to evaluate. Do not think there is any correct pattern: we'
		'are only interested in your opinions.'.format(cont_btn_label))

rel_survey = dict()
rel_survey[0] = '1. Physical Demand or 2. Temporal Demand. Press the number corresponding to the aspect that you consider the larger factor contributing to the workload.'
rel_survey[1] = '1. Effort or 2. Mental Demand. Press the number corresponding to the aspect that you consider the larger factor contributing to the workload.'
rel_survey[2] = '1. Frustration or 2. Physical Demand. Press the number corresponding to the aspect that you consider the larger factor contributing to the workload.'
rel_survey[3] = '1. Effort or 2. Frustration. Press the number corresponding to the aspect that you consider the larger factor contributing to the workload.'
rel_survey[4] = '1. Mental Demand or 2. Temporal Demand. Press the number corresponding to the aspect that you consider the larger factor contributing to the workload.'
rel_survey[5] = '1. Physical Demand or 2. Effort. Press the number corresponding to the aspect that you consider the larger factor contributing to the workload.'
rel_survey[6] = '1. Temporal Demand or 2. Performance. Press the number corresponding to the aspect that you consider the larger factor contributing to the workload.'
rel_survey[7] = '1. Frustration or 2. Mental Demand. Press the number corresponding to the aspect that you consider the larger factor contributing to the workload.'
rel_survey[8] = '1. Temporal Demand or 2. Frustration. Press the number corresponding to the aspect that you consider the larger factor contributing to the workload.'
rel_survey[9] = '1. Performance or 2. Effort. Press the number corresponding to the aspect that you consider the larger factor contributing to the workload.'
rel_survey[10] = '1. Effort or 2. Temporal Demand. Press the number corresponding to the aspect that you consider the larger factor contributing to the workload.'
rel_survey[11] = '1. Frustration or 2. Performance. Press the number corresponding to the aspect that you consider the larger factor contributing to the workload.'
rel_survey[12] = '1. Performance or 2. Physical Demand. Press the number corresponding to the aspect that you consider the larger factor contributing to the workload.'
rel_survey[13] = '1. Mental Demand or 2. Performance. Press the number corresponding to the aspect that you consider the larger factor contributing to the workload.'
rel_survey[14] = '1. Mental Demand or 2. Physical Demand. Press the number corresponding to the aspect that you consider the larger factor contributing to the workload.'
