# for testing and learning python
# fn name paradigm.wav

# Dirs of Stims
stimDir = '~/Desktop/Stims/'

# Extract variable names from .mat

# read in WAVS
# scipy or numpy load mat
# get paradigm for fn 

# What is sorted mean
for i in [0:1]
	for j in len(stim_blocks)
		stim_blocks[j] = sorted(glob(op.join(stimdir, ('block_' + str(j))))
		train_blocks[j] = sorted(glob(op.join(stimdir, 'training/', ('block_' + j))))

# Get Stims ready
stim = stim_blocks[]
ec.load_buffer(stim)
ec.wait_until(end_wait)

# play stim
ec.start_stimulus(flip=False)
start_time = ec.current_time

ec.screen_prompt(instr['start'], live_keys=[cont_btn])
ec.start_noise()

stim_blocks['t_two'] = sorted(glob(op.join(stimdir, 'train-two*')))
stim_blocks['t_fa_'] = sorted(glob(op.join(stimdir, 'train-four-a-*')))
stim_blocks['t_faa'] = sorted(glob(op.join(stimdir, 'train-four-aa*')))
stim_blocks['t_fab'] = sorted(glob(op.join(stimdir, 'train-four-ab*')))
stim_blocks['e_one'] = sorted(glob(op.join(stimdir, 'trial-0[01]*')))  # 0-19
stim_blocks['e_two'] = sorted(glob(op.join(stimdir, 'trial-0[23]*')))  # 20-39
stim_blocks['e_thr'] = sorted(glob(op.join(stimdir, 'trial-0[45]*')))  # etc
stim_blocks['e_fou'] = sorted(glob(op.join(stimdir, 'trial-0[67]*')))
stim_blocks['e_fiv'] = sorted(glob(op.join(stimdir, 'trial-0[89]*')))
stim_blocks['e_six'] = sorted(glob(op.join(stimdir, 'trial-1*')))


