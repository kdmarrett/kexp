"""
================================
Script 'DAS-cog-load experiment'
================================

This script runs an experiment with spatially distributed word streams.
"""
# Author: Karl Marrett <kdmarret@uw.edu>

#mainDir = /Users/nancygrulke/git/
#stimDir 
datadir = 'data'

import sys
sys.path[0:0] = ('/Users/nancygrulke/git')
#sys.path.append("/Users/nancygrulke/Drop/Dropbox/LABSN/Files _ Script/PythonScripts")
#sys.path[0:0] = ('/Users/nancygrulke/git/pyglet')
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

# import scipy.io
# mat = scipy.io.loadmat('file.mat')

# EXPERIMENT PARAMETERS
cont_btn = 8
cont_btn_label = 'Next'
resp_btn = 1
min_rt = 0.1
max_rt = 1.25
pretrial_wait = 2.5
feedback_dur = 1.5
isi = 0.2
std_args = ['dasCogLoad']
std_kwargs = dict(screen_num=0, window_size=[800, 600], full_screen=True,
                  stim_db=65, noise_db=40,  # session='1', participant='foo',
                  stim_rms=0.01, check_rms=None, suppress_resamp=False,
                  output_dir=datadir, stim_fs=24414)  # 44100.0

# RANDOM NUMBER GENERATOR
rng = np.random.RandomState(0)

#mainDir = /Users/nancygrulke/git/
#stimDir = 

# CONVERT AUDIO FILES

# INSTRUCTIONS AND FEEDBACK
#instr_args = [cont_btn_label, resp_btn, streams, bnum]  # , 2 * blocks]
instr = dict()
instr['start'] = ('In this experiment you will be listening to letter streams '
                  'in different spatial locations. The letters will contain'
                  ' most of the alphabet with some letters ommited.'
                  'Most trials will have letters grouped consecutively.'
				  'Your job is to press the "{}"" button when you '
                  'hear that the target letter that does not match the target letter '
                  'spatial location where the word occurred. There will be a '
                  'little bit of background noise to make the task more '
                  'challenging. Push {} to continue.').format(resp_btn, cont_btn_label)
instr['t_one'] = ('Practice first with just one letter stream. Push "{}" when '
                  'you\'re ready to start, then push the {} button when you '
                  'hear a word that does not match the category.'
                  ''.format(cont_btn_label, resp_btn))
instr['t_two'] = ('Now there will be two word streams. Listen to BOTH streams '
                  'and press the {} button for words in EITHER stream that '
                  'don\'t match the category of the stream they occur in. '
                  'Push "{}" to start.'.format(resp_btn, cont_btn_label))
instr['t_fa_'] = ('Good job. This time there will be four streams. Ignore the '
                  'streams whose category names are grey, and attend to the '
                  'stream whose category name is green. When you hear words '
                  'in that stream that don\'t match the category, press {}. '
                  'Push "{}" to start.'.format(resp_btn, cont_btn_label))
instr['t_faa'] = ('Good job. This time, you will have to attend to streams in '
                  'two different locations at the same time, but the '
                  'category will be the same across the two locations. Push '
                  '"{}" to start, then when you hear words in either stream '
                  'that don\'t match the category, press {}.'
                  ''.format(cont_btn_label, resp_btn))
instr['t_fab'] = ('Good job. Time for the last training block. Like the last '
                  'round, there are four streams and you have to ignore the '
                  'ones whose categories are grey. Only this time, the two '
                  'green categories will be different from each other. Press '
                  '"{}" to start, then when you hear a word in either one of '
                  'the green streams that doesn\'t match the category of that '
                  'stream, press {}.'
                  ''.format(cont_btn_label, resp_btn))
instr['tpass'] = ('Good job! You passed the training. Press "{}" to move on '
                  'to the experiment.'.format(cont_btn_label))
instr['twrna'] = ('It seems like you\'re struggling with this part of the '
                  'training. To pass, you will need to get all targets '
                  'correct with no extra button presses, on two trials in a '
                  'row. If you want to keep trying, press "{}". If you want '
                  'to stop the experiment now, you can just get up and leave '
                  'the booth (you will still be paid for the time you spent '
                  'so far).'.format(cont_btn_label))
instr['twrnb'] = ('It seems like you\'re struggling with this part of the '
                  'training. To pass, you will need to do two trials in a '
                  'row with no more than one mistake in each. If you want to '
                  'keep trying, press "{}". If you want '
                  'to stop the experiment now, you can just get up and leave '
                  'the booth (you will still be paid for the time you spent '
                  'so far).'.format(cont_btn_label))
instr['tfail'] = ('You have not passed this part of the training after 20 '
                  'attempts, so unfortunately we cannot let you continue '
                  'with the experiment (please don\'t feel bad; it is a hard '
                  'task and not everyone can do it well). Thank you for '
                  'participating; you will still be paid for the time you '
                  'spent so far.')
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
instr['e_six'] = ('You\'re about to begin the last block in this half of '
                  'the experiment. Reminder: there is no feedback, and new '
                  'trials start automatically shortly after the previous '
                  'trial ends. There are 20 trials in this block and six '
                  'words in each category. Press "{}" to begin and then press '
                  '{} to respond during the '
                  'trials.'.format(cont_btn_label, resp_btn))
instr['midpt'] = ('Good work! You\'re done with the first half of the '
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

# RUN EXPERIMENT
with ef.ExperimentController(*std_args, **std_kwargs) as ec:
    # counterbalance experiment order across subjects
    if int(ec._exp_info['session']) % 2 == 0:
        order = tr_blocks + sm_blocks + md_blocks
    else:
        order = tr_blocks + md_blocks + sm_blocks

    # startup
    ec.screen_prompt(instr['start'], live_keys=[cont_btn])
    ec.start_noise()

    # run blocks
    for bnum, block in enumerate(order):
        # training passed?
        if bnum == len(tr_blocks):
            ec.screen_prompt(instr['tpass'], live_keys=[cont_btn])
        # halfway point?
        elif bnum == len(tr_blocks) + len(sm_blocks):
            ec.screen_prompt(instr['midpt'], live_keys=[cont_btn])
        # between blocks
        elif bnum > len(tr_blocks):
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
            if block in tr_blocks:
                ecid = '{}-{}-{}-{}-{}'.format('training block {} trial {}'.format(bnum, tnum),
                                               ''.join(np.char.asarray(attn[block][tnum]).ravel()),
                                               ''.join(np.char.asarray(targ_locs[block][tnum]).ravel()),
                                               ' '.join(words[block][tnum].ravel()),
                                               ' '.join(np.char.asarray(onset_sec[block][tnum]).ravel()))
            else:
                ecid = np.binary_repr(cum_trial, width=8)
            ttlid = np.array(list(np.binary_repr(cum_trial, width=8)), int)
            ec.identify_trial(ec_id=ecid, ttl_id=ttlid)

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
            rt = []   # reaction time to targets
            ft = []   # reaction time to foils
            fa = []   # false alarms - pressed too late or non-targ non-foils
            stim_dur = stim.shape[-1] / ec.stim_fs
            ec.load_buffer(stim)
            ec.wait_until(end_wait)

            # play stim
            ec.start_stimulus(flip=False)

            # handle user responses
            presses = ec.wait_for_presses(stim_dur + max_rt, min_rt,
                                          [resp_btn], True)
            ec.stop()
            p_times = [x for _, x in presses]  # has to be list for .pop()
            t_times = onset_sec[block][tnum][targ_locs[block][tnum].astype(bool)]
            f_times = onset_sec[block][tnum][foil_locs[block][tnum].astype(bool)]
           # which presses were targ hits?
            for tt in t_times:
                not_early = np.where(tt < np.array(p_times) - min_rt)[0]
                not_late = np.where(tt > np.array(p_times) - max_rt)[0]
                viable = set(not_early) & set(not_late)
                if len(viable):
                    p_index = sorted(list(viable))[0]
                    pt = p_times.pop(p_index)
                    rt.append(pt - tt)
                else:
                    rt.append(-1)
            # which presses were foils?
            for ff in f_times:
                not_early = np.where(ff < np.array(p_times) - min_rt)[0]
                not_late = np.where(ff > np.array(p_times) - max_rt)[0]
                viable = set(not_early) & set(not_late)
                if len(viable):
                    p_index = sorted(list(viable))[0]
                    pt = p_times.pop(p_index)
                    ft.append(pt - ff)
                else:
                    ft.append(-1)
            # the rest were false alarms
            fa.extend(p_times)

            # clear screen
            ec.flip()

            # if training, give feedback
            if bnum < len(tr_blocks):
                rt = np.array(rt)
                ft = np.array(ft)
                fa = np.array(fa)
                n_targs = len(t_times)
                correct = np.sum(rt > 0)
                mean_rt = np.mean(np.concatenate((rt[rt > 0], ft[ft > 0])))
                if np.isnan(mean_rt):
                    mean_rt = ''
                else:
                    mean_rt = 'Mean reaction time {:.2f} sec.'.format(mean_rt)
                f_alarm = len(fa) + len(ft[ft > 0])
                if f_alarm > 0:
                    feedback = ('{} of {} targets correct. \n'
                                '{} presses incorrect or too slow. \n{}'
                                ''.format(correct, n_targs, f_alarm, mean_rt))
                elif correct == 0:
                    feedback = ('There were {} targets but you didn\'t push '
                                'the response button.'.format(n_targs))
                else:
                    feedback = ('{} of {} targets correct. \n{}'
                                ''.format(correct, n_targs, mean_rt))
                if f_alarm > 0:
                    ec.screen_prompt(feedback, feedback_dur + 0.5)
                else:
                    ec.screen_prompt(feedback, feedback_dur)

                #ec.wait_secs(pretrial_wait)  # TODO: is this needed?

                # check if trained
                prev = curr
                if block in ('t_faa', 't_fab'):
                    # hardest conditions; missing/omitting 1 targ press is ok
                    curr = n_targs - correct < 2 and len(presses) - len(t_times) < 2
                else:
                    curr = n_targs == correct and len(presses) - len(t_times) == 0
                if all([prev, curr]):
                    prev = False
                    curr = False
                    break
            else:
                # write out data
                ec.write_data_line('target_times', t_times)
                ec.write_data_line('foil_times', f_times)
                ec.write_data_line('press_times', [x for _, x in presses])
                ec.write_data_line('target_RTs', rt)
                ec.write_data_line('foil_RTs', ft)
                ec.write_data_line('false_alarm_times', fa)
                ec.trial_ok()

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

