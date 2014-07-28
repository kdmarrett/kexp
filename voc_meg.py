# -*- coding: utf-8 -*-

from os import path as op
from expyfun import (ExperimentController, EyelinkController, visual,
                     get_keyboard_input, assert_version, decimals_to_binary)
from expyfun.io import read_hdf5, read_wav

assert_version('ca52135')
p = read_hdf5(op.join(op.dirname(__file__), 'params.hdf5'))
stim_dir = op.join(op.dirname(__file__), p['stim_dir'])
feedback = False  # Should be False for MEG

session = 'MEG' if not feedback else 'Training'


with ExperimentController('voc_exp', stim_db=65, noise_db=45, check_rms=None,
                          session=session) as ec:
    el = EyelinkController(ec)
    ec.set_visible(False)
    ec.set_background_color([0.1] * 3)
    fix = visual.FixationDot(ec, colors=p['fix_c'])

    bi = get_keyboard_input('Enter block number (0): ', 0, int)
    while 0 <= bi < len(p['block_trials']):
        # start of each block
        ec.set_visible(True)
        ec.start_noise()
        ec.write_data_line('block', bi)
        el.calibrate(prompt=False)
        fix.draw()
        ec.flip()
        ec.wait_for_presses(5.0)  # wait to settle

        # each trial
        for ti in p['block_trials'][bi]:
            # stimulus
            samples = read_wav(op.join(stim_dir, p['stim_names'][ti]))[0]
            ec.load_buffer(samples)
            id_ = decimals_to_binary(p['cond_mat'][ti], [1, 1, 1, 3])
            ec.identify_trial(ec_id=id_, ttl_id=id_, el_id=id_)
            ec.listen_presses()

            # fixation dot changes
            for ci, circs in enumerate(p['fixes'][ti]):
                color = p['flash_c'] if circs['flash'] else p['fix_c']
                fix.set_colors(color)
                fix.draw()
                if ci == 0:
                    t0 = ec.start_stimulus()
                else:
                    ec.flip(t0 + circs['onset'])

            # response
            fix.set_colors(p['resp_c'])
            fix.draw()
            ec.flip(t0 + p['resp_onset'])
            ec.stamp_triggers([2])

            # inter-trial interval
            fix.set_colors(p['fix_c'])
            fix.draw()
            ec.flip(t0 + p['resp_offset'])
            ec.wait_secs(p['inter_trial_dur'])
            presses = ec.get_presses(relative_to=t0)
            ec.stop()
            ec.trial_ok()

            # feedback
            if feedback:
                nums = [{'1': 1, '2': 2, '3': 3, 'tab': 0}.get(pp[0], '?')
                        for pp in presses]
                targ = p['cond_mat'][ti][3]
                targ = targ if targ >= 0 else p['targ_pos'][ti][0].sum()
                if len(presses) == 1 and nums[0] == targ:
                    if p['resp_onset'] <= presses[0][1] <= p['resp_offset']:
                        txt = 'Correct!'
                    else:
                        txt = ('Correct, but please respond when the '
                               'response circle is on!')
                else:
                    txt = 'You pressed %s, wanted %s' % (nums, targ)
                ec.screen_prompt(txt + '\nPress a button to continue')

        # end of each block
        el.stop()
        ec.stop_noise()
        ec.set_visible(False)
        bi = get_keyboard_input('Enter block number ({0}): '.format(bi + 1),
                                bi + 1, int)