# Copyright (c) 2014, LABS^N
# Distributed under the (new) BSD License. See LICENSE.txt for more info.
from __future__ import print_function

import os
import numpy as np
from os import path as op
import mne
from mnefun import extract_expyfun_events, get_raw_fnames

from expyfun.io import read_hdf5, write_hdf5
from expyfun import decimals_to_binary

params = read_hdf5(op.join(os.getcwd(), 'params.hdf5'))
adj = np.array([0, 0, 0, 1], int)

trans = {}
assert params['n_bands'][0] == 10  # other is 20
for ai in range(2):
    for bi in range(2):
        for ti in range(2):
            for vi in range(-1, 4):
                id_ = decimals_to_binary([ai, bi, ti, vi + 1], [1, 1, 1, 3])
                id_ = np.sum(2 ** np.arange(len(id_))[::-1]
                             * np.array(id_, bool)) + 1
                # 30/40 are maint/sw, 1/2 are 10/20 bands, 50/60 is V0/VNz
                en = (bi + 1)
                if vi < 0:
                    en += 10 * (ai + 3)
                else:
                    en += 10 * ((vi > 0) + 5)
                trans[id_] = en


def extract_expyfun_events_special(fname, subj, ri):
    # Read events while discarding the early / fake triggers :(
    raw = mne.io.Raw(fname, allow_maxshield=True)
    orig_events = mne.find_events(raw, stim_channel='STI101', shortest_event=0)
    events = list()
    for ch in range(1, 9):
        ev = mne.find_events(raw, stim_channel='STI00%d' % ch)
        # de-bouncing code -- second of each pair is legit
        if subj == 'eric_voc_019' and ch == 1:
            mask = np.concatenate((np.diff(ev[:, 0]) > 1000, [True]))
            ev = ev[mask]
        ev[:, 2] = 2 ** (ch - 1)
        events.append(ev)
    events = np.concatenate(events)
    events = events[np.argsort(events[:, 0])]
    # first run had a TTL pop (TDT issue)
    if subj == 'eric_voc_020' and ri == 0:
        events = events[4:]

    # check for the correct number of trials
    aud_idx = np.where(events[:, 2] == 1)[0]
    breaks = np.concatenate(([0], aud_idx, [len(events)]))
    resps = []
    event_nums = []
    for ti in range(len(aud_idx)):
        # pull out responses (they come *after* 1 trig)
        these = events[breaks[ti + 1]:breaks[ti + 2], 2]
        resp = these[these > 8]
        resp = np.log2(resp) - 3
        resps.append(resp)

        # look at trial coding, double-check trial type (pre-1 trig)
        these = events[breaks[ti + 0]:breaks[ti + 1], 2]
        serials = these[np.logical_and(these >= 4, these <= 8)]
        en = np.sum(2 ** np.arange(len(serials))[::-1] * (serials == 8)) + 1
        event_nums.append(en)

    these_events = events[aud_idx]
    these_events[:, 2] = event_nums
    if subj == 'eric_voc_020' and ri == 0:
        these_events[0, 2] = 21
    return these_events, resps, orig_events


def score(p, subjects):
    """Scoring function"""
    for subj in subjects:
        perf = np.zeros((2, 2, 2, 2, 2), int)  # correct/count, ai, bi, ti, vi
        print('  Running subject %s... ' % subj, end='')

        # Figure out what our filenames should be
        out_dir = op.join(p.work_dir, subj, p.list_dir)
        if not op.isdir(out_dir):
            os.mkdir(out_dir)

        fnames = get_raw_fnames(p, subj, 'raw', True)
        for ri, fname in enumerate(fnames):
            use_ri = 0 if ri >= len(params['block_trials']) else ri  # wrap ERM
            conds = params['cond_mat'][params['block_trials'][use_ri]]
            expected = [decimals_to_binary(c + adj, [1, 1, 1, 3])
                        for c in conds]
            expected = [np.sum(2 ** np.arange(len(e))[::-1]
                        * np.array(e, bool)) + 1 for e in expected]

            if subj in ('eric_voc_019', 'eric_voc_020'):
                events, presses = extract_expyfun_events_special(fname,
                                                                 subj, ri)[:2]
            else:
                events, presses = extract_expyfun_events(fname)[:2]
            assert np.array_equal(expected, events[:, 2])

            for ii in range(len(events)):
                events[ii, 2] = trans[events[ii, 2]]
            run_name = op.basename(fname)[:-8]  # cut of _raw.fif
            fname_out = op.join(out_dir, 'ALL_%s-eve.lst' % run_name)
            mne.write_events(fname_out, events)

            # Figure out presses
            if ri < len(p.run_names):  # actual run
                corrects = []
                assert len(conds) == len(presses)
                for ii, (c, press) in enumerate(zip(conds, presses)):
                    ti = params['block_trials'][ri][ii]
                    assert all(ppp in (1, 2, 3, 4) for pp in presses
                               for ppp in pp)
                    targ = c[3]
                    if c[3] < 0:
                        targ = params['targ_pos'][ti][0].sum()
                    correct = (len(press) == 1 and press[0] - 1 == targ)
                    perf[:, c[0], c[1], c[2], int(c[3] >= 0)] += [correct, 1]
                    corrects.append(correct)
        write_hdf5(op.join(p.work_dir, subj, '%s_perf.hdf5' % subj), perf,
                   overwrite=True)
        pe = (100 * perf[0, ..., 0, 0].sum() / float(perf[1, ..., 0, 0].sum()),
              100 * perf[0, ..., 1, 0].sum() / float(perf[1, ..., 1, 0].sum()),
              100 * perf[0, ..., 1].sum() / float(perf[1, ..., 1].sum()))
        print('Performance: %0.1f%% 10, %0.1f%% 20, %0.1f%% visual' % pe)
