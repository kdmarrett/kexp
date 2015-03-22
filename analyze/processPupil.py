# author: Karl Marrett
# processes pupillometry data
# modified from pupil_deconv.py on voc_meg experiment by Eric Larson
# To do remove all references to MEG data (esp. in plotting examples)

import os
import glob
from os import path as op
import numpy as np
import time
import scipy
import pdb

from score import trans as event_dict
# assumed to be part of meg analysis
# from mne import read_events
from pyeparse import Raw, Epochs
from pyeparse.utils import pupil_kernel
from expyfun.io import read_hdf5, write_hdf5  # , read_tab
from expyfun import binary_to_decimals  # ,decimals_to_binary

subjects = ['Karl']
data_dir = op.join(os.pardir, 'Data')
tmin = -.5
tmax = 50  # needs to be assigned from trial length
peak = .512  # ??
sdec_dur = 2.0
n_jobs = 6
ainds = [0, 1, 2, 3, 4, 5, 8, 9, 10, 11, 12]
fs = 1000.0  # ?
session = 3
times = None
ev_nums = None
fits = list()
zscores = list()

# READ IN PARTICPANT SESSION VARIABLES FROM MAT FILE
# Reads in 'condition_bin', 'wheel_matrix_info', 'preblock_prime_sec'
for subj in subjects:
	final_data_dir = op.join(data_dir, 'Params', subj, str(session))
	global_vars = scipy.io.loadmat(op.join(final_data_dir, 'global_vars.mat'))
	condition_uni = global_vars['condition_bin']  # Unicode by default
	condition_asc = []  # ASCII
	for i in range(len(condition_uni)):
		condition_asc.append(condition_uni[i].encode('ascii'))
	condition_nums = len(condition_asc)
	wheel_matrix_info = global_vars['wheel_matrix_info'][0]
	# keep track of which new wav file to use
	wav_indices = dict(
	zip(condition_asc, np.zeros(len(condition_asc), dtype=int)))
	preblock = global_vars['preblock_prime_sec'][0]
	t0 = time.time()
	print('  Subject %s...' % subj)

	fnames = sorted(glob.glob(op.join(data_dir, '%s_*' % subj, '*.edf')))
	# assert len(fnames) == len(params['block_trials'])

	# subj_tab = glob.glob(op.join(data_dir, '%s_*.tab' % subj))
	# assert len(subj_tab) == 1
	# subj_tab = read_tab(subj_tab[0])
	# time_vecs = [s['play'][0][1] for s in subj_tab]

	raws = list()
	events = list()
        pdb.set_trace()
	for ri, fname in enumerate(fnames):
		print "fname:"
		print fname
		raw = Raw(fname)
		assert raw.info['sfreq'] == fs
		raw.remove_blink_artifacts()
		raws.append(raw)
		# cond_mat = params['cond_mat'][params['block_trials'][ri]]
		event = raw.find_events('SYNCTIME', 1)
		ttls = [np.array([int(mm) for mm in m[1].decode().split(' ')[1:]])
				for m in raw.discrete['messages']
				if m[1].decode().startswith('TRIALID')]
		# Add similar assertions to check MATLAB trial parameters with
		#	 parameters in this script
		# assert len(ttls) == len(event) == len(cond_mat)
		# conds = [binary_to_decimals(t, [1, 1, 1, 3]) - adj for t in ttls]
		# assert np.array_equal(cond_mat, conds)
		# convert event numbers
		# ev_num = read_events(op.join(meg_dir, 'lists',
									 # 'ALL_eric_voc_%s_%02g-eve.lst'
									 # % (subj, ri + 1)))
		# event[:, 1] = ev_num[:, 2]
		events.append(event)

	# if ev_nums is None:
	#   ev_nums = np.concatenate(events, axis=0)[:, 1]
	# assert np.array_equal(ev_nums, np.concatenate(events, axis=0)[:, 1])
	pdb.set_trace()
	print('    Epoching...')
# where is event_dict defined?
	epochs=Epochs(raws, events, event_dict, tmin, tmax)
	print('    Deconvolving...')
	kernel=pupil_kernel(epochs.info['sfreq'], t_max=peak, dur=dec_dur)
	fit, these_times=epochs.deconvolve(kernel=kernel, n_jobs=n_jobs)
	zscore=epochs.pupil_zscores()
	if times is None:
		times=these_times
	assert np.array_equal(times, these_times)
	fits.append(fit)
	zscores.append(zscore)
	print('  Done: %s.' % round(time.time() - t0, 1))

fits=np.array(fits)
zscores=np.array(zscores)
write_hdf5(op.join(data_dir, 'fits.hdf5'),
	   dict(fs = fs, fits = fits, zscores = zscores, times = times,
		kernel=kernel, subjects=subjects, ev_nums=ev_nums),
	   overwrite=True)

###############################################################################
# Stats and plotting

# from mne.stats import spatio_temporal_cluster_1samp_test, ttest_1samp_no_p
# from scipy import stats
# from functools import partial
# import matplotlib.pyplot as mpl
# mpl.ion()
# mpl.rcParams.update({'font.size': 10, 'mathtext.default': 'regular',
#                      'mathtext.fontset': 'stix',
#                      'font.sans-serif': ['Open Sans'], 'pdf.fonttype': 42})


# def box_off(ax):
#     ax.get_xaxis().tick_bottom()
#     ax.get_yaxis().tick_left()
#     ax.spines['right'].set_color('none')
#     ax.spines['top'].set_color('none')

# stat_fun = partial(ttest_1samp_no_p, sigma=1e-3)

# colors = [[(0.0, 0.5, 1.0), (0.2, 0.0, 0.8)],  # 200/600 gap
# [(0.0, 0.0, 0.0), (1.0, 0.0, 0.439)],  # m/s
# [(1.0, 0.3, 0.0), (0.0, 0.4, 0.3)],  # 10/20 bands
#           ]

# fig_size = [4.0, 4.0]
# stim_times = [0, 0.5, 1.6, 2.1, 2.6, 3.7, 4.2, 4.7]
# switch_time = 2.6
# switch_dur = 0.6
# stim_dur = 0.5
# dirs = [1, 1, 1]
# plot_types = ['Responses', 'Deconvolved']
# y_labels = ['Z-scores', 'Weights (AU)']

# data = read_hdf5(op.join(data_dir, 'fits.hdf5'))
# fs, fits, zscores = data['fs'], data['fits'], data['zscores']
# times, subjects = data['times'], data['subjects']
# ev_nums, kernel = data['ev_nums'], data['kernel']
# ztimes = np.arange(zscores.shape[-1]) / fs + tmin

# axs = [1, 2]
# contrasts = ['Attention', 'Vocoder']
# labels = [['Maintain', 'Switch'],
#           ['10 band', '20 band']]
# assert len(axs) == len(contrasts) == len(labels)

# reshape fits and zscores to be subj x trial x M/S x 10/20 x time
# idx = np.where(ev_nums < 50)[0]
# idx = idx[np.argsort(ev_nums[idx])]
# assert len(np.unique([(ev_nums[idx] == x).sum()
#                       for x in np.unique(ev_nums[idx])])) == 1
# fits = fits[sinds][:, idx]
# zscores = zscores[sinds][:, idx]
# shape = (fits.shape[0], 2, 2, len(idx) // 4)  # M/S x 10/20
# fits.shape = shape + (fits.shape[-1],)
# zscores.shape = shape + (zscores.shape[-1],)
# fits = np.rollaxis(fits, 3, 1)
# zscores = np.rollaxis(zscores, 3, 1)

# fig = mpl.figure(1, figsize=fig_size, facecolor='w')
# xlim = [np.maximum(times.min(), ztimes.min()),
#         np.minimum(times.max(), ztimes.max())]
# for ci, (t, data) in enumerate(zip([ztimes, times], [zscores, fits])):
#     ymax = np.ceil(np.max(np.mean(np.nanmean(data, axis=1), axis=0)))
#     ylim = [-0.2 * ymax, ymax]
#     stim_ylim = [0.65 * ylim[0], 0.35 * ylim[0]]
#     sig_lims = [ylim[0], ylim[1]]
#     for ii, (cont, ax) in enumerate(zip(contrasts, axs)):
# fit = np.nanmean(data, axis=1)  # eliminate trials axis
#         for aa in np.flipud(np.setdiff1d(np.arange(1, 3), [ax])):
#             fit = np.nanmean(fit, axis=aa)
#         ax_ = mpl.subplot(2, 2, 2 * ii + ci + 1)
#         ax_.plot(t[[0, -1]], [0, 0], linestyle=':', color='k')
#         this_fit = np.mean(fit, axis=0)

#         """
# do stats
#         X = (fit[:, 1, :] - fit[:, 0, :])[:, :, np.newaxis]
#         n_iter = np.inf
#         thresh = -stats.distributions.t.ppf(0.05 / 2, len(X) - 1)
#         out = spatio_temporal_cluster_1samp_test(X, threshold=thresh,
#                                                  stat_fun=stat_fun,
#                                                  n_jobs=6,
#                                                  buffer_size=None,
#                                                  n_permutations=n_iter)
#         T_obs, clusters, cluster_pv, H0 = out
#         good = np.where(np.array([p <= 0.05 for p in cluster_pv]))[0]
#         cluster_pv = cluster_pv[good]
#         clusters = [clusters[g] for g in good]
#         for clu, pv in zip(clusters, cluster_pv):
#             idx = (np.sign(T_obs[clu[0][0], 0]).astype(int) + 1) // 2
#             clu = clu[0]
#             ys = np.ones_like(ht[clu])
#             ymin = sig_lims[0] * ys
# ymax = sig_lims[1] * ys
# ymin = this_fit[0][clu]
#             tidx = int(np.mean(clu[[0, -1]]))
# p_y = 0.05 * ylim[1]  # np.max(this_fit[:, tidx:clu[-1]])
#             ymax = np.max(this_fit[:, clu], axis=0)
#             mpl.fill_between(t[clu], ymin, ymax, alpha=0.25,
#                              facecolor=colors[ii][idx], linewidth=0)
#             mpl.text(t[tidx], p_y, 'p = %0.3f' % pv,
#                      horizontalalignment='center',
#                      verticalalignment='baseline')
#         """
#         hs = [None] * 2
#         for jj in [0, 1]:
#             ax_.plot(t, fit[:, jj, :].T, color=colors[ii][jj], alpha=0.2)
#         for jj in [0, 1]:
#             hs[jj] = ax_.plot(t, this_fit[jj], color=colors[ii][jj],
#                               label=labels[ii])[0]
#         for st in stim_times:
#             ax_.fill_between([st, st + stim_dur],
#                              stim_ylim[0] * np.ones(2),
#                              stim_ylim[1] * np.ones(2), facecolor='w',
#                              color='k', linewidth=1)
# switch
#         ax_.fill_between([switch_time, switch_time + switch_dur],
#                          stim_ylim[0] * np.ones(2),
#                          stim_ylim[1] * np.ones(2),
#                          facecolor='w', color='k', linewidth=1)
# plot properties
#         ax_.set_xlim(xlim)
#         ax_.set_ylim(ylim)
#         ax_.legend(hs[::dirs[ii]], labels[ii][::dirs[ii]], loc=2,
#                    fontsize=8, labelspacing=0.25, handlelength=1.5,
#                    handletextpad=0.05, title=None, frameon=False)
#         box_off(ax_)
#         ax_.set_ylabel(y_labels[ci])
#         if ii == 0:
#             ax_.set_title(plot_types[ci])
#         if ii == len(axs):
#             ax_.set_xlabel('Time (sec)')
# mpl.tight_layout(pad=0.5, h_pad=2.5, w_pad=1.5)
# mpl.draw()
# if save_fig:
#     fig.savefig(op.join(data_dir, 'result.pdf'), dpi=600)
		


