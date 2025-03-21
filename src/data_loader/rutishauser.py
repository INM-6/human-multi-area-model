# -*- coding: utf-8 -*-

import os as os
from scipy.io import loadmat

__all__ = [
    "area_dict",
    "load_data",
    "get_neuron",
]

# area codes from end of README.m
area_dict = {1: 'left amygdala', 2: 'left dACC', 3: 'left hippocampus',
             4: 'left preSMA', 5: 'right amygdala', 6: 'right dACC',
             7: 'right hippocampus', 8: 'right preSMA'}


def load_data(folder, file='mfc.mat'):
    filename = os.path.join(folder, file)
    data = loadmat(filename, squeeze_me=True, mat_dtype=False,
                   chars_as_strings=True)
    data = data['data_mfc']
    return data


def get_neuron(data, neuron_id):
    data_id = data[neuron_id]
    # data structure described in README.m
    neuron = {
        'area': area_dict[data_id['cellinfo'][2]],
        'response_time': data_id['behavior']['RT'][()],
        'stim_on': data_id['ts']['stim_on'][()],
        'baseline_stim_on': data_id['ts']['baseline_stim_on'][()],
        'reply': data_id['ts']['reply'][()],
        'baseline_reply': data_id['ts']['baseline_reply'][()]
    }
    return neuron


if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt

    filename = 'experimental_data/rutishauser/spikes/'
    data = load_data(filename)
    print(f'loaded {len(data)} neurons')
    all_ids = range(len(data))

    # plot both stumulus onset and response alignment
    id = 117  # should be the same as in Fig. 2A
    neuron = get_neuron(data, id)
    response_time = neuron['response_time']
    stim_on = neuron['stim_on']
    reply = neuron['reply']
    fig, axs = plt.subplots(1, 2, figsize=(7, 2.5))
    for i, ind in enumerate(np.argsort(response_time)[::-1]):
        axs[0].plot(stim_on[ind], i*np.ones_like(stim_on[ind]), 'k.')
        axs[0].plot(response_time[ind]+neuron['baseline_stim_on'], i, 'rx')
        axs[0].axvline(neuron['baseline_stim_on'], c='r', ls='--')
        axs[1].plot(reply[ind], i*np.ones_like(reply[ind]), 'k.')
        axs[1].axvline(neuron['baseline_reply'], c='r', ls='--')
    for ax in axs:
        ax.set_xlim(0, 5)
        ax.set_ylim(0, len(response_time))
        ax.set_xlabel('time [s]')
        ax.set_ylabel('trial')
    axs[0].set_title('Alignment: Stimulus Onset')
    axs[1].set_title('Alignment: Response')
    plt.tight_layout()
    plt.show()

    # plot baseline period
    n_rows, n_cols = 4, 5
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(3*n_cols, 2*n_rows))
    rnd_ids = np.random.choice(all_ids, size=n_rows*n_cols, replace=False)
    for i, id in enumerate(rnd_ids):
        neuron = get_neuron(data, id)
        stim_on = neuron['stim_on']
        baseline = neuron['baseline_stim_on']
        ax = axs[i % n_rows, i // n_rows]
        for j, st in enumerate(stim_on):
            ax.plot(st, j*np.ones_like(st), 'k.')
        ax.set_xlim(0, baseline)
        ax.set_ylim(0, len(stim_on))
        if i % n_rows == n_rows - 1:
            ax.set_xlabel('time [s]')
        if i // n_rows == 0:
            ax.set_ylabel('trial')
        ax.set_title(f'Neuron {id}')
    plt.tight_layout()
    plt.show()

    # plot firing rate histogram
    rates = []
    CVs = []
    CV_min_spikes = 10
    for id in all_ids:
        neuron = get_neuron(data, id)
        stim_on = neuron['stim_on']
        baseline = neuron['baseline_stim_on']
        for st in stim_on:
            st = np.atleast_1d(st)
            st = st[st < baseline]
            rates.append(st.size / baseline)
            if st.size >= CV_min_spikes:
                isi = np.diff(st)
                CVs.append(np.std(isi) / np.mean(isi))
    rates = np.array(rates)
    CVs = np.array(CVs)
    max_rate = rates.max()
    bins = 0.5 * np.arange(2*max_rate + 1)
    fig, axs = plt.subplots(2, 2, figsize=(7, 5))
    axs[0, 0].hist(rates, bins=bins, color='k', log=False)
    axs[0, 1].hist(rates, bins=bins, color='k', log=True)
    axs[1, 0].hist(CVs, bins='auto', color='k', log=False)
    axs[1, 1].hist(CVs, bins='auto', color='k', log=True)
    for ax in axs[0, :]:
        ax.set_xlim(bins[0], bins[-1])
        ax.set_xlabel('rate [spks/s]')
        ax.set_ylabel('count')
    for ax in axs[1, :]:
        ax.set_xlim(0)
        ax.set_xlabel('CV')
        ax.set_ylabel('count')
    axs[0, 0].set_title('Rate histogram')
    axs[0, 1].set_title('Rate histogram (log)')
    axs[1, 0].set_title(f'CV histogram (>{CV_min_spikes} spikes)')
    axs[1, 1].set_title(f'CV histogram (>{CV_min_spikes} spikes, log)')
    plt.tight_layout()
    plt.show()
