from astropy.timeseries import LombScargle
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt

from utils import avg_unique, load_data, load_storedkey, plot_ls_periodogram


def extract_watermark(susp_file, stroedkey, fig_file, column_range, probabilities=None, fmax=60, span=8, saveFig=False):
    """
    Given a susp_file and a storedkey, extract watermark from it, output result
    :param susp_file: the path to suspicious file
    :param dataset: 'covtype'
    :param stroedkey: key info, dict
    :param fig_file: the path to the figure of lomb scargle periodogram
    :param probabilities: probabilities = [0.001, 0.003, 0.05, 0.01, 0.1, 0.2]
    :param fmax: maximum frequency, fmax = 60, default
    :param span: zoomed-in figure span
    :param saveFig: whether save ls periodogram
    :return: extract result (the probability)
    """

    # 1. load suspicious dataset
    feat_numeric = load_data(susp_file, column_range)
    feat_numeric = feat_numeric.astype(float)

    # 2. load stored key
    key_info = load_storedkey(stroedkey)

    # 3. projection
    time_seq_x = np.dot(feat_numeric, np.transpose(np.array(key_info['p_vec_x'])))
    time_seq_y = np.dot(feat_numeric, np.transpose(np.array(key_info['p_vec_y'])))
    time_seq = np.vstack((time_seq_x, time_seq_y)).T
    # 4. sort
    time_seq = time_seq[np.argsort(time_seq[:, 0])]
    # 5. binning
    bin_vals = np.floor(time_seq[:, 0] / key_info['bin_width'])
    bin_vals_dict = dict(Counter(bin_vals))
    num_bins = len(bin_vals_dict)
    # print('===== num of bins', num_bins)

    # == plot start
    # bins = list(bin_vals_dict.keys())
    # frequencies = list(bin_vals_dict.values())
    #
    # # Plotting the bar figure
    # plt.bar(bins, frequencies)
    # plt.xlabel('Bins')
    # plt.ylabel('Frequency')
    # plt.title(f'Histogram of Bins, num {num_bins}')
    #
    # figname = susp_file.rsplit('.', 1)[0].split('/')[-1]
    # plt.savefig(f"")
    # plt.close()
    # == plot end

    time_seq[:, 0] = np.floor(time_seq[:, 0] / key_info['bin_width']) / key_info['resolution']
    # 6. average unique
    time_seq_prime = avg_unique(time_seq)

    # 7. compute the Lomb-Scargle periodogram
    x = time_seq_prime[:, 0]
    y = time_seq_prime[:, 1]
    if any(x < 0):
        x = x - np.min(x)  # shift x so that it starts at zero

    # 7.1 ls
    ls = LombScargle(x, y, nterms=1, normalization='psd')

    # 7.2 compute the required peak height to attain any given false alarm probability
    if probabilities is None:
        probabilities = [0.001, 0.003, 0.05, 0.01, 0.1, 0.2]
    peak_height_prob = ls.false_alarm_level(probabilities)
    # for i, prob in enumerate(probabilities):
    #     print('false alarm probability={:.1f}%, required peak height={:.4f}'.format(prob, peak_height_prob[i]))

    # 7.3 autopower()
    frequency, power = ls.autopower(minimum_frequency=0, maximum_frequency=fmax)
    # for ii, freq in enumerate(frequency):
    #     print('freq={:.2f}, power={:.2f}'.format(freq, power[ii]))

    # 7.4 power at theta
    power_at_theta = ls.power(key_info['theta'])
    # print('power at theta', power_at_theta)

    # 7.5 how significant is the peak that at theta?
    # significant_max = ls.false_alarm_probability(power.max(), method='baluev')
    # print('significant max', significant_max)
    false_alarm_prob_at_theta = ls.false_alarm_probability(power_at_theta, method='baluev')
    # print('false alarm probability of the peak at theta', false_alarm_prob_at_theta)

    # 7.6 output result
    result = 1 - false_alarm_prob_at_theta
    # print('result', result)

    # 7.7 plot periodogram
    if saveFig:
        plot_ls_periodogram(frequency, power, probabilities, peak_height_prob, span, key_info['theta'], fig_file)
    else:
        pass
    return result
