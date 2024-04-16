import matplotlib.pyplot as plt
import numpy as np


def plot_ls_periodogram(frequency, power, probabilities, peak_height_prob, span, theta, fig_name):
    fig, axes = plt.subplots(1, 2, figsize=(8 * 2, 6))
    ax = axes[0]
    ax_zoom = axes[1]

    ax.plot(frequency, power, '*-', label="ls periodogram")
    ax.plot(frequency, peak_height_prob[0] * np.ones_like(frequency), label='false alarm prob {:.1f}%'.format(probabilities[0]*100))
    ax.plot(frequency, probabilities[4] * np.ones_like(frequency), label='false alarm prob {:.1f}%'.format(probabilities[4]*100))
    ax.set_xlabel('Frequency')
    ax.set_ylabel('Power')
    ax.legend()

    # zoomed-in subfigure
    idx_theta = (np.abs(frequency - theta)).argmin()
    ax_zoom.plot(frequency[idx_theta-span:idx_theta+span], power[idx_theta-span:idx_theta+span], '*-', label="zoomed-in periodogram")
    ax_zoom.plot(frequency[idx_theta-span:idx_theta+span], peak_height_prob[0] * np.ones_like(frequency[idx_theta-span:idx_theta+span]), label='{:.1f}%'.format((1-probabilities[0])*100))
    ax_zoom.plot(frequency[idx_theta-span:idx_theta+span], peak_height_prob[4] * np.ones_like(frequency[idx_theta-span:idx_theta+span]), label='{:.1f}%'.format((1-probabilities[4])*100))
    ax_zoom.set_xlabel("Frequency (zoomed-in)")
    ax_zoom.set_ylabel("Power (zoomed-in)")
    ax_zoom.legend()

    plt.savefig(fig_name)
    plt.close()

