from astropy.timeseries import LombScargle
import matplotlib.pyplot as plt


def plot_spectrum(frequency, power, save_path):
    plt.figure(figsize=(12, 10))
    plt.plot(frequency, power, label="LSP of Watermark's Signal", c='m')
    plt.xlabel('Frequency', fontsize=10)
    plt.ylabel('Power', fontsize=10)
    plt.legend(loc='upper right', fontsize=8)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')

    # Show the pie chart (if save_path is not provided)
    if not save_path:
        plt.show()


def plot_sine_signal(X_wm_hash, Y_wm, frequency, power, power_at_theta, wm_params, save_path):
    # Create a figure with two subplots in a single row
    plt.figure(figsize=(12, 4))  # Adjust the figure size as needed

    # Subplot 1 (Left)
    plt.subplot(1, 2, 1)  # 1 row, 2 columns, select the first subplot
    plt.scatter(X_wm_hash, Y_wm,
                label=r"Watermark's Signal $\lambda={}$, $\theta={}$".format(wm_params['ampt'], wm_params['theta']), marker='.',
                c='m', s=5)
    plt.xlabel('Time', fontsize=10)
    plt.ylabel('Ampt', fontsize=10)
    plt.legend(loc='upper right', fontsize=8)

    # Subplot 2 (Right)
    plt.subplot(1, 2, 2)  # 1 row, 2 columns, select the second subplot
    plt.plot(frequency, power, label="LSP of Watermark's Signal, power {:.2f}".format(power_at_theta), c='m')
    plt.xlabel('Frequency', fontsize=10)
    plt.ylabel('Power', fontsize=10)
    plt.legend(loc='upper right', fontsize=8)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')

    # Show the pie chart (if save_path is not provided)
    if not save_path:
        plt.show()


def ans_sine_signal(X_wm_hash, Y_wm, wm_params, save_path):
    # spectrum of sine signal
    ls = LombScargle(X_wm_hash, Y_wm, nterms=1, normalization='psd')
    fmax = 60
    frequency, power = ls.autopower(minimum_frequency=0, maximum_frequency=fmax)
    power_at_theta = ls.power(wm_params['theta'])
    # print(f"theta={wm_params['theta']}, power {power_at_theta}")

    # plot_spectrum(frequency, power, save_path)
    plot_sine_signal(X_wm_hash, Y_wm, frequency, power, power_at_theta, wm_params, save_path)

    return power_at_theta
