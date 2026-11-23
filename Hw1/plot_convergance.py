import numpy as np
import matplotlib.pyplot as plt


def plot_conv(files_labels, title, fname, ylabel="Best fitness", logy=True, hc_every=200):
    fig, ax = plt.subplots(figsize=(5, 3.2))
    gens_max = 0
    for path, label in files_labels:
        data = np.load(path)
        gens = np.arange(1, data.shape[1] + 1)
        gens_max = data.shape[1]
        if logy:
            mid = np.median(data, axis=0)
            lo = np.percentile(data, 25, axis=0)
            hi = np.percentile(data, 75, axis=0)
        else:
            mid = data.mean(axis=0)
            lo = mid - data.std(axis=0)
            hi = mid + data.std(axis=0)
        ax.plot(gens, mid, label=label)
        ax.fill_between(gens, lo, hi, alpha=0.2)
    for g in range(hc_every, gens_max + 1, hc_every):
        ax.axvline(g, color="k", ls=":", lw=0.8)
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel("Generations")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(fname)
    plt.close(fig)


if __name__ == "__main__":
    plot_conv([("hist_rosenbrock_30_gray_hc200.npy", "Gray GA+HC"),
               ("hist_rosenbrock_30_bin_hc200.npy", "Binary GA+HC")],
              "Rosenbrock 30D", "conv_rosenbrock.pdf")

    plot_conv([("hist_rastrigin_30_gray_hc200.npy", "GA+HC")],
              "Rastrigin 30D", "conv_rastrigin.pdf")

    plot_conv([("hist_griewangk_30_gray_hc200.npy", "GA+HC")],
              "Griewangk 30D", "conv_griewangk.pdf")

    plot_conv([("hist_michalewicz_30_gray_hc200.npy", "GA+HC")],
              "Michalewicz 30D", "conv_michalewicz.pdf", logy=False)

    plot_conv([("div_rastrigin_30_gray_hc200.npy", "GA+HC")],
              "Rastrigin 30D population diversity", "div_rastrigin_30.pdf",
              ylabel="Mean normalized Hamming distance", logy=False)