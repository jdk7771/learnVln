import matplotlib.pyplot as plt
import numpy as np
def show_heatmaps(matrices, xlabel, ylabel, titles=None, cmap = "Reds", figsize = (2.5,2.5), show_nums = 0):
    fig, ax = plt.subplots(figsize)
    im = ax.imshow(matrices, cmap)

    rows = matrices.shape(0)
    cols = matrices.shape(1)
    ax.set_xticks(np.arange(len(matrices.shape(0))))
    ax.set_yticks(np.arange(len(matrices.shape(1))))
    ax.set_xticklabels(xlabel)
    ax.set_yticklabels(ylabel)

    if show_nums:
        for i in range(len(rows)):
            for j in range(len(cols)):
                text = ax.text(j, i, f'{matrices[i, j]:.2f}',
                            ha="center", va="center", color="black")
    ax.set_title(titles)
    plt.colorbar(im)
    plt.tight_layout()
    plt.show()