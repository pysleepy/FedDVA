import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import colors, colorbar
import colorsys
import torch


def get_cmap(n, name='hsv'):
    """
    reference:
    https://stackoverflow.com/questions/14720331/how-to-generate-random-colors-in-matplotlib
    :param n: number of distinct RGB color
    :param name: a standard mpl colormap name
    :return: a function that maps each index in 0, 1, ..., n-1 to a distinct RGB color
    """
    return plt.cm.get_cmap(name, n)


def rand_cmap(nlabels, color_type='bright', first_color_black=True, last_color_black=False, verbose=True):
    """
    Generate random colormap refrerence: https://github.com/delestro/rand_cmap/blob/master/rand_cmap.py
    Creates a random colormap to be used together with matplotlib. Useful for segmentation tasks
    :param nlabels: Number of labels (size of colormap)
    :param color_type: 'bright' for strong colors, 'soft' for pastel colors
    :param first_color_black: Option to use first color as black, True or False
    :param last_color_black: Option to use last color as black, True or False
    :param verbose: Prints the number of labels and shows the colormap. True or False
    :return: colormap for matplotlib
    """
    if color_type not in ('bright', 'soft'):
        print('Please choose "bright" or "soft" for color_type')
        return

    # if verbose:
        # print('Number of labels: ' + str(nlabels))

    # Generate color map for bright colors, based on hsv
    if color_type == 'bright':
        randHSVcolors = [(np.random.uniform(low=0.0, high=1),
                          np.random.uniform(low=0.2, high=1),
                          np.random.uniform(low=0.9, high=1)) for i in range(nlabels)]

        # Convert HSV list to RGB
        randRGBcolors = []
        for HSVcolor in randHSVcolors:
            randRGBcolors.append(colorsys.hsv_to_rgb(HSVcolor[0], HSVcolor[1], HSVcolor[2]))

        if first_color_black:
            randRGBcolors[0] = [0, 0, 0]

        if last_color_black:
            randRGBcolors[-1] = [0, 0, 0]

        random_colormap = LinearSegmentedColormap.from_list('new_map', randRGBcolors, N=nlabels)

    # Generate soft pastel colors, by limiting the RGB spectrum
    else:
        low = 0.6
        high = 0.95
        randRGBcolors = [(np.random.uniform(low=low, high=high),
                          np.random.uniform(low=low, high=high),
                          np.random.uniform(low=low, high=high)) for i in range(nlabels)]

        if first_color_black:
            randRGBcolors[0] = [0, 0, 0]

        if last_color_black:
            randRGBcolors[-1] = [0, 0, 0]
        random_colormap = LinearSegmentedColormap.from_list('new_map', randRGBcolors, N=nlabels)

    # Display colorbar
    if verbose:
        plt.figure()
        fig, ax = plt.subplots(1, 1, figsize=(15, 0.5))

        bounds = np.linspace(0, nlabels, nlabels + 1)
        norm = colors.BoundaryNorm(bounds, nlabels)

        cb = colorbar.ColorbarBase(ax, cmap=random_colormap, norm=norm, spacing='proportional', ticks=None,
                                   boundaries=bounds, format='%1i', orientation=u'horizontal')
    else:
        fig, ax = None, None
    return random_colormap, fig, ax


def plot_client_sample_rate(vecs, cmap):
    """
    plot percentage of samples per client.
    :param vecs: [(c_id, [n_class_0, n_class,1, ...]),...]
    :param cmap: specify a color map
    :return: None
    """
    plt.figure()
    n_clients = len(vecs)
    n_class = len(vecs[0][1])
    plt.xlim(-1, n_clients)
    plt.ylim(0, 1)
    z_id = [term[0] for term in vecs]
    z_value = np.array([term[1] / np.sum(term[1]) for term in vecs])
    tmp = np.zeros_like(z_value[:, 0])
    for cls in np.arange(n_class):
        plt.bar(z_id, z_value[:, cls], width=0.8, bottom=tmp, color=cmap(cls+1))
        tmp += z_value[:, cls]
    ax = plt.gca()
    fig = plt.gcf()

    return fig, ax
