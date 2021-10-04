import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import os
import torch



def layout_fig(graph, mod=None):

    """
    function

    :param graph: number of axes to make
    :type graph: int
    :param mod: sets the number of figures per row
    :type mod: int (, optional)
    :return: fig:
                handle to figure being created
             axes:
                numpy array of axes that are created
    :rtype: fig:
                matplotlib figure
            axes:
                numpy array
    """

    # Sets the layout of graphs in matplotlib in a pretty way based on the number of plots
    if mod is None:
        # Selects the number of columns to have in the graph
        if graph < 3:
            mod = 2
        elif graph < 5:
            mod = 3
        elif graph < 10:
            mod = 4
        elif graph < 17:
            mod = 5
        elif graph < 26:
            mod = 6
        elif graph < 37:
            mod = 7

    # builds the figure based on the number of graphs and selected number of columns
    fig, axes = plt.subplots(graph // mod + (graph % mod > 0), mod,
                             figsize=(3 * mod, 3 * (graph // mod + (graph % mod > 0))))

    # deletes extra unneeded axes
    axes = axes.reshape(-1)
    for i in range(axes.shape[0]):
        if i + 1 > graph:
            fig.delaxes(axes[i])

    return fig, axes


def embedding_maps(data, image, colorbar_shown=True,
                   c_lim=None, mod=None,
                   title=None):

    """

    :param data: data need to be showed in image format
    :type data: array
    :param image: the output shape of the image
    :type image: array
    :param colorbar_shown: whether to show the color bar on the left of image
    :type colorbar_shown: boolean
    :param c_lim: Sets the scales of colorbar
    :type c_lim: list
    :param mod: set the number of image for each line
    :type mod: int
    :param title: set the title of figure
    :type title: string
    :return: handle to figure being created
    :rtype: matplotlib figure
    """
    fig, ax = layout_fig(data.shape[1], mod)

    for i, ax in enumerate(ax):
        if i < data.shape[1]:
            im = ax.imshow(data[:, i].reshape(image.shape[0], image.shape[1]))
            ax.set_xticklabels('')
            ax.set_yticklabels('')

            # adds the colorbar
            if colorbar_shown is True:
                divider = make_axes_locatable(ax)
                cax = divider.append_axes('right', size='10%', pad=0.05)
                cbar = plt.colorbar(im, cax=cax, format='%.1e')

                # Sets the scales
                if c_lim is not None:
                    im.set_clim(c_lim)

    if title is not None:
        # Adds title to the figure
        fig.suptitle(title, fontsize=16,
                     y=1, horizontalalignment='center')

    fig.tight_layout()


def imagemap(ax, data, colorbars=True, clim=None):
    """
    Plots an image map

    Parameters
    ----------
    axis : matplotlib, object
        axis which is plotted
    data  : numpy, float
        data to plot
    clim  : numpy, float, optional
        sets the climit for the image
    color_bar  : bool, optional
        selects to plot the colorbar bar for the image
    """
    if data.ndim == 1:
        data = data.reshape(np.sqrt(data.shape[0]).astype(
            int), np.sqrt(data.shape[0]).astype(int))

    cmap = plt.get_cmap('viridis')

    if clim is None:
        im = ax.imshow(data, cmap=cmap)
    else:
        im = ax.imshow(data, clim=clim, cmap=cmap)

    ax.set_yticklabels('')
    ax.set_xticklabels('')

    if colorbars:
        # adds the colorbar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='10%', pad=0.05)
        cbar = plt.colorbar(im, cax=cax, format='%.1e')


def find_nearest(array, value, averaging_number):
    """
    returns the indices nearest to a value in an image
    Parameters
    ----------
    array : float, array
        image to find the index closest to a value
    value : float
        value to find points near
    averaging_number : int
        number of points to find
    """
    idx = (np.abs(array - value)).argsort()[0:averaging_number]
    return idx


def make_folder(folder, **kwargs):
    """
    Function that makes new folders
    Parameters
    ----------'
    folder : string
        folder where to save
    Returns
    -------
    folder : string
        folder where to save
    """

    # Makes folder
    os.makedirs(folder, exist_ok=True)

    return (folder)


def latent_generator(model,
                     embeddings,
                     image,
                     number,
                     average_number,
                     indx=None,
                     ranges=None,
                     x_values=None,
                     y_scale=[-2.2, 4],
                     device = 'cuda'):
    """
    plots the generator results

    Parameters
    ----------
    model : tensorflow object
        neural network model
    encode : float, array
        the input embedding (or output from encoder)
    voltage : float, array
        voltage array
    number : int
        number of divisions to plot
    averaging_number : int
        number of points to consider in the average
    ranges : float, array
        sets the ranges for the embeddings
    folder : string
        set the folder where to export the images
    plot_format  : dict
        sets the plot format for the images
    printing : dictionary
        contains information for printing
        'dpi': int
            resolution of exported image
        print_EPS : bool
            selects if export the EPS
        print_PNG : bool
            selects if print the PNG
    graph_layout : int, array (optional)
        sets the layout for the figure.

    """

    # sets the colormap
    cmap = plt.cm.viridis

    if indx is None:
        embedding_small = embeddings.squeeze()
    else:
        embedding_small = embeddings[:, indx].squeeze()

    # creates the figures and axes in a pretty way
    fig, ax = layout_fig(embedding_small.shape[1] * 2, mod=3)

    # plots all of the embedding maps
    for i in range(embedding_small.shape[1]):
        im = imagemap(ax[i], embedding_small[:, i].reshape(image.shape[0], image.shape[1]))

    # loops around the number of example loops
    for i in range(number):

        # loops around the number of embeddings from the range file
        for j in range(embedding_small.shape[1]):

            if ranges is None:
                value = np.linspace(np.min(embedding_small[:, j]),
                                    np.max(embedding_small[:, j]), number)
            else:
                # sets the linear spaced values
                value = np.linspace(0, ranges[j], number)

            idx = find_nearest(
                embedding_small[:, j], value[i], average_number)
            gen_value = np.mean(embeddings[idx], axis=0)
            gen_value[j] = value[i]

            # computes the generated results
            gen_value_1 = torch.from_numpy(np.atleast_2d(gen_value)).to(device)
            generated = model(gen_value_1)
            generated = generated.to('cpu')
            generated = generated.detach().numpy().squeeze()

            # plots and formats the graphs
            if x_values is None:
                ax[j + embedding_small.shape[1]
                   ].plot(generated, color=cmap((i + 1) / number))
            else:
                ax[j + embedding_small.shape[1]
                   ].plot(x_values, generated, color=cmap((i + 1) / number))

            ax[j + embedding_small.shape[1]].set_ylim(y_scale)
            #ax[j + embedding_small.shape[1]].set_yticklabels('')
            plt.tight_layout(pad=1)