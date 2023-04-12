import matplotlib.pyplot as plt
import numpy as np

class Fig():
    """
    Class for creating and saving a figure with multiple plots.
    """
    def __init__(self, path, n_plots, figsize=(12,12)):
        """
        Initializes the figure with specified number of plots.

        Parameters:
        path (str): path and filename to save the figure as a pdf.
        n_plots (int): number of plots to add to the figure.
        figsize (tuple, optional): size of the figure. Default is (12,12).

        Attributes:
        path (str): path and filename to save the figure as a pdf.
        fig (Figure): a figure object of size figsize.
        n_ax (int): number of subplots in the figure.
        current_ax (int): index of the current axis to plot on.
        color_bm (str): color of the plot for the benchmark data.
        color_ref (str): color of the plot for the reference data.
        lw (int): linewidth of the plot.
        """
        self.path = path
        self.fig = plt.figure(figsize=figsize)
        self.n_ax = int(np.ceil(np.sqrt(n_plots)))
        self.current_ax = 1

        self.color_bm = "red"
        self.color_ref = "black"
        self.lw = 2

    def save(self):
        """
        Saves the figure as a pdf at the specified path.
        """
        plt.savefig(self.path)
        plt.close()

    def add_plot(self, dstr, dict, ref_dstr=None):
        """
        Adds a plot to the figure.

        Parameters:
        dstr (dict): dictionary containing the data to plot.
        dict (dict): dictionary containing the plot title, xlabel, and ylabel.
        ref_dstr (dict, optional): dictionary containing the reference data to plot. Default is None.
        """
        ax = self.fig.add_subplot(self.n_ax, self.n_ax, self.current_ax)
        ax.set_title(dict["title"], fontsize=12)

        values = list(dstr.values())
        keys = list(dstr.keys())
        ax.plot(keys, values, label="bm", color=self.color_bm, linewidth=self.lw, linestyle='-')

        if ref_dstr:
            ref_values = list(ref_dstr.values())
            ref_keys = list(ref_dstr.keys())
            ax.plot(ref_keys, ref_values, label="ref", color=self.color_ref, linewidth=self.lw, linestyle='-')

        ax.set_xlabel(dict["xlabel"])
        ax.set_ylabel(dict["ylabel"])
        plt.legend()
        plt.tight_layout()

        self.current_ax += 1


    def plot(self, dstr, dict, ref_dstr=None):
        """
        Plots a single plot.

        Parameters:
        dstr (dict): dictionary containing the data to plot.
        dict (dict): dictionary containing the plot title, xlabel, ylabel, and name to save the plot as a pdf.
        ref_dstr (dict, optional): dictionary containing the reference data to plot. Default is None.
        """

        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(1, 1, 1)
        ax.set_title(dict["title"], fontsize=4)

        values = list(dstr.values())
        keys = list(dstr.keys())
        ax.plot(keys, values, label="bm", color="red", linewidth=2, linestyle='-')

        if ref_dstr:
            ref_values = list(ref_dstr.values())
            ref_keys = list(ref_dstr.keys())
            ax.plot(ref_keys, ref_values, label="ref", color="black", linewidth=2, linestyle='-')

        ax.set_xlabel(dict["xlabel"])
        ax.set_ylabel(dict["ylabel"])
        plt.legend()
        plt.tight_layout()
        plt.savefig(dict["name"] + ".pdf")