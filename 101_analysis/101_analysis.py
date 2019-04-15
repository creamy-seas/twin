import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
plt.style.use('ilya_plot')


class transmission():

    def __init__(self, plot_or_not):
        self.plot_or_not = plot_or_not
        self.prepare_plot(1, 1)
        self.transmission_x = []
        self.transmission_y = []

    def prepare_plot(self, nrows, ncols):
        """
        __ Parameters __
        nrows: number of subplot rows
        ncols: number of suplot columns

        __ Description __
        Prepare figure on which plotting will be performed
        """
        # 1 - by default, nothing is plotting
        self.fig = None
        self.ax = None
        plt.ioff()
        plt.close("all")

        if(self.plot_or_not):
            print("==> 'prepare_plot' is setting up figure and axes")
            # 2 - interactive mode, to alow updating
            plt.ion()

            # 3 - define plots
            self.fig, self.ax = plt.subplots(nrows=nrows, ncols=ncols)
            try:
                # 3  - adjust position on the screen
                mngr = plt.get_current_fig_manager()
                mngr.window.setGeometry(0, 30, 1280, 1600)
                self.fig.canvas.set_window_title('Run time data')

            except AttributeError:
                pass
            print("==> 'prepare_plot' finished")

    def raise_error(self, display):
        output = "\n****************************************\n" + \
            display + "\n****************************************\n"
        print(output)
        raise ValueError(display)

    def transmission_load(self, colX, colY, convert_to_mhz):
        """
        __ Parameters __
        colX, colY: column to treat as the X,Y coordinate
        convert_to_mhz: nromalise by 10^6 or not

        __ Description __
        Loads in the file to 'self.data_transmission'
        """
        temp_load = np.loadtxt("data/extinction_22.txt").transpose()
        self.transmission_x = temp_load[colX, :]
        self.transmission_y = temp_load[colY, :]

        if(convert_to_mhz):
            self.transmission_x = self.transmission_x / 10**6

    def transmission_filter(self, xMin, xMax, yMin, yMax):
        """
        __ Parameters __
        min/max:

        __ Description __
        Remove any data points out of the ones specified
        """
        if(len(self.transmission_x) == 0):
            self.raise_error("No data loaded")

        for i in range(0, len(self.transmission_x)):
            if((self.transmission_x[i] < xMin) or (self.transmission_x[i] > xMax)):
                self.transmission_x = np.delete(self.transmission_x, i)
                self.transmission_y = np.delete(self.transmission_y, i)

        for i in range(0, len(self.transmission_y)):
            if((self.transmission_y[i] < yMin) or (self.transmission_y[i] > yMax)):
                self.transmission_x = np.delete(self.transmission_x, i)
                self.transmission_y = np.delete(self.transmission_y, i)

    def transmission_plot(self, plot_axes):
        """
        Plots loaded transmission data
        """
        if(self.plot_or_not):
            plot_axes.scatter(self.transmission_x,
                              self.transmission_y, marker='.', color="C2")
            plt.show()

    def __transmission_fit_function(self, x, Gamma1, Gamma2, Omega, offset):
        """
        __ Description __
        Fits the REAL part of transmission intesnity (which near the resonance will
        dominate over the IMAGINARY part (see major_project p.12))

        R[t]**2
        """

        x = x - offset
        return (1 - Gamma1 / (2 * Gamma2 * (1 + (x / Gamma2)**2 + Omega ** 2 / Gamma1 / Gamma2)))**2
        # return (1 - 1 / (Gamma1 + (x / Gamma2)**2 + Omega ** 2 / Gamma1 / Gamma2))**2

    def transmission_fit(self, plot_axes):
        # 1 - do fitting
        minTransmission = np.argmin(self.transmission_y)
        offset = self.transmission_x[minTransmission]

        popt, pcov = curve_fit(
            self.__transmission_fit_function, self.transmission_x, self.transmission_y,
            bounds=([0, 0, 0, offset - 0.1], [np.inf, np.inf, np.inf, offset + 0.1]))

        print("  > Gamma1:\t%.3f" % (popt[0]))
        print("  > Gamma2:\t%.3f" % (popt[1]))
        print("  > Omega:\t%.3f" % (popt[2]))
        print("  > Offset:\t%.3f" % (popt[3]))

        # 2 - prepare arrays and plot
        temp_x = np.linspace(min(self.transmission_x),
                             max(self.transmission_x), 500)
        temp_y = self.__transmission_fit_function(
            temp_x, popt[0], popt[1], popt[2], popt[3])

        minTransmission = np.argmin(temp_y)
        xoffset = temp_x[minTransmission]
        yoffset = temp_y[minTransmission]

        if(self.plot_or_not):
            plot_axes.plot(xoffset,
                           yoffset,
                           marker="o", color="C8", markeredgewidth=8, alpha=1)
            plot_axes.plot(temp_x, temp_y, color="C4")
            plot_axes.set_xlabel("$\omega_{21}/ 2 \pi$ (MHz)")
            plot_axes.set_ylabel("$|t|^2$")
            # 3 - save image
            plot_axes.set_facecolor("white")
            plt.savefig("output/transmission.png")
            plot_axes.set_facecolor("C7")
            plt.show()


if (__name__ == "__main__"):
    test = transmission(True)
    test.transmission_load(0, 1, True)
    test.transmission_filter(0.8, 10**12, 0, 2)
    test.transmission_plot(test.ax)
    test.transmission_fit(test.ax)
