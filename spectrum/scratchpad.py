import matplotlib as mpl
import matplotlib.pyplot as plt
import honkler
from twin import twin
from flux import flux
plt.style.use('ilya_plot')

# ############################################################
# ################### Paper Plot Inset ####################
# ############################################################
# test = twin(1, 1, 3, 100, True, False)
# test.sparse_matrix_visualise()
# honkler.save_ree(test.ax, "output/fig4_matrix", "svg")

# ############################################################
# ################### Paper Plot Inset ####################
# ############################################################
# # 1 - colouring and size
# mpl.rcParams["xtick.color"] = (0.53, 0.53, 1)
# mpl.rcParams["ytick.color"] = (0.53, 0.53, 1)
# mpl.rcParams["axes.labelcolor"] = (0.53, 0.53, 1)

# honkler.config_plot_size(0.4, 0.6, 0.3, 0.7)

# # 2 - load spectrum
# test = twin(1, 1, 7, 100, True, False)
# test.experimental_data_load(test.ax, True)

# # 3 - limits and labels
# test.ax.set_xlim([0.4, 0.6])
# test.ax.set_xticks([0.4, 0.5, 0.6])
# test.ax.set_ylim([5, 20])
# test.ax.set_yticks([0, 10, 20])
# test.ax.set_xlabel("Magnetic Flux ($\Phi$)")
# test.ax.set_ylabel("$\omega/2\pi$ (GHz)")

# # 4 - save
# honkler.save_ree(test.ax, "output/fig2_spectrum", "svg")

# ############################################################
# ################### Dipole transition ######################
# ############################################################
no_points = 1000
honkler.config_plot_size(0.2, 0.9, 0.15, 0.9)

# 1 - twin qubit
EC = 13.5
EJ = 92
alpha = 1.023
assymetry = 1.011
twinclass = twin(alpha, assymetry, 7, no_points, True, False)
twinclass.prepare_operators()
twinclass.override_parameters(EC, EJ, alpha, assymetry)
twinclass.simulate([True, False])

# 2 - flux qubit
EC = 20
EJ = 30
alpha = 0.45
fluxclass = flux(alpha, 7, no_points, True, False)
fluxclass.prepare_operators()
fluxclass.override_parameters(EC, EJ, alpha)
fluxclass.simulate([True, False])

# 3 - plotting
# a - axes setup
fig, axTwin = plt.subplots(nrows=1, ncols=1)
mngr = plt.get_current_fig_manager()
mngr.window.setGeometry(0, 30, 1280, 1600)
fig.canvas.set_window_title('Run time data')
axFlux = fig.add_subplot(111, sharex=axTwin, frameon=False)
axFlux.yaxis.tick_right()
axFlux.yaxis.set_label_position("right")

# b - plot
fluxclass.plot_dipole_moment_voltage_beta(axFlux)
twinclass.plot_dipole_moment_voltage_beta(axTwin)

# c - coloring
# axTwin.spines['right'].set_color('C6')
axTwin.yaxis.label.set_color('C6')
axTwin.tick_params(axis='y', colors='C6')
# axFlux.spines['right'].set_color('C9')
axFlux.yaxis.label.set_color('C9')
axFlux.tick_params(axis='y', colors='C9')

# d - scaling
axTwin.set_ylim([0, 1e9])
axFlux.set_ylim([0, 1e9])

# e - saving
honkler.save_ree(axTwin, "output/fig6_dipoleBeta", "svg")
