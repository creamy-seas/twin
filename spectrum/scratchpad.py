import matplotlib.pyplot as plt
import honkler
from twin import twin
from flux import flux
plt.style.use('ilya_plot')


# ############################################################
# ################### Dipole transition ######################
# ############################################################
no_points = 5000
honkler.config_plot_size(0.2, 0.9, 0.15, 0.9)
EC = 13.5
EJ = 92
alpha = 1.023
assymetry = 1.011
twinclass = twin(alpha, assymetry, 7, 1000, True, False)
twinclass.prepare_operators()
twinclass.override_parameters(EC, EJ, alpha, assymetry)
twinclass.simulate([True, False])

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
axTwin.spines['right'].set_color('C6')
axTwin.yaxis.label.set_color('C6')
axTwin.tick_params(axis='y', colors='C6')
axFlux.spines['right'].set_color('C8')
axFlux.yaxis.label.set_color('C8')
axFlux.tick_params(axis='y', colors='C8')

# d - scaling
axTwin.set_ylim([0, 1e9])
axFlux.set_ylim([0, 1e9])

# e - saving
honkler.save_ree(axTwin, "output/fig5_dipoleBeta", "svg")
