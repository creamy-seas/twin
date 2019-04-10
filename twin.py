import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
import matplotlib.pyplot as plt
from quantum_master import quantum_master
import time
plt.style.use('ilya_plot')


class twin(quantum_master):
    """
    Quantum class to work with the twin, 5JJ qubit.
    """

    def __init__(self, alpha, assymetry, states_per_island, flux_points, plot_or_not):
        """
        __ Parameters __
        EC:     charging energy
        EJ:     josephson energy
        alpha:  central island
        assymetry: assymetry between two loops
        states_per_island: accuracy of simulation
        plot_or_not: show plot output or not

        __ Description __
        Perform the general setup
        - measured in nm
        - working with frequencies (normalised by hbar)
        - working in unit of Phi0
        """
        quantum_master.__init__(self, plot_or_not)

        # 1 - store user supplied parameters
        self.alpha = alpha
        self.assymetry = assymetry
        # --------------------
        self.delta = 0

        # 2 - global parameters
        self.flux_min = -1
        self.flux_max = 1
        self.flux_points = flux_points
        self.states_per_island = states_per_island
        self.states_total_number = self.states_per_island**3
        self.flux_list = np.linspace(
            self.flux_min, self.flux_max, self.flux_points)

        # 3 - simulation preparation
        self.prepare_correction()
        self.prepare_structure()
        self.prepare_normalised_hamiltonian()

    def override_parameters(self, EC, EJ, alpha, assymetry):
        """
        __ Parameters __
        EC, EJ, alpha, assymetry: parameters to set

        __ Description __
        For new simulations, set the new system parameters
        """
        print("==> 'override_parameters' with EC=%.4f\tEJ=%.4f\talpha=%.4f\tass=%.4f" %
              (EC, EJ, alpha, assymetry))
        self.EC = EC
        self.EJ = EJ
        self.alpha = alpha
        self.assymetry = assymetry

    def prepare_correction(self):
        """
        Correct common errors before program begins
        """
        if ((self.states_per_island % 2) == 0):
            print("==> Changing states_per_island from %data_set -> %data_set" %
                  (self.states_per_island, self.states_per_island - 1))
            self.states_per_island = self.states_per_island - 1

    def prepare_structure(self):
        """
        Prepares the structure parameters
        """
        print("==> 'prepare_structure' creating energies and capacitances")
        # 1 - set jj dimensions
        self.param_jj_squares = 2
        self.param_jj_overlap_area = 200 * 200 * \
            self.param_jj_squares

        # 2 - set the energies EC and EJ
        self.energy_charging(self.param_jj_overlap_area)
        self.energy_josephson(self.param_jj_squares)

        # 3 - set capacitance matrix
        self.capacitance_normalised = np.matrix([
            [2, -1, 0],
            [-1, 2 + self.alpha + self.delta, -1],
            [0, -1, 2]])

    def prepare_operators_normalised(self):
        # additional operator components
        self.op_V_row = []
        self.op_V_col = []
        self.op_V_elm = []
        self.op_Phi_row = []
        self.op_Phi_col = []
        self.op_Phi_elements = []

        for x in range(0, self.states_total_number):

            # 2 - convert the index to island occupation
            self.convert_index_to_island_state(x)

            # 3b - voltage operator
            voltage_element = 2 * self.const_eCharge / self.param_capacitance * (
                (2 + self.alpha) * self.state_cp_distribution[0, 1]
                - self.state_cp_distribution[0, 0]
                - self.state_cp_distribution[0, 2])
            self.op_V_row.append(x)
            self.op_V_col.append(x)
            self.op_V_elm.append(voltage_element)

            if (self.state_numerical_distribution[1] < (self.states_per_island - 1)):
                # island 2 (element 1)
                y = self.convert_numerical_state_to_index(
                    self.state_numerical_distribution + [0, 1, 0])

                # 4b - phi element
                self.op_Phi_row.append(x)
                self.op_Phi_col.append(y)
                self.op_Phi_elements.append(1)

        # 5 - finalise operators
        self.op_V = sp.coo_matrix((self.op_V_elm,
                                   (self.op_V_row, self.op_V_col))).tocsr()
        self.op_Phi = sp.coo_matrix((self.op_Phi_elements,
                                     (self.op_Phi_row, self.op_Phi_col))).tocsr()

    def prepare_normalised_hamiltonian(self):
        """
        Generate elements for a normalised Hamiltonian.

        During the simulation 'prepare_hamiltonian()' must be run, to change energies
        """
        print("==> 'prepare_normalised_hamiltonian' creating Hamiltonian entries")
        # 0 - Individual Hamiltonian components.
        # constant
        self.op_H_charging_elm = []
        self.op_H_charging_row = []
        self.op_H_charging_col = []
        self.op_H_diag_row = []
        self.op_H_diag_col = []
        self.op_H_diagA_row = []
        self.op_H_diagA_col = []
        # changing
        self.op_H_phi_row = []
        self.op_H_phi_col = []
        self.op_H_phiAss_row = []
        self.op_H_phiAss_col = []

        # 1 - generate matrix elements, by going across all the states
        for x in range(0, self.states_total_number):

            # 2 - convert the index to island occupation
            self.convert_index_to_island_state(x)

            # 3 - diagonal charging energy
            charging_energy = (self.state_cp_distribution *
                               (self.capacitance_normalised.I) *
                               (self.state_cp_distribution.transpose()))

            self.op_H_charging_row.append(x)
            self.op_H_charging_col.append(x)
            self.op_H_charging_elm.append(charging_energy[0, 0])

            # 4 - offdiagonal JJ energy - check that within matrix boundaries
            # offset y coordinate from diagonal, and fill out the symmetrical entries
            # ! 'diag_elm' array is not created, since all elements will be the same
            if (self.state_numerical_distribution[0] < (self.states_per_island - 1)):
                # island 1 (element 0)
                y = self.convert_numerical_state_to_index(
                    self.state_numerical_distribution + [1, 0, 0])
                self.op_H_diag_row.extend([x, y])
                self.op_H_diag_col.extend([y, x])

                # cross diagonal terms, with cp exchange between 1 <-> 2
                # if (self.state_numerical_distribution[1] > 0):
                #     y = self.convert_numerical_state_to_index(
                #         self.state_numerical_distribution + [1, -1, 0])
                #     self.op_H_phi_row.extend([x, y])
                #     self.op_H_phi_col.extend([y, x])

            if (self.state_numerical_distribution[1] < (self.states_per_island - 1)):
                # island 2 (element 1)
                y = self.convert_numerical_state_to_index(
                    self.state_numerical_distribution + [0, 1, 0])
                self.op_H_diagA_row.extend([x, y])
                self.op_H_diagA_col.extend([y, x])

            if (self.state_numerical_distribution[2] < (self.states_per_island - 1)):
                # island 3 (element 2)
                y = self.convert_numerical_state_to_index(
                    self.state_numerical_distribution + [0, 0, 1])
                self.op_H_diag_row.extend([x, y])
                self.op_H_diag_col.extend([y, x])

                # cross diagonal terms, with cp exchange between 2 <-> 3
                # if (self.state_numerical_distribution[1] > 0):
                #     y = self.convert_numerical_state_to_index(
                #         self.state_numerical_distribution + [0, -1, 1])
                #     self.op_H_phiAss_row.extend([x, y])
                #     self.op_H_phiAss_col.extend([y, x])

        print("  > Unchaning part of Hamiltonian has %i entries" %
              (len(self.op_H_charging_row + self.op_H_diag_row + self.op_H_diagA_row)))
        print("  > Flux-dependent part of Hamiltonian has %i entries" %
              (len(self.op_H_phiAss_row + self.op_H_phiAss_row)))
        print("==> 'prepare_normalised_hamiltonian' finished")

    def prepare_hamiltonian(self):
        """
        __ Description __
        Using supplied EC, EJ, alpha, scale the normalised list created in 'prepare_normalised_hamiltonian' and
        combine them into 1.

        The lists are used in 'simulate' functions
        """
        print("==> 'prepare_hamiltonian' with EC=%.4f\tEJ=%.4f\talpha=%.4f\tass=%.4f" %
              (self.EC, self.EJ, self.alpha, self.assymetry))

        # 1 - main part of the Hamiltonian, which remains unchanged during simulation
        # temp_charging_elm = self.EC * np.array(self.op_H_charging_elm)
        # temp_diag_elm = - self.EJ / 2 * np.ones(len(self.op_H_diag_row))
        # temp_diagA_elm = - self.alpha * self.EJ / \
        #     2 * np.ones(len(self.op_H_diagA_row))

        temp_charging_elm = np.array(self.op_H_charging_elm)
        temp_diag_elm = np.ones(len(self.op_H_diag_row))
        temp_diagA_elm = 0.99 * np.ones(len(self.op_H_diagA_row))

        # self.op_H_elm = list(temp_charging_elm) + \
        #     list(temp_diag_elm) + list(temp_diagA_elm)
        # self.op_H_row = self.op_H_charging_row + \
        #     self.op_H_diag_row + self.op_H_diagA_row
        # self.op_H_col = self.op_H_charging_col + \
        #     self.op_H_diag_col + self.op_H_diagA_col

        self.op_H_elm = list(temp_diag_elm)
        self.op_H_row = self.op_H_diag_row
        self.op_H_col = self.op_H_diag_col

        # self.op_H_elm = list(temp_diag_elm) + list(temp_diagA_elm)
        # self.op_H_row = self.op_H_diag_row + self.op_H_diagA_row
        # self.op_H_col = self.op_H_diag_col + self.op_H_diagA_col

        if((len(self.op_H_row) != len(self.op_H_elm)) or (len(self.op_H_col) != len(self.op_H_elm))):
            self.raise_error("Hamiltonin has %i rows, %i columns non zero coordinatres and only %i elements" % (
                len(self.op_H_row), len(self.op_H_col), len(self.op_H_elm)))

        # 2 - exchange elements - used by 'prepare_hamiltonian_exchange' to fill out at different fluxes
        no_exchange = int(len(self.op_H_phi_row) / 2)
        no_exchangeAss = int(len(self.op_H_phiAss_row) / 2)
        if(no_exchange != no_exchangeAss):
            self.raise_error("Number of 1 <-> 2 exchanges (%i) different from 2 <-> 3 exchanges (%i)" %
                             (no_exchange, no_exchangeAss))
        self.op_H_SUPPORT_exchange_elm = np.ones(no_exchange)

        print("==> 'prepare_hamiltonian' finished")

    def prepare_hamiltonian_exchange(self, phi_external, phi_externalAss):
        """
        __ Parameters __
        phi_external, phi_externalAss: fluxes to the two loops

        __ Description __
        The exchange parts of the Hamiltonian depend on the external fluxes, which changes for
        each simulation as the field is being swept. This method fills out a list
        of exchange values that needs to be added to the Hamiltonian list:

                    op_H_elm.extend(elm_list)

        ! remember to extend the row and column lists as well e.g.
                    op_H_row.extend(self.op_H_phi_row)

        __ Return __
        List that should extend the Hamiltonian in each simulation run
        """

        # 1 - generate lists with coefficients scaled by external flux
        temp_phi_p = (-self.EJ / 2 * np.exp(1j * phi_external)) * \
            self.op_H_SUPPORT_exchange_elm
        temp_phi_n = (-self.EJ / 2 * np.exp(-1j * phi_external)) * \
            self.op_H_SUPPORT_exchange_elm
        temp_phiAss_n = (-self.EJ / 2 * np.exp(-1j *
                                               phi_externalAss)) * self.op_H_SUPPORT_exchange_elm
        temp_phiAss_p = (-self.EJ / 2 * np.exp(1j *
                                               phi_externalAss)) * self.op_H_SUPPORT_exchange_elm

        # 2 - interleave the lists, alternating between +ve and -ve fluxes, for the (x,y) and (y,x)
        # pairings in the Hamiltonian
        elm_list = list(np.ravel(np.column_stack((temp_phi_p, temp_phi_n))))
        elm_list = elm_list + \
            list(np.ravel(np.column_stack((temp_phiAss_n, temp_phiAss_p))))

        return elm_list

    def simulate(self):
        """
        __ Description __
        Method performs the eigenvalue simulations:
        1 - finish building the Hamiltonian, with the flux-dependent elements
        2 - evaluate the lowest eigenstate and eigenenergies of the Hamiltonian
        3 - plot out the spectrum

        """
        # 0 - prepare hamiltonian for this simulation
        self.prepare_hamiltonian()
        print("==> 'simulate' running")
        self.spectrum_eigvals = []
        self.spectrum_eigvecs = []
        self.spectrum_simulation_12 = []
        self.spectrum_simulation_23 = []

        # 1 - simulation performed for each value of the external flux
        for ext_flux_number in range(0, len(self.flux_list)):

            # 2 - extraction of the onset phase, due to external flux
            phi_external = (self.flux_list[ext_flux_number]) * 2 * np.pi
            phi_externalAss = phi_external * self.assymetry

            ####################
            # 3 - FINISH MAIN HAMILTONIAN for this particular bias
            ####################
            # a - copy base elements
            op_H_row = self.op_H_row.copy()
            op_H_col = self.op_H_col.copy()
            op_H_elm = self.op_H_elm.copy()

            # b - add on the phase dependent elements
            op_H_row.extend(self.op_H_phi_row)
            op_H_row.extend(self.op_H_phiAss_row)
            op_H_col.extend(self.op_H_phi_col)
            op_H_col.extend(self.op_H_phiAss_col)
            op_H_elm.extend(self.prepare_hamiltonian_exchange(
                phi_external, phi_externalAss))

            if((len(op_H_row) != len(op_H_elm)) or (len(op_H_col) != len(op_H_elm))):
                self.raise_error("Hamiltonin lists have %i rows, %i columns  and only %i elements" % (
                    len(op_H_row), len(op_H_col), len(op_H_elm)))

            # 2 - construct sparse matrix
            self.op_H = sp.coo_matrix((op_H_elm,
                                       (op_H_row, op_H_col))).tocsr()
            # self.sparse_matrix_plot(self.op_H)

            # 3 - evaluate 3 lowest eigenenergies and eigenvectors, |1> |2> |3>
            eigvals, eigvecs = eigsh(self.op_H, 4, which='SA', tol=0)
            self.spectrum_eigvals.append(eigvals)
            self.spectrum_simulation_12.append([eigvals[1] - eigvals[0]])
            self.spectrum_simulation_23.append([eigvals[2] - eigvals[1]])

            self.track_progress(ext_flux_number, len(
                self.flux_list), 20, False)

            time.sleep(0.005)

        # 4 - finalise arrays
        self.spectrum_eigvals = np.array(self.spectrum_eigvals)
        self.spectrum_simulation_12 = np.array(self.spectrum_simulation_12)
        self.spectrum_simulation_23 = np.array(self.spectrum_simulation_23)
        print("==> 'simulate' finished")

        # 5 - plotting
        print("==> Plotting results")
        self.simulation_plot(self.ax)

    def track_progress(self, current_number, total_number, increment, heavy):
        """
        __ Parameters __
        current_number: current iteration
        total_number: full iteration
        increment: what % increments to plot in
        heavy: if false, a simple ouput is shown

        __ Description __
        print out current progress, with a given increment
        """
        no_stars = 50

        # 1 - find completed runs
        completion = int(current_number / total_number * 100)

        # 2 - generate array is we hit increment
        if((completion % increment) == 0):
            if(heavy):
                current_stars = int(completion / 100 * no_stars)
                stars = ["*"] * current_stars
                space = ["-"] * (no_stars - current_stars)
                stars.extend(space)
                output = "".join(stars)
                print("[%s][%i/%i]" %
                      (output, current_number, total_number))
            else:
                print("  > [%i/%i]" %
                      (current_number, total_number))

    def sparse_matrix_plot(self, sparse_matrix_to_plot):
        """
        __ Parameters __
        sparse_matrix_to_plot: matrix to show

        __ Description __
        show the sparse matrix via a plot
        """
        if(self.plot_or_not):
            plt.spy(sparse_matrix_to_plot, markersize=3)
            plt.show()

    def simulation_plot(self, plotAxes):
        """
        Plot the eigenvalues and transition spectrum
        """
        # 1 - prepare plot
        if(self.plot_or_not):
            plotAxes.plot(self.flux_list,
                          self.spectrum_simulation_12, label="1<->2", color='#004BA8')
            plotAxes.plot(self.flux_list,
                          self.spectrum_simulation_23, label="2<->3", color='#348ABD')
            plotAxes.set_ylim(0, 20)

            plt.show()

    def experimental_data_load(self, plotAxes, set_flux_list):
        """
        __ Parameters __
        plotAxes: axes to plot the raw data on
        set_flux_list: True to set experimental data for the flux list

        __ Description __
        1 - load up experimental data
        2 - perform conversion from mA to flux
        3 - plot data
        4 - store the magnetic field points of the experiment

        Plots and finds differences between simulation and experiment
        """
        print("==> 'experimental_data_load' Importing data files")

        # 1 - data files to load
        base_data_name = "qubit2_data/Qubit15_5JJ_Q2_"
        transition_12 = ["m3", "m2", "m1", "1", "2", "3"]
        transition_23 = ["m3b", "m2b", "m1b", "1b", "2b"]

        # 2 - for each data file
        self.flux_list_experimental_12 = []
        self.spectrum_experimental_12 = []
        self.flux_list_experimental_23 = []
        self.spectrum_experimental_23 = []

        for data_set in range(0, len(transition_12)):
            # a - generate file name
            temp_name = base_data_name + transition_12[data_set] + ".txt"

            # b - import, sort and convert to flux
            temp_data = self.import_and_sort_array(temp_name, 0)
            temp_data[0] = self.convert_mA_to_flux(temp_data[0], 0.125, 0.7)

            # c - plot data
            if(self.plot_or_not):
                plotAxes.plot(temp_data[0], temp_data[1], marker='o',
                              color='#004BA8', markersize=2, linestyle='')

            # d - store imported data in 1 array
            self.flux_list_experimental_12.extend(temp_data[0])
            self.spectrum_experimental_12.extend(temp_data[1])

        for data_set in range(0, len(transition_23)):
            # a - generate file name
            temp_name = base_data_name + transition_23[data_set] + ".txt"

            # b - import, sort and convert to flux
            temp_data = self.import_and_sort_array(temp_name, 0)
            temp_data[0] = self.convert_mA_to_flux(temp_data[0], 0.125, 0.7)

            # c - plot data
            if(self.plot_or_not):
                plotAxes.plot(temp_data[0], temp_data[1], marker='o',
                              color='#348ABD', markersize=2, linestyle='')

            # d - store imported fluxes
            self.flux_list_experimental_23.extend(temp_data[0])
            self.spectrum_experimental_23.extend(temp_data[1])

        # 3 - tidy up by sorting arrays
        temp_array = self.sort_array(
            np.array([self.flux_list_experimental_12, self.spectrum_experimental_12]), 0)
        self.flux_list_experimental_12 = temp_array[0]
        self.spectrum_experimental_12 = temp_array[1]

        temp_array = self.sort_array(
            np.array([self.flux_list_experimental_23, self.spectrum_experimental_23]), 0)
        self.flux_list_experimental_23 = temp_array[0]
        self.spectrum_experimental_23 = temp_array[1]

        print("  > Imported %i flux points" % (
            len(list(self.flux_list_experimental_12) + list(self.flux_list_experimental_23))))

        # 4 - set the flux array is required (then simulations are only done to compare with
        # experimental points)
        if(set_flux_list):
            print("  > Set experimental flux points for simulation")
            self.flux_list = np.array(
                list(self.flux_list_experimental_12) + list(self.flux_list_experimental_23))
            self.flux_list.sort()

        if(self.plot_or_not):
            plt.show()

        print("==> 'experimental_data_load' finished")

    def experimental_data_error(self):
        """
        __ Description __
        Error between the experimental data points and the simulation
        """
        try:
            print("==> 'experimental_data_error' comparing simulation to experiment")
            if (not hasattr(self, "spectrum_simulation_12")):
                self.raise_error(
                    "*** WARNING - run the simulation first before calling 'experimental_data_error'")
            if (not hasattr(self, "spectrum_experimental_12")):
                self.raise_error(
                    "*** WARNING - import experimental data before calling 'experimental_data_error'")

            # 1 - prepare parameters
            error_cumulative = 0

            # 2 - transition12
            for i in range(0, len(self.flux_list_experimental_12)):
                # a - find the simulation entry corresponding to the experimental data point
                entry = np.where(self.flux_list ==
                                 self.flux_list_experimental_12[i])[0][0]

                # b - compute the difference
                error_cumulative = error_cumulative + (
                    self.spectrum_simulation_12[entry] - self.spectrum_experimental_12[i])[0]**2

            # 3 - transition23
            for i in range(0, len(self.flux_list_experimental_23)):
                # a - find the simulation entry corresponding to the experimental data point
                entry = np.where(self.flux_list ==
                                 self.flux_list_experimental_23[i])[0][0]

                # b - compute the difference
                error_cumulative = error_cumulative + (
                    self.spectrum_simulation_23[entry] - self.spectrum_experimental_23[i])[0]**2

            print("==> 'experimental_data_error' finished")
            return error_cumulative
        except TypeError as e:
            print(e)

    def import_and_sort_array(self, file_name, i):
        """
        __ Parameters __
        file_name: file to import, with path and extensions
        i: column to sort by

        __ Description __
        Import the 2D array and sort by the i-th column

        __ Return __
        return the sorted array
        """

        # 1 - import array
        array_to_return = np.loadtxt(file_name).transpose()

        # 2 - sort array by the i-th column
        sorted_index = np.argsort(array_to_return[i])
        array_to_return[0] = np.array(array_to_return[0])[sorted_index]
        array_to_return[1] = np.array(array_to_return[1])[sorted_index]

        return array_to_return

    def sort_array(self, array_to_sort, column_to_sort_by):
        """
        __ Parameters __
        array_to_sort: array to perform sorting for
        column_to_sort_by: which column to use for sorting

        __ Description __
        sorts the supplied array
        """
        # 1 - sort array by the i-th column
        sorted_index = np.argsort(array_to_sort[column_to_sort_by])
        for i in range(0, len(array_to_sort)):
            array_to_sort[i] = np.array(array_to_sort[i])[sorted_index]
        return array_to_sort

    def convert_index_to_island_state(self, index):
        """
        __ Parameters __
        index: number convert (0 <= index < states_total_number)

        __ Description __
        Takes an index and converts it to a unique cooper pair distribution
        across the 3 islands of the system - (n1, n2, n3).

        __ Changes __
        state_numerical_distribution:
            an ARRAY of the 3-island state
        state_cp_distribution:
            a MATRIX of the 3-island states, so that central index has a charge
            distribution (0,0,0)
        """

        # 0 - error checking
        if((index < 0) or (index >= self.states_total_number)):
            print("Index %data_set must be within 0 and %data_set -> converting to 0" %
                  (index, self.states_total_number))
            index = 0

        # 1 - evaluate the offset to apply, to get centering about (0,0,0)
        minimal_number_of_cp_on_island = (1 - self.states_per_island) / 2

        # 2 - represent 'index' by a trinary and cooper pair distribution
        self.state_numerical_distribution = []
        self.state_cp_distribution = []
        workingNumber = index

        # 3 - decomposition of a number to self.states_per_island base-system
        for island in [2, 1, 0]:

            # a - get the cp representation on this island
            island_no_cp = int(
                workingNumber / (self.states_per_island**island))

            # b - store numerical and cp values
            self.state_numerical_distribution.append(island_no_cp)
            self.state_cp_distribution.append(
                island_no_cp + minimal_number_of_cp_on_island)

            # c - decrease the working number
            workingNumber = workingNumber - island_no_cp * \
                (self.states_per_island)**island

        # 4 - convert to numpy arrays
        self.state_numerical_distribution = np.array(
            self.state_numerical_distribution)
        self.state_cp_distribution = np.matrix(self.state_cp_distribution)

    def convert_numerical_state_to_index(self, state_numerical_distribution):
        """
        __ Parameters __
        state_numerical_distribution: array of the trinary state (NOT NORMALISED!)

        __ Description __
        Given a number in the 3 island representation
        convert it to an index for the matrix

        __ Return __
        index between 0 and total number of states that corresponds to supplied
        array
        """
        index = 0
        for data_set in range(0, 3):
            index = index + \
                state_numerical_distribution[data_set] * \
                self.states_per_island**data_set

        return index

    def convert_mA_to_flux(self, mA_array, offset, period):
        """
        __ Parameters __
        mA_array: array of mA values to convert
        offsset:  offset from 0
        period:   period in mA

        __ Description __
        Converts array measured in experimental mA to flux units

        __ Return __
        flux_array
        """
        # return mA_array
        return (mA_array - offset) / period


if (__name__ == "__main__"):
    print("\nRunning 'twin.py'\n")
    start = time.time()

    EC = 13.5
    EJ = 91
    alpha = 1.011
    assymetry = 1.023
    test = twin(alpha, assymetry, 3, 100, False)
    test.override_parameters(EC, EJ, alpha, assymetry)
    test.prepare_hamiltonian()
    temp = sp.coo_matrix((test.op_H_elm,
                          (test.op_H_row, test.op_H_col)))

    print(temp)
    # test.experimental_data_load(test.ax, False)
    # test.simulate()
    # test.ax.set_xlim([-1, 1])
    # print(test.experimental_data_error())

    # end = time.time()
    # print(end - start)
