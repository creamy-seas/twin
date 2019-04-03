# from twin_cuda import twin_cuda
from twin import twin
import numpy as np
from multiprocessing import Process
import multiprocessing


def individual_run(twinInstance, EC, EJ, alpha, assymetry):
    # 1 - create instance, load experimental data, perform simulation and crank error
    twinInstance.override_parameters(EC, EJ, alpha, assymetry)
    twinInstance.simulate()
    error = twinInstance.experimental_data_error()

    # 2 - string file to write to file
    string_to_write = str(EC) + "\t" + str(EJ) + "\t" + \
        str(alpha) + "\t" + str(assymetry) + \
        "\t" + str(error) + "\n"

    # 3 - write file
    with open("simulation_error.txt", 'a') as file_to_write:
        file_to_write.write(string_to_write)


if (__name__ == "__main__"):
    # 1 - parameters
    EC_array = np.linspace(5, 55, 11)
    EJ_array = np.linspace(5, 55, 11)
    alpha_array = np.linspace(1, 1.1, 11)
    assymetry_points = 16
    assymetry_array = np.linspace(1, 1.04, assymetry_points)

    # 2 - create class instances
    twinInstance = []
    for i in range(assymetry_points):
        twinInstance.append(twin(1, 1, 5, 300, False))
        twinInstance[i].experimental_data_load(twinInstance[i].ax, True)

    for EC in EC_array:
        for EJ in EJ_array:
            for alpha in alpha_array:
                p = []

                # 1 - launch parrallel processes for the assymetry arrray
                for i in range(0, len(assymetry_array)):
                    p.append(Process(target=individual_run,
                                     args=(twinInstance[i],
                                           EC, EJ, alpha, assymetry_array[i])))
                    p[i].start()

                # 2 - collect parrallel arrays together
                for i in range(0, len(assymetry_array)):
                    p[i].join()
