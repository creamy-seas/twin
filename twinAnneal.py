# from twin_cuda import twin_cuda
from twin import twin
import numpy as np
from multiprocessing import Process
import multiprocessing
import time


def individual_run(twinInstance, EC, EJ, alpha, assymetry):
    # 1 - create instance, load experimental data, perform simulation and crank error
    twinInstance.override_parameters(EC, EJ, alpha, assymetry)
    twinInstance.simulate()
    error = twinInstance.experimental_data_error()

    # 2 - string file to write to file
    string_to_write = str(twinInstance.EC) + "\t" + str(twinInstance.EJ) + "\t" + \
        str(twinInstance.alpha) + "\t" + str(twinInstance.assymetry) + \
        "\t" + str(error) + "\n"

    # 3 - write file
    with open("simulation_error.txt", 'a') as file_to_write:
        file_to_write.write(string_to_write)


if (__name__ == "__main__"):
    # 1 - parameters
    EC_array = np.linspace(5, 95, 31)
    EJ_array = np.linspace(5, 95, 31)
    alpha_points= 16
    alpha_array = np.linspace(1, 1.2, alpha_points)
    assymetry_array = np.linspace(1, 1.05, 11)

    # 2 - create class instances
    twinInstance = []
    for i in range(len(alpha_array)):
        twinInstance.append(twin(1, 1, 5, 300, False))
        twinInstance[i].experimental_data_load(twinInstance[i].ax, True)

    start = time.time()
    for EC in EC_array:
        for EJ in EJ_array:
            for assymetry in assymetry_array:
                p = []

                # 1 - launch parrallel processes for the assymetry arrray
                for i in range(0, len(alpha_array)):
                    p.append(Process(target=individual_run,
                                     args=(twinInstance[i],
                                           EC, EJ, alpha_array[i], assymetry)))
                    p[i].start()

                # 2 - collect parrallel arrays together
                for i in range(0, len(alpha_array)):
                    p[i].join()

    end = time.time()
    print("Total time:")
    print(end - start)
