from twin_cuda import twin_cuda
from twin import twin
import numpy as np
from multiprocessing import Process
import multiprocessing


def individual_run(EC, EJ, alpha, assymetry, string_array, index):
    # 1 - create instance, load experimental data, perform simulation and crank error
    twinInstance = twin(EC, EJ, alpha, assymetry, 7, 300, False)
    twinInstance.experimental_data_load(twinInstance.ax, True)
    twinInstance.simulate()
    error = twinInstance.experimental_data_error()

    # 2 - string file to write to file
    string_to_write = str(EC) + "\t" + str(EJ) + "\t" + \
        str(alpha) + "\t" + str(assymetry) + \
        "\t" + str(error) + "\n"
    print(string_to_write)
    string_array[index] = string_to_write


if __name__ == "__main__":
    # 1 - parameters
    EC_array = np.linspace(5, 55, 11)
    EJ_array = np.linspace(5, 55, 11)
    alpha_array = np.linspace(1, 1.1, 11)
    assymetry_array = np.linspace(1, 1.04, 2)

    for EC in EC_array:
        for EJ in EJ_array:
            for alpha in alpha_array:
                p = [None] * len(assymetry_array)
                string_array = [None] * len(assymetry_array)

                # 1 - launch parrallel processes for the assymetry arrray
                for i in range(0, len(assymetry_array)):
                    p[i] = Process(target=individual_run, args=(
                        EC, EJ, alpha, assymetry_array[i], string_array, i))
                    p[i].start()

                # 2 - collect parrallel arrays together
                for i in range(0, len(assymetry_array)):
                    p[i].join()

                print(string_array)

                string_to_write = "".join(string_array)
                print(string_to_write)

                # with open("simulation_error.txt", 'a') as file_to_write:
                #     file_to_write.write(string_to_write)
