#!/usr/bin/python3
import sys


def main(file_location):
    total_times, total_iterations, kernels, elements = process_file(file_location)
    mean_times = process_times_dicts(total_times=total_times, total_iterations=total_iterations)
    generate_dat_file(mean_times=mean_times, kernels=kernels, elements=elements)


def process_times_dicts(total_times, total_iterations):
    mean_times = {}
    for element_number, element_number_kernel_total_times in total_times.items():
        element_kernel_total_iteration = total_iterations[element_number]
        mean_times[element_number] = {}
        for kernel_name, kernel_total_time in element_number_kernel_total_times.items():
            kernel_total_iterations = element_kernel_total_iteration[kernel_name]
            kernel_mean_time = kernel_total_time / kernel_total_iterations
            mean_times[element_number][kernel_name] = kernel_mean_time
    return mean_times


def generate_dat_file(mean_times, kernels, elements):
    with open('output.dat', 'a') as f:
        num_lines = 0
        for element in elements:
            element_kernel_mean_times = mean_times[element]
            line = str(element) + " "
            for kernel_name in kernels:
                kernel_mean_time = element_kernel_mean_times[kernel_name]
                # for kernel_name, kernel_mean_time in element_kernel_mean_times.items():
                line = line + str(kernel_mean_time) + " "
            line = line.rstrip()
            if num_lines < len(mean_times) - 1:
                line = line + "\n"
            num_lines = num_lines + 1
            f.write(line)


def process_file(file_location):
    fo = open(file_location)
    line = fo.readline()
    
    try:
        # Loop until EOF
        total_times = {}
        total_iterations = {}
        kernels = []
        elements = []
        while line != '':
            line = line.strip()
            parts = line.split(',')
            kernel = parts[1]
            if kernel not in kernels:
                kernels.append(kernel)
            element = int(parts[3])
            if element not in elements:
                elements.append(element)
            time = float(parts[4])
            if element not in total_times:
                total_times[element] = {}
            if element not in total_iterations:
                total_iterations[element] = {}
            if kernel not in total_times[element]:
                total_times[element][kernel] = 0
            if kernel not in total_iterations[element]:
                total_iterations[element][kernel] = 0
            total_times[element][kernel] = total_times[element][kernel] + time
            total_iterations[element][kernel] = total_iterations[element][kernel] + 1
            line = fo.readline()
    finally:
        fo.close()
    elements.sort()
    return total_times, total_iterations, kernels, elements


if __name__ == '__main__':
    file = "output.csv"
    if len(sys.argv) > 1:
        file = sys.argv[1]
    main(file_location=file)
