For this assignment, you will need to update the search kernel function to attempt to find a passed value in a passed array

1. Ensure that you are in the project folder
2. Implement the search kernel
3. Run make clean build
4. You can run the search.exe CLI tool in the following ways:
   1. ./search.exe - This will create a random 10 value input array, roll a dice to determine if the search value is in the array or is random, and 128 threads per block
   2. ./search.exe -s true|false -t T -n N - This will create a random sorted|unsorted input array of N integers, coin flip if the search value is in the array or is random, and T threads per block
   3. ./search.exe -s true|false -t T -n N -v S -f test_data.csv - This will read a csv file any size (only tested with one line of comma-separated integers) and create a sorted|unsorted array, run with T threads per block, N is not used, and S as the search value. This will not completely work, this will be your extra credit for this assignment, as there is an issue with the copying the data into device memory.
5. If you do not include -p with an id, the output of the function will be added to a text file called output-test.txt, if you do it will be output-INPUT_FILENAME.txt

search.exe command line arguments:
   -s true|false - sort data prior to search
   -n numElements - the number of elements of random data to create
   -v searchValue - the value to search for in the data
   -f inputFile - the file for non-random input data
   -p currentPartId - the Coursera Part ID
   -t threadsPerBlock - the number of threads to schedule for concurrent processing

DO NOT TOUCH THE EXISTING PRINT STATEMENTS, THIS MAY CAUSE YOU TO FAIL AN ASSIGNMENT. You can add debug print statement but it should at the end look like the following:

Data: 8063503 93301624 236192674 453135755 604963279 636111415 717899641 1270883455 2004344247 2024977982 
Searching for value: 2004344247 
Found Index: -1
