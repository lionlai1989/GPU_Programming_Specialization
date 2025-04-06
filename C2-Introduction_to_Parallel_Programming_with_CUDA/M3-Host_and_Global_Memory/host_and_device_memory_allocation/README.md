# Assignment Instructions

For this assignment, you will need to update the host code to appropriately allocate host and device memory, copy from host to device, execute the kernel, copy device to host memory, and print results. Much of the code will be provided, but certain areas will be left for the student to fill out and for the full 100% grade you will need to fix a broken program that may run but it will return unexpected results. The following steps should be taken while developing your solutions to the various steps of the assignment:

1. F​ill in the small bit of code that handles allocation of host and device memory in the memory_allocation.cu file. Note this block will start and end with comments like // FILL IN HOST AND DEVICE MEMORY ALLOCATION CODE.
2. Click the Build Parts 1 and 2 buttons.
3. F​ill in the small bit of code that copies host to device memory in the memory_copy.cu file. Note this block will start and end with comments like // FILL IN HOST TO DEVICE MEMORY COPY CODE.
4. Click the Build Parts 3 and 4 buttons.
5. U​pdate code in the broken_paged_pinned_memory_allocation.cu file. There will be no indications as to what is broken.
6. C​lick the Build Part 5 button.
7. U​pdate code in the broken_mapped_memory_allocation.cu file. There will be no indications as to what is broken.
8. C​lick the Build Part 6 button.
9. O​nce you are satisfied with your work, click the Submit Parts 1-6 button.

For this assignment all executables will take the following command line arguments:
   -n numElements - the number of elements of random data to create
   -f inputFile - the file for non-random input data
   -o mathematicalOperation - this will decide which math operation kernel will be executed
   -p currentPartId - the Coursera Part ID
   -t threadsPerBlock - the number of threads to schedule for concurrent processing

DO NOT TOUCH THE EXISTING PRINT STATEMENTS, THIS MAY CAUSE YOU TO FAIL AN ASSIGNMENT. You can add debug print statement but it should at the end look like the following:

PartID: part-id
Input A: 1 2 3 4 5 6 7 8 9 10
Input B: 1 2 3 4 5 6 7 8 9 10
Result: 1 2 3 4 5 6 7 8 9 10
