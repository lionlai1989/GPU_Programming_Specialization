set terminal png size 2000,1000 enhanced
set output 'output.png'
set xlabel 'data points per thread'
set ylabel 'kernel execution (sec)'
set grid
set title 'CUDA Device Memory Analysis'
set key right outside 
plot 'output.dat' u 1:2 w lp t 'global', 'output.dat' u 1:3 w lp t 'shared', 'output.dat' u 1:4 w lp t 'register'