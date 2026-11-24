# parallel-numerical-integration-using-gaussian-quadrature
HPC for data science course project. Parallelization of complex algorithm. 

#### Build instructions
``` bash
gcc -O3 -std=c11 -Wall -Wextra -Isrc src/serial/main_serial.c src/common/gauss_rule.c -o bin/serial_quadrature -lm
```
