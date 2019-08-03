## Partitioned Tensor Factorization using Orthogonal Procrustes Matching
  * Provides Matlab code for the paper: http://proceedings.mlr.press/v70/tan17a/tan17a.pdf
  * Also includes fast precompiled C++ binaries for Linux and MacOS

## A Demo using the Fast C++ Implementation
  * The demo shows how to estimate both the multinomial parameters for each topic as well as the topic proportions of each observation.
  * Dataset: data/n10000d10000k5eps0.05.csv contains 10,000 observations each has 10,000 variables. Each variable takes values {0,1,2} and 5% noise is also added (e.g., draw from a uniform distribution with probability 0.05). The ground truth parameters used to generate the data are also given under data/n10000d10000k5eps0.05_\*.csv.
  * To use other datasets, make sure each variable takes values in {0,...,M_i}, where M_i can vary across variables.

To run the demo, decompress the data and use the following command (or simply ./run_example.sh in bash):
```bash
./gdlm.Platform -j200 -a output/Est_Alpha.csv -o output/Est_Multinomial.csv -r output/Est_Proportions.csv problems/n10000d10000k5eps0.05.conf
```
The command will create multiple threads to process the partitions whose number is specified by -j.

## Matlab Code
### Requirements
  * The Matlab code requires Tensor Toolbox 2.6 in the MATLAB default PATH. Available at: http://www.sandia.gov/~tgkolda/TensorToolbox/index-2.6.html

### Usage
To run the comparison on synthetic data:
In Matlab, run the script run_comparison.m. You may need to modify the configurations in the scrirpt.
The result will be saved in data/result.txt.
NOTE: the other tensor methods take time to run. Consider running this script on multiple machines with different configurations.

To run the crowdsourcing experiment, download the datasets and framework at: https://github.com/zhangyuc/SpectralMethodsMeetEM.
In the framework, replace the calls to tensor power method with the functions defined here. 

### Main files
- ptpqp.m  --- implements the proposed PTPQP algorithm
- tpqp.m   --- same as above but without partitioning

### Other files
- no_tenfact.m      --- implements nojd0 and nojd1, obtained from https://github.com/kuleshov/tensor-factorization
- qrj1d.m           --- same as above
- jacobi.m          --- same as above
- tpm.m             --- implements the tensor power method, obtained from https://github.com/kuleshov/tensor-factorization
- nonnegfac-matlab  --- implements Hals, obtained from https://gist.github.com/panisson/7719245
- MELD              --- implements MELD, obtained from https://github.com/judyboon/MELD


```
