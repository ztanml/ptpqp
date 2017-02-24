# PTPQP
This repository contains the implementation of Partitioned Tensor Parallel Quadratic Programming (PTPQP)

## License
This software is distributed under the terms of the MIT license. See the COPYRIGHT file for details.

## Requirements
Requires Tensor Toolbox 2.6 in the MATLAB default PATH. Available at: http://www.sandia.gov/~tgkolda/TensorToolbox/index-2.6.html

## Usage
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

