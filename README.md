# PTPQP
This repository contains the implementation of Partitioned Tensor Parallel Quadratic Programming (PTPQP)

See our ICML 2017 paper at: http://proceedings.mlr.press/v70/tan17a/tan17a.pdf

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

If you like this project, consider citing the paper:
```bibtex
@InProceedings{tan17a,
  title = 	 {Partitioned Tensor Factorizations for Learning Mixed Membership Models},
  author = 	 {Zilong Tan and Sayan Mukherjee},
  booktitle = 	 {Proceedings of the 34th International Conference on Machine Learning},
  pages = 	 {3358--3367},
  year = 	 {2017},
  editor = 	 {Doina Precup and Yee Whye Teh},
  volume = 	 {70},
  series = 	 {Proceedings of Machine Learning Research},
  address = 	 {International Convention Centre, Sydney, Australia},
  month = 	 {06--11 Aug},
  publisher = 	 {PMLR},
  pdf = 	 {http://proceedings.mlr.press/v70/tan17a/tan17a.pdf},
  url = 	 {http://proceedings.mlr.press/v70/tan17a.html}
}

```
