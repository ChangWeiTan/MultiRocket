# MultiRocket

**Effective summary statistics for convolutional outputs in time series classification**

Rocket and MiniRocket, while two of the fastest methods for time series classification, 
are both somewhat less accurate than the current most accurate methods (namely, HIVE-COTE and 
its variants).  We show that it is possible to significantly improve the accuracy of 
MiniRocket (and Rocket), with some additional computational expense, by expanding the set of 
features produced by the transform, making MultiRocket (for MiniRocket with Multiple Features) 
overall the single most accurate method on the datasets in the UCR archive, while still being 
orders of magnitude faster than any algorithm of comparable accuracy other than its precursors.

## Requirements
All python packages needed are listed in [pip-requirements.txt](pip-requirements.txt) file 
and can be installed simply using the pip command. 

* [pandas](https://pandas.pydata.org/)
* [numpy](https://numpy.org/)
* [numba](http://numba.pydata.org/) 
* [sklearn](https://scikit-learn.org/stable/)
* [catch22](https://github.com/chlubba/catch22) (optional)

## Code
The [main.py](main.py) file contains a simple code to run the program on a single UCR dataset.

The [main_ucr_109.py](main_ucr_109.py) file runs the program on all 109 UCR datasets.
```
Arguments:
-d --data_path          : path to dataset
-p --problem            : dataset name
-i --iteration          : determines the resample of the UCR datasets
-f --featureid          : feature id for MultiRocket
-n --num_features       : number of features 
-k --kernel_selection   : 0=use MiniRocket kernels (default), 1=use Rocket kernels
-t --num_threads        : number of threads (> 0)
-s --save               : 0=don't save results, 1=save results
``` 

## Results
These are the results on 30 resamples of the 109 UCR Time Series archive 
from the [UCR Time Series Classification Archive](https://www.cs.ucr.edu/~eamonn/time_series_data_2018/).
MultiRocket is on average the current most accurate scalable TSC algorithm, that is more accurate than 
HIVE-COTE/TDE (HC-TDE).
![image](results/figures/cd_multirocket_sota_resample.png)
![image](results/figures/timings_vs_minirocket.png)

## Reference
If you use any part of this work, please cite:
```
@article{Tan2021MultiRocket,
  title={MultiRocket: Effective summary statistics for convolutional outputs in time series classification},
  author={Tan, Chang Wei and Dempster, Angus and Bergmeir, Christoph and Webb, Geoffrey I},
  year={2021}
}
```

## Acknowledgement
We would like to thank Professor Eamonn Keogh, Professor Tony Bagnall and their team who have provided the 
[UCR time series classification archive](https://www.cs.ucr.edu/~eamonn/time_series_data_2018/) and 
making a comprehensive benchmark [results](http://timeseriesclassification.com/results.php) widely available.
We also appreciate the open source code to draw the critical difference diagrams from 
[Hassan Fawaz](https://github.com/hfawaz/cd-diagram).