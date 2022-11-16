# codem-hydra

## Overview
This repository contains script to transform an input AOI point cloud, register the point cloud to a foundational data set using `codem`, and calculate pre- and post-registration error metrics. The `hydra` package is used to organize output files and keep records of the configurations used for each run.

Respository structure:

```
src
├── conf
│   └── config.yaml # contains parameters to edit
├── data # put input data here, or link to pre-existing folder using config.yaml
├── config.py # contains data classes built from parameters in config.yaml
└── main.py # main source code to run
environment.yaml # use to create environment with appropriate dependences
```

## User steps
1. Clone the repo:
```
git clone https://github.com/hannahjhein/codem-hydra
```

2. Create and activate a Conda environment containing the required dependences. From inside the `codem-hydra` directory:
```
conda env create --file environment.yaml
```
```
conda activate codem-hydra
```

3. Open config.yaml in the conf folder and edit the parameters as desired. These parameters include:

	Params
	- range of radius for resampling (in m)
	- range of rotations to be applied to AOI point cloud in x, y, and z (in radians)
	- range of translations to be applied to AOI point cloud in x, y, and z (in m)
	- number of repetitions for the data transformation and evaluation loop

	Files
	- name of the output .csv with transformation parameters and error metrics
	- name of the input complement point cloud to be transformed and registered
	- name of the input foundational data set
	- prefix to apply to the output point clouds (e.g. location abbreviation such as MUTC)
	- suffix to apply to the output point clouds (suggest keeping the same as the input)
	
	Paths
	- file pathway to the input data sets

4. Run 
```
python main.py hydra.job.chdir=True
``` 
in the `codem-hydra` conda environment

5. Open the output .csv to view the random transformations applied to each iteration and the pre- and post-registration metrics

## Output structure
Given an input data set named "complement.las" with a sample radius of X meters:

```
src
├── conf
│   └── config.yaml
├── data
├── outputs
│   └── 2022-11-01
│       └── 17-14-24
│           ├── .hydra
│           │   ├── config.yaml # copy of config file passed to function
│           │   ├── hydra.yaml # copy of hydra config file
│           │   └── overrides.yaml # copy of arguments provided through command line
│           ├── registration_2022-11-01_17-14-24
│           │   ├── config.yml # record of parameters used in registration
│           │   ├── log.txt # log file
│           │   ├── registration.txt # record of coarse and fine registration transformation parameters and some statistics
│           │   ├── dsm_feature_matches.png # image of the tie points used in coarse registration step
│           │   └── complement_Xm_perturb_registered.las # the registered point cloud
│           ├── main.log # logger output
│           ├── complement_Xm_truth.las # the resampled, untransformed point cloud
│           ├── complement_Xm_perturb.las # the resampled and transformed point cloud
│           ├── complement_Xm_registered.las # the registered point cloud
│           └── output.csv # .csv which contains all the randomly generated transformation parameters and error metrics          
├── config.py
└── main.py
environment.yaml
```


