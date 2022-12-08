# codem_eval

## Overview
This repository contains script to test and evaluate the efficacy of `codem` at co-registering 3D data sets. `codem_eval` randomly crops and transforms an input AOI point cloud, registers the point cloud to a foundational data set using `codem`, and calculates pre- and post-registration error metrics. The `hydra` package is used to organize output files and keep records of the configurations used for each run.

### Respository structure:

```
codem_eval
├── src
│   ├── conf
│       └── config.yaml # contains parameters to edit
|   ├── config.py # contains data classes built from parameters in config.yaml
|   └── main.py # main source code to run
└── environment.yaml # use to create environment with appropriate dependences
```

### `main.py` outline
1. Prepare environment
2. Link configuration
	- Link `config.yaml` and `config.py` to `main.py`
3. Write new .csv for output
	- Write .csv titled with date and time of run
	- Stored in folder created by `hydra` package denoting the date and time
4. Transform data
	- Randomly select transformation values
	- Resample the AOI
	- Crop the AOI
	- Apply the transformation
5. Registration with `codem`
	- Pre-processing
	- Coarse (DSM) registration
	- Fine (ICP) registration
	- Apply registration
6. Calculate results
	- Calculate pre- and post-registration RMSE and Fontana score
	- Call `codem` registration parameters (coarse and fine registration)
	- Call `codem` configuration parameters
7. Write results to .csv

## User steps
1. Clone the repo:
```
git clone https://github.com/hannahjhein/codem_eval.git
```

2. Create and activate a Conda environment containing the required dependences. From inside the `codem_eval` directory:
```
conda env create --file environment.yaml
```
```
conda activate codem_eval
```

3. Open config.yaml in the conf folder and edit the parameters as desired. These parameters include:

	Options
	- `reps`
		- number of repetitions for the data transformation, registration, and evaluation loop
		- integer
	- `resample_aoi`
		- option to resample the complement AOI point cloud, **default = True**
		- True = resample based on randomly generated sampling radius from range defined in `config.yaml`
		- False = do not resample, maintain original AOI point density
	- `crop_aoi`
		- option to crop the complement AOI point cloud, **default = True**
		- True = crop complement AOI to square with side length randomly generated from range defined by `min_len` parameter and the maximum side length of the input AOI
		- False = do not crop, maintain original AOI dimensions
	- `transform_aoi`
		- option to apply random translations and rotations to complement AOI point cloud, **default = True**
		- True = transform the complement AOI with values randomly generated from range of XYZ rotations and XYZ translations as defined in `params` in `config.yaml`
		- False = do not apply any artificial rotation or translation to the complement AOI
	- `register_aoi`
		- option to use `codem` to register the complement AOI point cloud to a foundational DEM or foundational point cloud, **default = True**
		- True = register the complement AOI to the foundational data set defined in `files` in `config.yaml`
		- False = do not register data sets, simply output the unregistered file(s)
	- `remomve_files`
		- option to remove generated files after metrics are calculated and written to .csv, **default = True**
		- True = remove resampled, cropped, transformed, and registered files output by the `codem_eval` program
			- use this option if your main purpose is to run a lot of iterations and don't want to worry about output file storage
		- False = do not remove any generated files output by `codem_eval`
			- use this option if your main purpose is to resample/crop/transform data sets and want to keep perturbed file(s)
		- *note: codem will output the registered file regardless of the option chosen for this field*
	
	Params
	- `lo_radius` & `hi_radius` 
		- range of radius for resampling
		- in meters, float
	- `lo_angle` & `hi_angle`
		- range of rotations to be applied to AOI point cloud in x, y, and z
		- in radians, float
		- if no rotation desired in a particular dimension, set `lo_angle` and `hi_angle` to 0.0
	- `lo_trans` & `hi_trans`
		- range of translations to be applied to AOI point cloud in x, y, and z
		- in meters, float
		- if no translation desired in a particular dimension, set `lo_trans` and `hi_trans` to 0.0
	- `min_len`
		- the minimum length of the side for the cropping bounding box
		- in meters, integer

	Files
	- `csv`
		- prefix for the output .csv name, **default = codem_eval_**
		- string
		- *note: date and time will be automatically applied to .csv name following the prefix designated in this field*
	- `input_comp_data`
		- full name of the input complement point cloud file
		- string
	- `input_found_data`
		- full name of the foundational point cloud or DEM to be used in registration
		- string
		- *note: if no registration desired (i.e.* `register_aoi` *= False), leave this field blank*
	- `output_prefix`
		- prefix to apply to output data set
		- suggestion to use location tag followed by underscore (e.g. MUTC_)
		- string
	
	Paths
	- `input_data_path`
		- file pathway to folder the input complement AOI point cloud and foundatioanl data sets
		- string

4. Run 
```
python main.py hydra.job.chdir=True
``` 
in from the `src` folder in the `codem_eval` conda environment

5. Open the output .csv to view the random transformations applied to each iteration and the pre- and post-registration metrics

## Output structure
Given an input data set named "complement.las" with a sample radius of X meters:

```
codem_eval
├── src
|   ├── conf
│       └── config.yaml
|   ├── data
|   ├── outputs
│       └── 2022-11-01
│           └── 17-14-24
│               ├── .hydra
│               │   ├── config.yaml # copy of config file passed to function
│               │   ├── hydra.yaml # copy of hydra config file
│               │   └── overrides.yaml # copy of arguments provided through command line
│               ├── registration_2022-11-01_17-14-24
│               │   ├── config.yml # record of parameters used in registration
│               │   ├── log.txt # log file
│               │   ├── registration.txt # record of coarse and fine registration transformation parameters and some statistics
│               │   ├── dsm_feature_matches.png # image of the tie points used in coarse registration step
│               │   └── complement_Xm_perturb_registered.las # the registered point cloud
│               ├── main.log # logger output
│               └── output.csv # .csv which contains all the randomly generated transformation parameters and error metrics          
|   ├── config.py
|   └── main.py
└── environment.yaml
```


