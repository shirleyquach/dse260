This repo contains a working example for a submission to the [TrojAI leaderboard](https://pages.nist.gov/trojai/). 
This "solution" loads the model file, extracts its weights, and transform these
weights into a set of features. The features are extracted by flattening every layer and 
applying Kernel PCA to each architecture and [FastICA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.FastICA.html#sklearn.decomposition.FastICA) 
to fit a XGBoost Classifier.

--------------
# Table of Contents
* [Using the detector](#using-the-detector)
* [Reconfiguration Mode](#reconfiguration-mode)
* [Inferencing Mode](#inferencing-mode)
* [System Requirements](#system-requirements)
* [Example Data](#example-data)
* [How to Build this Detector](#how-to-build-this-detector)
  * [Install Anaconda Python](#install-anaconda-python)
  * [Set up the Conda Environment](#set-up-the-conda-environment)
  * [Train and Run Detector](#train-and-run-detector)

--------------
# Using the Detector

You will need to modify at least 1 directory:
* learned_parameters/: Directory containing data created at training time (that can be 
  changed with re-training the detector)

The detector class (in detector.py) needs to implement 4 methods to work properly: 
* `__init__(self, metaparameter_filepath, learned_parameters_dirpath)`: The initialization
function that loads the metaparameters from the given file path, and 
learned_parameters if necessary.
* `automatic_configure(self, models_dirpath)`: A function that re-configures 
the detector by performing a grid search on a preset range of meta-parameters. This 
function automatically changes the meta-parameters, call `manual_configure` and 
output a new meta-parameters.json file (in the learned_parameters folder) when optimal 
meta-parameters are found.   
* `manual_configure(self, models_dirpath)`: A function that re-configure (re-train) the 
detector given a metaparameters.json file. 
* `infer(self, model_filepath, result_filepath, scratch_dirpath, examples_dirpath, round_training_dataset_dirpath)`: Inference
function to detect if a particular model is poisoned (1) or clean (0).

--------------
# Reconfiguration Mode

Executing the `entrypoint.py` in reconfiguration mode will produce the necessary metadata for your detector and save them into the specified "learned_parameters" directory.

Example usage for one-off reconfiguration:
   ```bash
  python entrypoint.py configure \
  --scratch_dirpath <scratch_dirpath> \
  --metaparameters_filepath <metaparameters_filepath> \
  --schema_filepath <schema_filepath> \
  --learned_parameters_dirpath <learned_params_dirpath> \
  --configure_models_dirpath <configure_models_dirpath>
   ```

Example usage for automatic reconfiguraiton:
   ```bash
   python entrypoint.py configure \
    --scratch_dirpath <scratch_dirpath> \
    --metaparameters_filepath <metaparameters_filepath> \
    --schema_filepath <schema_filepath> \
    --learned_parameters_dirpath <learned_params_dirpath> \
    --configure_models_dirpath <configure_models_dirpath> \
    --automatic_configuration
   ```

# Inferencing Mode

Executing the `entrypolint.py` in infernecing mode will output a result file that contains whether the model that is being analyzed is poisoned (1.0) or clean (0.0).

Example usage for inferencing:
   ```bash
   python entrypoint.py infer \
   --model_filepath <model_filepath> \
   --result_filepath <result_filepath> \
   --scratch_dirpath <scratch_dirpath> \
   --examples_dirpath <examples_dirpath> \
   --round_training_dataset_dirpath <round_training_dirpath> \
   --metaparameters_filepath <metaparameters_filepath> \
   --schema_filepath <schema_filepath> \
   --learned_parameters_dirpath <learned_params_dirpath>
   ```


--------------
# System Requirements

- Linux (tested on Ubuntu 20.04 LTS)

Note: This example assumes you are running on a version of Linux (like Ubuntu 20.04 LTS). 

--------------
# Example Data

Example data can be downloaded from the NIST [Leader-Board website](https://pages.nist.gov/trojai/). 

--------------
# How to Build this Detector

## Install Anaconda Python

[https://www.anaconda.com/distribution/](https://www.anaconda.com/distribution/)

## Set up the Conda Environment

1. `conda create --name trojai-example python=3.8 -y` ([help](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html))
2. `conda activate trojai-example`
3. Install required packages into this conda environment

    - `conda install cuda -c "nvidia/label/cuda-11.6.2"`
    - `conda install pytorch=1.13.1 -c pytorch`
    - `pip install -r requirements.txt`

## Train and Run Detector

1. Test the detector to confirm pytorch is set up correctly and can utilize the GPU.

    ```bash
    python entrypoint.py infer \
   --model_filepath ./model/id-00000002/model.pt \
   --result_filepath ./scratch/output.txt \
   --scratch_dirpath ./scratch \
   --examples_dirpath ./model/id-00000002/clean-example-data \
   --round_training_dataset_dirpath /path/to/train-dataset \
   --learned_parameters_dirpath ./learned_parameters \
   --metaparameters_filepath ./metaparameters.json \
   --schema_filepath=./metaparameters_schema.json \
   --scale_parameters_filepath ./scale_params.npy
    ```

    Example Output:
    
    ```bash
    Trojan Probability: 0.07013004086445151
    ```

2. Test self-configure functionality, note to automatically reconfigure should specify `--automatic_configuration`.

    ```bash
    python entrypoint.py configure \
    --scratch_dirpath=./scratch/ \
    --metaparameters_filepath=./metaparameters.json \
    --schema_filepath=./metaparameters_schema.json \
    --learned_parameters_dirpath=./new_learned_parameters/ \
    --configure_models_dirpath=/path/to/new-train-dataset \
    --scale_parameters_filepath ./scale_params.npy
    ```

    The tuned parameters can then be used in a regular run.

    ```bash
    python entrypoint.py infer \
    --model_filepath=./model/id-00000002/model.pt \
    --result_filepath=./output.txt \
    --scratch_dirpath=./scratch/ \
    --examples_dirpath=./model/id-00000002/clean-example-data/ \
    --round_training_dataset_dirpath=/path/to/training/dataset/ \
    --metaparameters_filepath=./new_learned_parameters/metaparameters.json \
    --schema_filepath=./metaparameters_schema.json \
    --learned_parameters_dirpath=./new_learned_parameters/ \
    --scale_parameters_filepath ./scale_params.npy
    ```
