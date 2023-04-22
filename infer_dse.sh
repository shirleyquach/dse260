#!/bin/bash
# Change round_training_dataset_dirpath and model_filepath as needed
#
python entrypoint.py infer \
--model_filepath /home/ec2-user/round1-holdout-dataset \
--result_filepath ./holdout_results.txt \
--scratch_dirpath ./scratch/ \
--examples_dirpath=./model/id-00000002/clean-example-data/ \
--round_training_dataset_dirpath /home/ec2-user/round1-holdout-dataset \
--metaparameters_filepath ./new_learned_parameters/metaparameters.json  \
--schema_filepath ./metaparameters_schema.json \
--learned_parameters_dirpath ./new_learned_parameters/ \
--scale_parameters_filepath ./scale_params.npy
