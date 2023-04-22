#!/bin/bash
# Configure / Train 
# Be sure to change the models_dirpath for the appropriate directory for training
python entrypoint.py configure \
--scratch_dirpath=./scratch/ \
--metaparameters_filepath=./metaparameters.json \
--schema_filepath=./metaparameters_schema.json \
--learned_parameters_dirpath=./new_learned_parameters/ \
--configure_models_dirpath=/home/ec2-user/round1-dataset-train \
--scale_parameters_filepath ./scale_params.npy \
--automatic_configuration
