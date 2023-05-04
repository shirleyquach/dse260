#!/bin/bash
declare -a bad_models=("id-00000184" "id-00000599" "id-00000858" "id-00001088")
for model in "${bad_models[@]}"
do 
    echo $model
    rm -rf ~/round1-dataset-train/models/$model
    echo "Removed"
done
