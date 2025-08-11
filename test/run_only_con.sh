#!/bin/bash
source /opt/conda/etc/profile.d/conda.sh
conda activate /data/ge/conda/envs/alphafold3
INPUT_JSON="/home/ge/app/MSD_design/test/run_only_con/input_json"
INPUT_PDB="/home/ge/app/MSD_design/test/designed_sequences/packed"
for pdb_file in "$INPUT_PDB"/*.pdb; do
    filename=$(basename "$pdb_file" .pdb)
    json_file="$INPUT_JSON/$filename.json"
    if [[ -f "$json_file" ]]; then
        echo "Processing $filename"
        python /home/ge/app/MSD_design/run_only_confidence1.py \
        --json_path="$json_file" \
        --model_dir="/data/share/alphafold3" \
        --output_dir="/home/ge/app/MSD_design/test/run_only_con/outputs" \
        --max_template_date="9999-01-01" \
        --run_data_pipeline=False \
        --structure_pdb_path="$pdb_file" 
    else
        echo "JSON file $json_file does not exist for $filename"
    fi
done
