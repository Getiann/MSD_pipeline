#!/bin/bash
source "/venv/alphafold3_venv/bin/activate"
python /home/ge/app/MSD_design/run_alphafold.py \
    --input_dir="/home/ge/app/MSD_design/test/run_af3/input_json" \
    --model_dir="/data/share/alphafold3"  \
    --output_dir="/home/ge/app/MSD_design/test/run_af3/outputs" \
    --max_template_date="9999-01-01" \
    --run_data_pipeline=False