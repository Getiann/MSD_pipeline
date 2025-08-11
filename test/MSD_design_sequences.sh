#!/bin/bash
source /opt/conda/etc/profile.d/conda.sh
conda activate /data/ge/conda/envs/mpnn_env
python /home/ge/app/MSD_design/LigandMPNN_MSD/run.py \
    --seed 111 \
    --pdb_path_multi "/home/ge/app/MSD_design/test/pdbs/multi.json" \
    --out_folder "/home/ge/app/MSD_design/test/designed_sequences" \
    --multistate_design True
