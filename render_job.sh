#!/bin/bash -l
#$ -P medaihack
#$ -N fs_render
#$ -l h_rt=06:00:00
#$ -pe omp 4
#$ -l gpus=1
#$ -l gpu_c=8.0
#$ -j y
#$ -o render_job.log

cd /projectnb/medaihack/team4/ferrosight
module load python3
source frroSig/bin/activate
export OMP_NUM_THREADS=3
python -m ferrosight.datagen.generate --out data/synth --n 20 --render-only \
    --blender $PWD/blender-4.2.3-linux-x64/blender
