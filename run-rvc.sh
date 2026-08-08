#!/bin/bash
source /opt/miniforge/etc/profile.d/conda.sh
conda activate rvc
cd /home/ajimu/software/RVC

export HSA_OVERRIDE_GFX_VERSION=11.0.0
export MIOPEN_FIND_MODE=2
export PYTHONMALLOC=malloc
export MALLOC_CHECK_=0
export ALSA_PCM_CARD=default
export PA_ALSA_PLUGHW=1

python realtime_gui.py
