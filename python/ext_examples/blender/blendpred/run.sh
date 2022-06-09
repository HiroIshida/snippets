#!/bin/bash

OUT_DIR="./out"

RESOLUTION=100
SAMPLINGS=128
ANIM_FRAMES_OPTION="--render-anim"

# Make this "true" when testing the scripts
TEST=false
if ${TEST}; then
  RESOLUTION=10
  SAMPLINGS=16
  ANIM_FRAMES_OPTION="--render-frame 1..5"
fi

# Create the output directory
mkdir -p ${OUT_DIR}

blender --background -noaudio --python ./example.py --render-frame 1 -- ${OUT_DIR}/07_texturing_ ${RESOLUTION} ${SAMPLINGS}
