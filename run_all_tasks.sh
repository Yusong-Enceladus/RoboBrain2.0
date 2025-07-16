#!/bin/bash

# =================================================================
# RoboBrain 2.0 - Inference Demo Script
#
# This script runs inference for all supported tasks:
# 1. Grounding: Provide bounding box for a described region.
# 2. Pointing: Identify specific coordinates based on a query.
# 3. Affordance: Predict possible interaction areas for a robot.
# 4. Trajectory: Generate a sequence of points for a task.
#
# Results with visual annotations will be saved in the `result/` directory.
# =================================================================

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Task 1: Grounding ---
echo "Running Task 1: Grounding..."
python inference.py \
  --image ./assets/demo/grounding.jpg \
  --prompt "the yellow sponge" \
  --task grounding \
  --plot

# --- Task 2: Pointing ---
echo "Running Task 2: Pointing..."
python inference.py \
  --image ./assets/demo/pointing.jpg \
  --prompt "Point to the red apple" \
  --task pointing \
  --plot

# --- Task 3: Affordance ---
echo "Running Task 3: Affordance..."
python inference.py \
  --image ./assets/demo/affordance.jpg \
  --prompt "the handle of the pot" \
  --task affordance \
  --plot

# --- Task 4: Trajectory ---
echo "Running Task 4: Trajectory..."
python inference.py \
  --image ./assets/demo/trajectory.jpg \
  --prompt "place the red bowl into the top drawer" \
  --task trajectory \
  --plot

echo "All tasks completed. Results are saved in the 'result/' directory." 