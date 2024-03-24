#!/bin/bash

THIS_DIR=$(dirname $0)
LOG_FILE="$THIS_DIR/metrics.log"

# while loop sleep every 30 seconds
while true; do
    echo "logging..."
    echo "-----" >> $LOG_FILE
    date >> $LOG_FILE

    echo "Temperature:" >> $LOG_FILE
    sensors >> $LOG_FILE

    echo "Power Usage:" >> $LOG_FILE
    # segmentation fault ...
    # sudo powertop --time=1 --csv=/dev/stdout | tail -n 1 >> $LOG_FILE
    sudo timeout 3s powertop >> $LOG_FILE 2>&1

    echo "nvidia usage:" >> $LOG_FILE
    nvidia-smi >> $LOG_FILE

    echo "" >> $LOG_FILE

    sleep 30
done
