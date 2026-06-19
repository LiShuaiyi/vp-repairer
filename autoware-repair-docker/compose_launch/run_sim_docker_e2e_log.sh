#!/bin/bash

clear

current_dir=$(basename "$PWD")
if [ "$current_dir" != "compose_launch" ]; then
    echo "This script must be run from the compose_launch directory: "
    echo "commonroad-repairer/autoware-repair-docker/compose_launch/"
    exit 1
fi

timestamp=$(date +"%Y-%m-%d_%H-%M-%S")
log_dir="../../output/crrepairer2autoware/docker_logs"
log_file="$log_dir/$timestamp.log"
log_file_absolute=$(realpath $log_file)

if [ ! -d "$log_dir" ]; then
    mkdir -p "$log_dir"
    mkdir -p "$log_dir/../figures"
fi

cleanup() {
    echo "Stopping Docker Compose..."
    docker compose -f e2e.yml --env-file .env stop --timeout 1
    csv_path_trajectory=$(realpath $(find ../../output/crrepairer2autoware/data_logs/ -type f -name "*.csv" -print0 | xargs -0 ls -t | head -n1))
    if [ -z "$csv_path_trajectory" ]; then
        echo "No CSV file found in the specified directory."
        exit 1
    fi
    echo "------------------------------------"
    echo ">>> Docker log: $log_file_absolute"
    echo "------------------------------------"
    cd ../..
    csv_filename=$(basename "$csv_path_trajectory")
    crrepair_log_file_timestamp_formatted=${csv_filename:0:22}
    python3 crrepairer/repairer_cr2autoware/crrepair_wrapper.py $crrepair_log_file_timestamp_formatted

    # Collect log files in a separate folder
    mkdir -p output/crrepairer2autoware_collected_copy/$crrepair_log_file_timestamp_formatted
    cd output/crrepairer2autoware_collected_copy/$crrepair_log_file_timestamp_formatted
    mkdir -p data_logs docker_logs figures scenario
    cp -r ../../crrepairer2autoware/data_logs/$crrepair_log_file_timestamp_formatted* ./data_logs/
    cp -r ../../crrepairer2autoware/docker_logs/$timestamp.log ./docker_logs/
    cp -r ../../crrepairer2autoware/figures/$crrepair_log_file_timestamp_formatted* ./figures/
    cp -r ../../crrepairer2autoware/scenario/$crrepair_log_file_timestamp_formatted* ./scenario/
    echo "------------------------------------"
    cd ../../.. # go back to the root directory
    log_file_collected_copy_absolute=$(realpath output/crrepairer2autoware_collected_copy/$crrepair_log_file_timestamp_formatted/docker_logs/$timestamp.log)
    # Delete all occurrences of ANSI color escape characters in the log files
    sed -i 's/\x1b\[[0-9;]*m//g' "$log_file_absolute"
    sed -i 's/\x1b\[[0-9;]*m//g' "$log_file_collected_copy_absolute"
    # Delete unnecessary log information
    sed -i '/acceleration_costs:/d' "$log_file_absolute"
    sed -i '/desired_velocity_costs:/d' "$log_file_absolute"
    sed -i '/distance costs:/d' "$log_file_absolute"
    sed -i '/lateral_jerk_costs:/d' "$log_file_absolute"
    sed -i '/steering_velocity_costs:/d' "$log_file_absolute"
    sed -i '/final costs:/d' "$log_file_absolute"

    sed -i '/acceleration_costs:/d' "$log_file_collected_copy_absolute"
    sed -i '/desired_velocity_costs:/d' "$log_file_collected_copy_absolute"
    sed -i '/distance costs:/d' "$log_file_collected_copy_absolute"
    sed -i '/lateral_jerk_costs:/d' "$log_file_collected_copy_absolute"
    sed -i '/steering_velocity_costs:/d' "$log_file_collected_copy_absolute"
    sed -i '/final costs:/d' "$log_file_collected_copy_absolute"

    log_file_collected_copy_dir=$(dirname $(dirname $log_file_collected_copy_absolute))
    echo ">>> Log files collected to: $log_file_collected_copy_dir"
    cd autoware-repair-docker/compose_launch
    exit 0
}

trap cleanup SIGINT

docker compose -f e2e.yml --env-file .env up 2>&1 | tee "$log_file"
