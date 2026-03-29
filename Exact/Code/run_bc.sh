#!/bin/bash

# Set the directory containing the VRP files
DATA_DIR="../../Dataset/A/"

# Set common parameters
TIME_LIMIT=600
MAX_NODES=5000

# Create logs directory
LOGS_DIR="logs"
mkdir -p "$LOGS_DIR"

# Check if the directory exists
if [ ! -d "$DATA_DIR" ]; then
    echo "Error: Directory $DATA_DIR does not exist"
    exit 1
fi

# Loop through all .vrp files in the directory
for vrp_file in "$DATA_DIR"*.vrp; do
    # Check if there are any .vrp files
    if [ ! -f "$vrp_file" ]; then
        echo "No .vrp files found in $DATA_DIR"
        exit 1
    fi
    
    # Get the filename without path
    filename=$(basename "$vrp_file")
    log_file="$LOGS_DIR/${filename%.vrp}.log"
    
    echo "========================================="
    echo "Processing: $vrp_file"
    echo "Log file: $log_file"
    echo "========================================="
    
    # Run the Python script and save output to log file
    python cvrp_branch_cut.py "$vrp_file" --time_limit "$TIME_LIMIT" --max_nodes "$MAX_NODES" > "$log_file" 2>&1
    
    # Check exit status
    if [ $? -eq 0 ]; then
        echo "Successfully processed: $vrp_file"
    else
        echo "Error processing: $vrp_file (see $log_file)"
    fi
    echo ""
done

echo "All files processed! Logs saved in $LOGS_DIR/"