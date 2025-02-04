#!/bin/bash

arch_file="config.yaml_arch"
base_file="config.yaml_base"
config_file="config.yaml"
python_script="example.py"

# Initialize the variable to store arch line
arch_line=""

# Process the arch file line by line
while IFS= read -r line || [ -n "$line" ]; do
    # Check if the line contains '#arch:'
    if [[ "$line" == '#arch:'* ]]; then
        # Store the uncommented arch line
        arch_line="${line#'#'}"
    elif [[ "$line" == '#arch_args:'* ]]; then
	echo "$arch_line with $line"
        # Uncomment arch_args line and create a new config file
        echo "$arch_line" > "$config_file"
        echo "${line#'#'}" >> "$config_file"
        
        # Append the contents of the base file to the config file
        cat "$base_file" >> "$config_file"
        
        # Invoke the Python script
        python "$python_script" -l labeled_anomalies.csv
    fi
done < "$arch_file"
