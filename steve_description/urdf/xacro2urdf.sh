#!/bin/bash

# Navigate to the directory containing the xacro file
cd "$(dirname "$0")"

# Execute the xacro command to generate the URDF file
xacro steve.urdf.xacro -o steve_ability.urdf

# Print a message indicating the process is complete
echo "URDF file generated: steve_ability.urdf"