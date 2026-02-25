#!/bin/bash
# setup_research.sh

# 1. Load the Anaconda module
module purge
module load anaconda3/2023.09-0

# 2. Define the scratch path
ENV_PATH="/scratch/$USER/persona_env"

# Initialize conda for the script shell
eval "$(conda shell.bash hook)"

# 3. Create a clean environment if it doesn't exist
if [ ! -d "$ENV_PATH" ]; then
    echo "Building a fresh Python 3.10 environment in scratch space..."
    conda create --prefix "$ENV_PATH" python=3.10 -y
else
    echo "Environment already exists at $ENV_PATH."
fi

# 4. Activate and install
source activate "$ENV_PATH"
pip install --upgrade pip
pip install -r requirements.txt

echo "------------------------------------------------"
echo " Environment Ready!"
echo "Current Python: $(python -V)"
echo "------------------------------------------------"
