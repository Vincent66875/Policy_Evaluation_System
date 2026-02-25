🚀 Persona Research Environment Setup

This repository contains the scripts and configuration needed to evaluate privacy policies using LLMs on the Clemson Palmetto Cluster. We utilize Anaconda and Scratch Space to ensure high performance and bypass home directory storage limits.

📋 Prerequisites

Before starting, ensure you have:

A Palmetto account and an active session on a compute node.

A requirements.txt file in your project directory containing:

langchain
langchain-ollama
langchain-community
pandas
httpx
🛠️ Initial Installation

Run the setup script to build your isolated Python 3.10 environment in the 5TB /scratch zone. This keeps your /home directory clean and avoids quota issues.

1️⃣ Give permission to the setup file:
chmod +x setup_research.sh
2️⃣ Run the setup:
./setup_research.sh

Note: This script automatically loads anaconda3/2023.09-0 and creates the environment at /scratch/$USER/persona_env.

⚡ Quick Access (Daily Use)

You don't need to run the setup script every time. Once the environment is built, use this shortcut to enter it instantly.

1️⃣ Set up the Alias (Run Once)

Run this command once to create the start_persona shortcut in your bash configuration:

echo "alias start_persona='module load anaconda3/2023.09-0 && source activate /scratch/\$USER/persona_env'" >> ~/.bashrc
source ~/.bashrc
2️⃣ Activate Environment

Whenever you log in to a compute node, simply type:

start_persona

You will know it worked when your terminal prompt starts with:

(/scratch/yourname/persona_env)
