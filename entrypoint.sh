#!/usr/bin/env bash

# Default to starting the webUI
CLI_ARGS="${CLI_ARGS:-webUI.py}"

# Entrypoint used to start the application in a container

# Start the virtual environment
source venv/bin/activate

# If install has not been run to download the models, run it
if [ ! -f models/.installed ]; then
  echo "models/.installed not found, running install"
  python install.py
  touch models/.installed
  echo "install.py finished"
fi

# Start the application
python "$CLI_ARGS"
