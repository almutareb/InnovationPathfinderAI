import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))

# Needed to import successfully
# Add the project root directory to the Python path
sys.path.append(project_root)