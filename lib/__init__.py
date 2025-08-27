# Defines a shared PATH you can import everywhere.
# Here we point PATH to the *package root* (the my_project/ folder).
import os
#===============================#
# Get the directory where the script is located
PATH = os.path.dirname(os.path.abspath(__file__))
# Get the project root directory
PATH = os.path.abspath(os.path.join(PATH, '..'))
#===============================#

__all__ = ["PATH"]
