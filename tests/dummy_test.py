# Path settings
import os, sys
#===============================#
# Get the directory where the script is located
PATH = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory of the current directory
PATH = os.path.abspath(os.path.join(PATH, '..'))
sys.path.insert(0, PATH)
#===============================#

from lib.utils import path_debug_test

if __name__ == "__main__":
    path_debug_test()


