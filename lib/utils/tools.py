import os
from datetime import datetime
import re

from lib import PATH

def make_unique_output_dir(base_dir: str) -> str:
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
        return base_dir
    else:
        timestamp = datetime.now().strftime("ver%y%m%d_%H%M")
        new_dir = f"{base_dir}_{timestamp}"
        os.makedirs(new_dir)
        return new_dir
    
def extract_sort_key(name: str):
    # Match prefix + numeric part (optional 'p' decimal), optional 'm' at end
    match = re.match(r"(.+?)(\d+p?\d*)(m)?$", name)
    if match:
        prefix, number_str, _ = match.groups()
        number = float(number_str.replace('p', '.'))
        return (prefix, number)
    else:
        return (name, float('inf'))

def path_debug_test(): 
    print(PATH)
    pass