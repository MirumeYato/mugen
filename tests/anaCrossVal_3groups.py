import os
import pandas as pd

# Path settings
import sys
#===============================#
# Get the directory where the script is located
PATH = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory of the current directory
PATH = os.path.abspath(os.path.join(PATH, '..'))
sys.path.insert(0, PATH)
#===============================#

from lib.training import ST
from lib.data import Data
from lib.data import fold_3groups
from lib.data import name_list_3groups
from models.model_configurations import oneway_model

from lib.pipelines import TarDC
from lib.pipelines.Unpackers import TargetDataUnpacker as TDU


def groups_test(run_name, dataset_name, create_model, unpack, pack, create_folding, name_list):
    """
    Simple test.
        This model is general example of how you can use :func:``fold_3groups`` 
        and model :func:``oneway_model`` for predicting real target from only vectored fetures (without time stamps)
    """
    eph = 1

    # Here you cna choose data for train and test
    df:pd.DataFrame = Data()(dataset_name, 'dummy_data.pkl')

    # here you choose folding method
    folds = create_folding(df)
    
    # here we define model tester
    tester = ST(
    folds=folds,
    df=df,    
    name_list=name_list,            # needs if you want correct naming incide folding plots
    create_model_fn=create_model,   # your model configuration
    input_shape=(2, 256),           # All models should work with this shape. If not fix shape incide create_model function (NOT HERE!)
    name=run_name,                  # output directory name
    epochs=eph,
    aug_params=None                 # if you do not want to do augmentation choose None in opposite use:
    )                               # {"num_points": 10, "mu": [0, 0], "sigma": [0.01, 0.0001, 0.0001]}

    print(f"DEBUG: model parameters are:\npath = {tester.output_dir},\nepochs = {tester.epochs}")
    os.makedirs(tester.output_dir, exist_ok=True)

    tester.run(fold_unpacking=unpack, post_processing = pack, model_save_flag=False) # main loop of training
    # print(tester.get_model_fpath())

def main():
    name_list = [name_list_3groups] # correct names of foldings for plots
    dataset_name = 'real'

    for i, fold_method in enumerate([fold_3groups]):# loop for folding metheds
            for unpack_i in [TDU]: # loop unpacking types
                groups_test(
                    f'{dataset_name}_{unpack_i.__name__[:-12]}_{fold_method.__name__[5:]}',
                    dataset_name = dataset_name, 
                    create_model = oneway_model, 
                    unpack = unpack_i, 
                    pack = TarDC, 
                    create_folding = fold_method, 
                    name_list = name_list[i]
                )

if __name__ == "__main__":
    main()