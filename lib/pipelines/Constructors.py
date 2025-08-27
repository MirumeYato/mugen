import numpy as np
import pandas as pd

from abc import ABC, abstractmethod
##################################
#        results packing
##################################
#
# We want to see all results
# as real target comparison
# (of course it would not work 
# for synth data, because for
# this time we do not have any 
# info about its real target)
#
##################################

class BaseDataConstructor(ABC):
    """
    This class helps you to make an correct post processing of trained model and used data. 
    (in case of random choosing train and test you want to remember which data model have never seen)

    1) After training you want to look at cross validation or more simplier tests on test data.
    2) You want to save all test data for future checks
    """
    def __init__(self, name, y_pred, data, additional, optional_data: pd.DataFrame=None):
        self.name = name
        self.y_pred = y_pred
        self.y_test = data[3]
        [self.slope_test, self.music_test], self.lenth, [__, self.rtt_test] = additional
        self.optional_data = optional_data

    @abstractmethod
    def get_nn(self):
        pass
    
    @abstractmethod
    def get_slope(self):
        pass

    @abstractmethod
    def get_music(self):
        pass

    @abstractmethod
    def construct_dataFrame(self, N:int):
        pass

    def process(self):
        print(f"DEBUG: Using {self.__class__.__name__} for data construction.")
        flag_res_check = True

        mae_nn = self.get_nn()
        if any([x is None for  x in self.slope_test]): flag_res_check = False
        else: mae_slope = self.get_slope()
        if any([x is None for  x in self.music_test]): flag_res_check = False
        else: mae_music = self.get_music()

        if flag_res_check: mae = mae_nn, mae_slope, mae_music
        else: mae = mae_nn, None, None

        df = self.construct_dataFrame(len(mae_nn))

        return mae, df, flag_res_check
    
class  TimeDataConstructor(BaseDataConstructor):
    """
    y_pred here is delta_t in units of real target, trained on some  delta_t in units of real target
    y_test here is delta_t in units of real target, that we've got from real delta_t = (real target - rtt)
    X shape should be (2, 256) 
    """    
    def get_nn(self):
        return np.abs(self.y_pred - self.y_test) # here y_test and predict is  delta_ts in units of time_stamps
    
    def get_slope(self):
        return np.abs(self.slope_test - (self.y_test + self.rtt_test)) # y_test =  delta_t in units of real target. Convert to real target by rtt

    def get_music(self):
        return np.abs(self.music_test - (self.y_test + self.rtt_test))

    def construct_dataFrame(self, N:int): 
        df = pd.DataFrame({
                "dataset": [self.name] * N,
                "nn_predict": list(self.y_pred.T), #  delta_t in units of real target
                "target": self.y_test,             #  delta_t in units of real target
                "slope": self.slope_test,
                "music": self.music_test,
                "rtt": self.rtt_test
            })
        return df
    
class TargetDataConstructor(BaseDataConstructor):
    """
    y_pred here is real target in its units, trained on some real target in its units
    y_test here is real target in its units, that we've got from real target
    X shape should be (2, 256) 
    """
    def get_nn(self):
        return np.abs(self.y_pred - self.y_test) # here y_test and predict is real target in corresponding units
    
    def get_slope(self):
        return np.abs(self.slope_test - self.y_test)

    def get_music(self):
        return np.abs(self.music_test - self.y_test)

    def construct_dataFrame(self, N:int): 
        df = pd.DataFrame({
                "dataset": [self.name] * N,
                "nn_predict": list(self.y_pred.T), # real target
                "target": self.y_test,             # real target
                "slope": self.slope_test,
                "music": self.music_test,
                "rtt": None
            })
        return df