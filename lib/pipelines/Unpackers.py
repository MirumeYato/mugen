import tensorflow as tf

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod

from .tools import augment, reduplicate, to_mag_phase

##################################
#         data unpacking
##################################

from abc import ABC, abstractmethod
import numpy as np
import pandas as pd

class BaseDataUnpacker(ABC):
    """
    This class helps you to transfor your df: pd.DataFrame data into X_train, y_train, X_test, y_test, X_val, y_val e.t.c.

    Additional ouput needed for plotting and BaseDataConstructor 
    """
    def __init__(self, df: pd.DataFrame, train_ids, test_ids, aug_param=None, optional_data=None, batch_size = 64):
        self.df = df
        self.train_ids = train_ids
        self.test_ids = test_ids
        self.aug_param = aug_param
        self.optional_data = optional_data
        self.batch_size = batch_size

    @abstractmethod
    def get_targets(self):
        pass

    @abstractmethod
    def get_features(self):
        pass

    def feature_shape(self, X_train):
        return X_train.shape[0]

    def augmentation(self, X_train, y_train):
        return augment(X_train, y_train, self.aug_param)
    
    def set_type(self, obj1, obj2, obj3, obj4):
            return np.asarray(obj1,  dtype=np.float64), np.asarray(obj2,  dtype=np.float64), \
                  np.asarray(obj3,  dtype=np.float64), np.asarray(obj4,  dtype=np.float64)
    
    def get_ds(self, X_train, y_train, X_val, y_val):
        AUTOTUNE = tf.data.AUTOTUNE

        train_ds = (
            tf.data.Dataset.from_tensor_slices((X_train, y_train))
            # # Full-buffer shuffle for true epoch-level randomness
            .shuffle(buffer_size=len(X_train), reshuffle_each_iteration=True)
            .batch(self.batch_size, drop_remainder=False)   # set True if using BatchNorm & want stable batch size
            # # .cache()                              # tiny dataset → cache fits in RAM
            .prefetch(AUTOTUNE)
        )

        val_ds = (
            tf.data.Dataset.from_tensor_slices((X_val, y_val))
            # # No shuffle for validation
            .batch(self.batch_size, drop_remainder=False)
            # # .cache()
            .prefetch(AUTOTUNE)
        )
        return train_ds, val_ds

    def process(self):
        print(f"DEBUG: Using {self.__class__.__name__} for data unpacking.")
        
        y_train, y_test, additional_targets = self.get_targets()
        # test for comparison of algorithms
        slope_test = self.df.loc[self.test_ids, 'slope_est'].values # always correspond to real target
        music_test = self.df.loc[self.test_ids, 'music_est'].values

        X_train, X_test = self.get_features()

        print(f"DEBUG: feature shape is {self.feature_shape(X_train)}, target shape is {y_train.shape[0]}")
        if  self.feature_shape(X_train) != y_train.shape[0]:
            y_train = reduplicate(y_train)
            y_test = reduplicate(y_test)
            slope_test = reduplicate(slope_test)
            music_test = reduplicate(music_test)
            if additional_targets[0] is not None: additional_targets[0] = reduplicate(additional_targets[0])
            if additional_targets[1] is not None: additional_targets[1] = reduplicate(additional_targets[1])

        X_train, y_train = self.augmentation(X_train, y_train)
        X_train, y_train, X_test, y_test = self.set_type(X_train, y_train, X_test, y_test)
        
        X_val, y_val = X_test, y_test

        tain_ds, test_ds = self.get_ds(X_train, y_train, X_val, y_val)

        return [tain_ds, test_ds, X_test, y_test, X_val, y_val], \
               [[slope_test, music_test], len(self.df), additional_targets]
    
class TargetDataUnpacker(BaseDataUnpacker):
    """
    y here is real target in its units
    X shape should be (2, 256) 
    """
    def get_targets(self):
        y_train = self.df.loc[self.train_ids, 'target'].values
        y_test = self.df.loc[self.test_ids, 'target'].values
        return y_train, y_test, [None, None]  # No RTT data

    def get_features(self):
        if all(k in self.df.columns for k in ['features_incended', 'features_reflected']):
            X_train_inc = np.stack(self.df.loc[self.train_ids, 'features_incended'])
            X_train_ref = np.stack(self.df.loc[self.train_ids, 'features_reflected'])
            X_test_inc = np.stack(self.df.loc[self.test_ids, 'features_incended'])
            X_test_ref = np.stack(self.df.loc[self.test_ids, 'features_reflected'])
            return np.concatenate([X_train_inc, X_train_ref]), np.concatenate([X_test_inc, X_test_ref])
        else:
            return (np.stack(self.df.loc[self.train_ids, 'features']),
                    np.stack(self.df.loc[self.test_ids, 'features']))

class FullDataUnpacker(TargetDataUnpacker):
    """
    y here is real target in its units
    X shape should be (2, 256) 
    """
    def time_augmentation(self, t1, t2, t3, t4):
        p = self.aug_param
        delta_t21 = t2-t1 # = delta_t21 + d_sync + (shift_on_2-shift_on_1) > 0
        delta_t43 = t4-t3 # = delta_t43 - d_sync + (shift_on_4-shift_on_3) < 0
        expunded_dt21 = np.repeat(delta_t21, p["num_points"], axis=0) # > 0
        expunded_dt43 = np.repeat(delta_t43, p["num_points"], axis=0) # < 0
        uniform_scale = np.random.uniform(0, 5e13, (2, expunded_dt21.shape[0])) # time shift variation
        d_sync = np.abs(np.random.standard_cauchy((2, expunded_dt21.shape[0]))) # small syncromization noise
        return uniform_scale[0],\
            d_sync[0]*self.aug_param["sigma"][2] + uniform_scale[0] + expunded_dt21,\
            d_sync[1]*self.aug_param["sigma"][2] + uniform_scale[1] - expunded_dt43,\
            uniform_scale[1]

    def augmentation(self, doubled_features, targets):
        p = self.aug_param
        time_features, h_features = doubled_features
        if p is not None:
            augmented_obj = []
            print("DEBUG: Augmentation is used")
            for i, obj in enumerate([targets, h_features]):
                expunded_obj = np.repeat(obj, p["num_points"], axis=0)
                noise = np.random.standard_cauchy(expunded_obj.shape) * p["sigma"][i] + p["mu"][i]
                augmented_obj.append(expunded_obj + noise)
            augmented_targets, augmented_h  = augmented_obj

            return (time_features, augmented_h), augmented_targets
        else:
            print("DEBUG: No augmentation. Just shuffling")
            return doubled_features, targets
    
    def feature_shape(self, X_train):
        return X_train[0].shape[0]
    
    def get_ds(self, X_train, y_train, X_val, y_val):
        BATCH = 32
        AUTOTUNE = tf.data.AUTOTUNE
        X1_tr, X2_tr = X_train
        X1_val, X2_val = X_val

        train_ds = (
            tf.data.Dataset.from_tensor_slices(((X1_tr, X2_tr), y_train))
            # Full-buffer shuffle for true epoch-level randomness
            .shuffle(buffer_size=len(X1_tr), reshuffle_each_iteration=True)
            .batch(BATCH, drop_remainder=False)   # set True if using BatchNorm & want stable batch size
            # .cache()                              # tiny dataset → cache fits in RAM
            .prefetch(AUTOTUNE)
        )

        val_ds = (
            tf.data.Dataset.from_tensor_slices(((X1_val, X2_val), y_val))
            # No shuffle for validation
            .batch(BATCH, drop_remainder=False)
            # .cache()
            .prefetch(AUTOTUNE)
        )
        return train_ds, val_ds
    
    def times_stamp_process(self, time_stamp_1, time_stamp_2, time_stamp_3, time_stamp_4):
        return np.concatenate([np.stack([time_stamp_1, time_stamp_2], axis = 1), np.stack([time_stamp_3, time_stamp_4], axis = 1)])
    
    def set_type(self, obj1, obj2, obj3, obj4):
            x1, x2 = obj1
            obj1 = (np.asarray(x1,  dtype=np.float32), np.asarray(x2,  dtype=np.float32))
            x1, x2 = obj3
            obj3 = (np.asarray(x1,  dtype=np.float32), np.asarray(x2,  dtype=np.float32))
            return obj1, np.asarray(obj2,  dtype=np.float32), \
                  obj3, np.asarray(obj4,  dtype=np.float32)
    
    def get_times(self):
        time_stamp_1_train = np.stack(self.df.loc[self.train_ids, 'time_stamp_1']) * 0.3 # to units of real target (t[time_stamp units] * 0.3 [target\time_stamp units] = t * 10**(-9) * 0.3 * 10**9 = 0.3 t )
        time_stamp_2_train = np.stack(self.df.loc[self.train_ids, 'time_stamp_2']) * 0.3
        time_stamp_3_train = np.stack(self.df.loc[self.train_ids, 'time_stamp_3']) * 0.3
        time_stamp_4_train = np.stack(self.df.loc[self.train_ids, 'time_stamp_4']) * 0.3
        
        t1, t2, t3, t4 = self.time_augmentation(time_stamp_1_train, time_stamp_2_train, time_stamp_3_train, time_stamp_4_train)
        print('DEBUG: example of times:', t1[0], t2[0], t3[0], t4[0])
        time_step_train = self.times_stamp_process(t1, t2, t3, t4)

        time_stamp_1_test = np.stack(self.df.loc[self.test_ids, 'time_stamp_1']) * 0.3 # to units of real target (t[time_stamp units] * 0.3 [target\time_stamp units] = t * 10**(-9) * 0.3 * 10**9 = 0.3 t )
        time_stamp_2_test = np.stack(self.df.loc[self.test_ids, 'time_stamp_2']) * 0.3
        time_stamp_3_test = np.stack(self.df.loc[self.test_ids, 'time_stamp_3']) * 0.3
        time_stamp_4_test = np.stack(self.df.loc[self.test_ids, 'time_stamp_4']) * 0.3
        
        time_step_test = self.times_stamp_process(time_stamp_1_test, time_stamp_2_test, time_stamp_3_test, time_stamp_4_test)

        return time_step_train, time_step_test

    def get_features(self):
        H_train_inc = np.stack(self.df.loc[self.train_ids, 'features_incended'])
        H_train_ref = np.stack(self.df.loc[self.train_ids, 'features_reflected'])
        H_test_inc = np.stack(self.df.loc[self.test_ids, 'features_incended'])
        H_test_ref = np.stack(self.df.loc[self.test_ids, 'features_reflected'])
        H_train, H_test =  np.concatenate([H_train_inc, H_train_ref]), np.concatenate([H_test_inc, H_test_ref])

        time_step_train, time_step_test = self.get_times()

        print(f'DEBUG: shape of time_stamps_fetures are {time_step_train.shape}, shape of feature vector are {H_train.shape}')

        return (time_step_train, H_train), (time_step_test, H_test)
    
class FullTSubtractionDataUnpacker(FullDataUnpacker):
    def times_stamp_process(self, time_stamp_1, time_stamp_2, time_stamp_3, time_stamp_4):
        t2_t1 = time_stamp_2 - time_stamp_1
        t4_t3 = time_stamp_4 - time_stamp_3
        return np.concatenate([np.stack([t2_t1 / time_stamp_1, t2_t1 / time_stamp_2], axis = 1), np.stack([t4_t3 / time_stamp_3, t4_t3 / time_stamp_4], axis = 1)])
    
class FullOneDataUnpacker(FullDataUnpacker):
    def times_stamp_process(self, time_stamp_1, time_stamp_2, time_stamp_3, time_stamp_4):
        # eps = 1

        # t2_t1 = time_stamp_2 - time_stamp_1
        # # Stack along a new axis so you can average over it
        # stacked = np.stack([time_stamp_2, time_stamp_1], axis=0)   # shape (2, 4, 1)
        # t2_t1_avg = np.mean(stacked, axis=0)      # same as (a+b)/2
        # t2_t1_std  = np.std(stacked, axis=0)       # same as your sqrt(...) formula
        # t2_norm = (time_stamp_2 - t2_t1_avg)/ t2_t1_std
        # t1_norm = (time_stamp_1 - t2_t1_avg)/ t2_t1_std
        # t2_t1_log = np.log(time_stamp_2+eps) - np.log(time_stamp_1+eps)

        # t4_t3 = time_stamp_4 - time_stamp_3
        # stacked = np.stack([time_stamp_4, time_stamp_3], axis=0)   # shape (2, 4, 1)
        # t4_t3_avg = np.mean(stacked, axis=0)      # same as (a+b)/2
        # t4_t3_std  = np.std(stacked, axis=0)       # same as your sqrt(...) formula
        # t4_norm = (time_stamp_4 - t4_t3_avg)/ t4_t3_std
        # t3_norm = (time_stamp_3 - t4_t3_avg)/ t4_t3_std
        # t4_t3_log = np.log(time_stamp_4+eps) - np.log(time_stamp_3+eps)
        return np.concatenate([np.stack([time_stamp_1, time_stamp_2], axis = 1), np.stack([time_stamp_3, time_stamp_4], axis = 1)])
    
class FullFWDataUnpacker(FullDataUnpacker):
    def time_augmentation(self, t1, t2, t3, t4):
        p = self.aug_param
        return NotImplemented
    
    def times_stamp_process(self, time_stamp_1, time_stamp_2, time_stamp_3, time_stamp_4):
        return np.stack([time_stamp_1, time_stamp_2, time_stamp_3, time_stamp_4], axis = 1)
    
    def get_features(self):
        H_train_inc = np.stack(self.df.loc[self.train_ids, 'features_incended'])
        H_train_ref = np.stack(self.df.loc[self.train_ids, 'features_reflected'])
        H_test_inc = np.stack(self.df.loc[self.test_ids, 'features_incended'])
        H_test_ref = np.stack(self.df.loc[self.test_ids, 'features_reflected'])
        H_train, H_test =  np.concatenate([H_train_inc, H_train_ref], axis=1), np.concatenate([H_test_inc, H_test_ref], axis=1)

        time_step_train, time_step_test = self.get_times()

        print(f'shape of time_stamps_fetures are {time_step_train.shape}, shape of feature vector are {H_train.shape}')

        return (time_step_train, H_train), (time_step_test, H_test)

class TimeDataUnpacker(BaseDataUnpacker):
    """
    y here is delta_t in units of real target
    X shape should be (2, 256) 
    """
    def get_targets(self):
        rtt_train = self.df.loc[self.train_ids, 'rtt'].values
        rtt_test = self.df.loc[self.test_ids, 'rtt'].values
        y_train = self.df.loc[self.train_ids, 'target'].values - rtt_train
        y_test = self.df.loc[self.test_ids, 'target'].values - rtt_test
        return y_train, y_test, [rtt_train, rtt_test]

    def get_features(self):
        X_train_inc = np.stack(self.df.loc[self.train_ids, 'features_incended'])
        X_train_ref = np.stack(self.df.loc[self.train_ids, 'features_reflected'])
        X_test_inc = np.stack(self.df.loc[self.test_ids, 'features_incended'])
        X_test_ref = np.stack(self.df.loc[self.test_ids, 'features_reflected'])
        return np.concatenate([X_train_inc, X_train_ref]), np.concatenate([X_test_inc, X_test_ref])

class MusicTargetDataUnpacker(TargetDataUnpacker):
    """
    y_train here is real target in its units, that we've got from music predictions
    y_test here is real target in its units, that we've got from real distance
    X shape should be (2, 256) 
    """
    def get_targets(self):
        y_train = self.df.loc[self.train_ids, 'music_est'].values # predict real target as Music in units of real target
        y_test = self.df.loc[self.test_ids, 'target'].values      # but test on real target as real target in units of real target
        return y_train, y_test, [None, None]  # No RTT data
    
class MusicTimeDataUnpacker(TimeDataUnpacker):
    """
    y_train here is real target in its units, that we've got from music predictions
    y_test here is real target in its units, that we've got from real distance
    X shape should be (2, 256) 
    """
    def get_targets(self):
        rtt_train = self.df.loc[self.train_ids, 'rtt'].values
        rtt_test = self.df.loc[self.test_ids, 'rtt'].values
        y_train = self.df.loc[self.train_ids, 'music_est'].values - rtt_train # predict real target as Music in units of real target
        y_test = self.df.loc[self.test_ids, 'target'].values - rtt_test       # but test on real target as real target in its units
        return y_train, y_test, [rtt_train, rtt_test]  # No RTT data
    
class MagPhaseTargetDataUnpacker(BaseDataUnpacker):
    """
    y here is real target in its units
    X shape should be (2, 256) 
    We convert features from [feat_imag, feat_real] to [feat_mag, feat_phase]
    """
    def convert_feature(self, ids:np.ndarray, feature_name: str):
        return np.stack([to_mag_phase(x) for x in self.df.loc[ids, feature_name]])
    
    def get_targets(self):
        y_train = self.df.loc[self.train_ids, 'target'].values
        y_test = self.df.loc[self.test_ids, 'target'].values
        return y_train, y_test, [None, None]  # No RTT data

    def get_features(self):
        if all(k in self.df.columns for k in ['features_incended', 'features_reflected']):
            X_train_inc = self.convert_feature(self.train_ids, 'features_incended')
            X_train_ref = self.convert_feature(self.train_ids, 'features_reflected')
            X_test_inc = self.convert_feature(self.test_ids, 'features_incended')
            X_test_ref = self.convert_feature(self.test_ids, 'features_reflected')
            return np.concatenate([X_train_inc, X_train_ref]), np.concatenate([X_test_inc, X_test_ref])
        else:
            return (self.convert_feature(self.train_ids, 'features'),
                    self.convert_feature(self.test_ids, 'features'))
        
class MagPhaseTimeDataUnpacker(MagPhaseTargetDataUnpacker):
    """
    y here is delta_t in units of real target
    X shape should be (2, 256) 
    We convert features from [feat_imag, feat_real] to [feat_mag, feat_phase]
    """    
    def get_targets(self):
        rtt_train = self.df.loc[self.train_ids, 'rtt'].values
        rtt_test = self.df.loc[self.test_ids, 'rtt'].values
        y_train = self.df.loc[self.train_ids, 'target'].values - rtt_train
        y_test = self.df.loc[self.test_ids, 'target'].values - rtt_test
        return y_train, y_test, [rtt_train, rtt_test]

    def get_features(self):
        X_train_inc = self.convert_feature(self.train_ids, 'features_incended')
        X_train_ref = self.convert_feature(self.train_ids, 'features_reflected')
        X_test_inc = self.convert_feature(self.test_ids, 'features_incended')
        X_test_ref = self.convert_feature(self.test_ids, 'features_reflected')
        return np.concatenate([X_train_inc, X_train_ref]), np.concatenate([X_test_inc, X_test_ref])

class S2R_TargetDataUnpacker(BaseDataUnpacker):
    """
    y here is real target in its units
    X shape should be (2, 256) 

    Main difference is to use self.optional_data as way to validate model during training
    So we can catch when training on synthetic or other data will start losing on real data.
    """
    def get_targets(self):
        y_train = self.df.loc[self.train_ids, 'target'].values
        y_test = self.optional_data['target'].values
        return y_train, y_test, [None, None]  # No RTT data

    def get_features(self):
        if all(k in self.df.columns for k in ['features_incended', 'features_reflected']):
            X_train_inc = np.stack(self.df.loc[self.train_ids, 'features_incended'])
            X_train_ref = np.stack(self.df.loc[self.train_ids, 'features_reflected'])
            X_test_inc = np.stack(self.optional_data['features_incended'])
            X_test_ref = np.stack(self.optional_data['features_reflected'])
            return np.concatenate([X_train_inc, X_train_ref]), np.concatenate([X_test_inc, X_test_ref])
        else:
            X_test_inc = np.stack(self.optional_data['features_incended'])
            X_test_ref = np.stack(self.optional_data['features_reflected'])
            return (np.stack(self.df.loc[self.train_ids, 'features']),
                    np.concatenate([X_test_inc, X_test_ref]))
        
class LinearRTTDataUnpacker(BaseDataUnpacker):
    """
    y here is real target in its units (true)
    X here is real target in its units (rtt)
    """
    def get_targets(self):
        y_train = np.stack(self.df.loc[self.train_ids, 'target'])
        y_test = np.stack(self.df.loc[self.test_ids, 'target'])
        return y_train, y_test, [None, None]  # No RTT data

    def get_features(self):
        return np.stack(self.df.loc[self.train_ids, 'rtt']), np.stack(self.df.loc[self.test_ids, 'rtt'])
    
class LinearDataUnpacker(LinearRTTDataUnpacker):
    """
    y here is real target in its units (true)
    t1,t2,t3,t4 here is time stamps in units of real target
    """
    def feature_shape(self, X_train):
        return X_train.shape[0]
    
    # def times_stamp_process(self, time_stamp_1, time_stamp_2, time_stamp_3, time_stamp_4):
    #     return np.concatenate([np.stack([time_stamp_1, time_stamp_2], axis = 1), np.stack([time_stamp_3, time_stamp_4], axis = 1)])
    def times_stamp_process(self, time_stamp_1, time_stamp_2, time_stamp_3, time_stamp_4):
        rtt = (time_stamp_4 - time_stamp_1 - (time_stamp_3 -time_stamp_2))/2
        return rtt#np.stack([], axis = 1)
    
    def get_features(self):
        time_stamp_1_train = np.stack(self.df.loc[self.train_ids, 'time_stamp_1']) * 0.3 # to in units of real target (t[ns] * c [m\s] = t * 10**(-9) * 0.3 * 10**9 = 0.3 t )
        time_stamp_2_train = np.stack(self.df.loc[self.train_ids, 'time_stamp_2']) * 0.3
        time_stamp_3_train = np.stack(self.df.loc[self.train_ids, 'time_stamp_3']) * 0.3
        time_stamp_4_train = np.stack(self.df.loc[self.train_ids, 'time_stamp_4']) * 0.3
        
        # t1, t2, t3, t4 = self.time_augmentation(time_stamp_1_train, time_stamp_2_train, time_stamp_3_train, time_stamp_4_train)
        # print('DEBUG: example of times:', t1[0], t2[0], t3[0], t4[0])
        time_step_train = self.times_stamp_process(time_stamp_1_train, time_stamp_2_train, time_stamp_3_train, time_stamp_4_train)

        time_stamp_1_test = np.stack(self.df.loc[self.test_ids, 'time_stamp_1']) * 0.3 # to in units of real target (t[ns] * c [m\s] = t * 10**(-9) * 0.3 * 10**9 = 0.3 t )
        time_stamp_2_test = np.stack(self.df.loc[self.test_ids, 'time_stamp_2']) * 0.3
        time_stamp_3_test = np.stack(self.df.loc[self.test_ids, 'time_stamp_3']) * 0.3
        time_stamp_4_test = np.stack(self.df.loc[self.test_ids, 'time_stamp_4']) * 0.3
        
        time_step_test = self.times_stamp_process(time_stamp_1_test, time_stamp_2_test, time_stamp_3_test, time_stamp_4_test)

        print(f'shape of time_stamps_fetures are {time_step_train.shape}')

        return time_step_train, time_step_test