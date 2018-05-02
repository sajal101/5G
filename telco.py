import pandas as pd
import numpy as np
import collections
import tensorflow as tf
import os

root_dir = os.path.abspath('../..')
data_dir = os.path.join(root_dir, 'home', 'worker')



COLUMN_TYPES = collections.OrderedDict([
    ("sms_in_activity", float), 
    ("sms_out_activity", float),
    ("call_in_activity", float),
    ("call_out_activity", float),
    ("internet_traffic_activity", float),
    ("total_activity", float),
    ("activity", int)
])


ACTIVITY = ['Level 1 traffic', 'Level 2 traffic', 'Level 3 traffic', 'Level 4 traffic', 'Level 5 traffic', 'Level 6 traffic']

def maybe_download():
    df = pd.read_csv(os.path.join(data_dir, 'sms-call-internet-mi_datset_with_labels.csv'), names=COLUMN_TYPES.keys(), dtype=COLUMN_TYPES, header=0)

    return df

def load_data(y_name='activity', train_fraction=0.7, seed=None):
    
    data = maybe_download()
    np.random.seed(seed)
    train_x = data.sample(frac=train_fraction, random_state=seed)
    test_x = data.drop(train_x.index)
    train_y = train_x.pop(y_name)
    test_y = test_x.pop(y_name)
    return (train_x, train_y), (test_x, test_y)


def train_input_fn(features, labels, batch_size):
  
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

   
    dataset = dataset.shuffle(70000000).repeat().batch(batch_size)

   
    return dataset


def eval_input_fn(features, labels, batch_size):
   
    features=dict(features)
    if labels is None:
       
        inputs = features
    else:
        inputs = (features, labels)

    
    dataset = tf.data.Dataset.from_tensor_slices(inputs)

    
    assert batch_size is not None, "batch_size must not be None"
    dataset = dataset.batch(batch_size)

    
    return dataset
