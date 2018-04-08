import pandas as pd
import tensorflow as tf
import os

train1 = os.path.abspath('C:/Users/cupid/dnn/sms-call-internet-mi-2013-11-01_new.csv')
test1 = os.path.abspath('C:/Users/cupid/dnn/sms-call-internet-mi-2013-11-02_new.csv')


CSV_COLUMN_NAMES = ['sms_in_activity', 'sms_out_activity', 'call_in_activity', 'call_out_activity', 'internet_traffic_activity', 'total_activity', 'activity']

ACTIVITY = ['Level 1 traffic', 'Level 2 traffic', 'Level 3 traffic', 'Level 4 traffic', 'Level 5 traffic', 'Level 6 traffic']

def maybe_download():
    train_path = tf.keras.utils.get_file(train1.split('/')[-1], train1)
    test_path = tf.keras.utils.get_file(test1.split('/')[-1], test1)

    return train_path, test_path

def load_data(y_name='activity'):
    
    train_path, test_path = maybe_download()

    train = pd.read_csv(train_path, names=CSV_COLUMN_NAMES, header=0)
    train_x, train_y = train, train.pop(y_name)

    test = pd.read_csv(test_path, names=CSV_COLUMN_NAMES, header=0)
    test_x, test_y = test, test.pop(y_name)

    return (train_x, train_y), (test_x, test_y)


def train_input_fn(features, labels, batch_size):
  
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

   
    dataset = dataset.shuffle(2000000).repeat().batch(batch_size)

   
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
