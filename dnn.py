from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import tensorflow as tf

import telco


parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=10000, type=int, help='batch size')
parser.add_argument('--train_steps', default=1000, type=int,
                    help='number of training steps')

def main(argv):
    args = parser.parse_args(argv[1:])

    
    (train_x, train_y), (test_x, test_y) = telco.load_data()

   
    my_feature_columns = []
    for key in train_x.keys():
        my_feature_columns.append(tf.feature_column.numeric_column(key=key))

   
    classifier = tf.estimator.DNNClassifier(
        feature_columns=my_feature_columns,
        hidden_units=[200, 200, 200, 200, 200, 200, 200],
	    optimizer = tf.train.AdamOptimizer(learning_rate=0.01),
        n_classes=6)


    classifier.train(
        input_fn=lambda:telco.train_input_fn(train_x, train_y,
                                                 args.batch_size),
        steps=args.train_steps)

    
    eval_result = classifier.evaluate(
        input_fn=lambda:telco.eval_input_fn(test_x, test_y,
                                                args.batch_size))

    print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))
  
    expected = ['Level 1 traffic', 'Level 2 traffic', 'Level 3 traffic', 'Level 4 traffic', 'Level 5 traffic', 'Level 6 traffic']
    predict_x = {
        'sms_in_activity': [0.16033934024272, 0.362811761613819, 0.386824212358037, 2.0323739668121, 2.45173595717526, 1.4424266046465],
        'sms_out_activity': [0.15674761016999, 0.360796705611827, 0.0705157218261135, 0.811113284550833, 0.876051762193693, 2.97551670275416],
        'call_in_activity': [0.175125027268656, 2.07729665223295, 2.78842780665289, 2.11352834750183, 2.54343481774661, 3.38567200810542],
        'call_out_activity': [0.340171238281068, 1.7719497761992, 2.70231911027249, 3.81818872119848, 2.63967907042337, 2.84367515470732],
        'internet_traffic_activity': [8.3341667074409, 31.2604810928211, 35.5398500846228, 52.57864159823, 75.5395669242859, 119.331281141024],
        'total_activity': [9.16654992340333, 35.8333359884789, 41.4879369357323, 61.3538459182932, 84.0504685318248, 129.978571611237],
 
    }
    predictions = classifier.predict(
        input_fn=lambda:telco.eval_input_fn(predict_x,
                                                labels=None,
                                                batch_size=args.batch_size))

    template = ('\nPrediction is "{}" ({:.1f}%), expected "{}"')

    for pred_dict, expec in zip(predictions, expected):
        class_id = pred_dict['class_ids'][0]
        probability = pred_dict['probabilities'][class_id]

        print(template.format(telco.ACTIVITY[class_id],
                              100 * probability, expec))


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
tf.app.run(main)