_EPSILON = 1e-08

import numpy as np
import pandas as pd
import tensorflow as tf
import random
import os
# import sys

from termcolor import colored
from tensorflow.contrib.layers import fully_connected as FC_Net
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import train_test_split

import import_data as impt
import utils_network as utils

from class_DeepHit import Model_DeepHit
from utils_eval_ctd import c_index, brier_score, weighted_c_index, weighted_brier_score, ctd_index


def load_logging(filename):
    data = dict()
    with open(filename) as f:
        def is_float(input):
            try:
                num = float(input)
            except ValueError:
                return False
            return True

        for line in f.readlines():
            if ':' in line:
                key,value = line.strip().split(':', 1)
                if value.isdigit():
                    data[key] = int(value)
                elif is_float(value):
                    data[key] = float(value)
                elif value == 'None':
                    data[key] = None
                else:
                    data[key] = value
            else:
                pass # deal with bad lines of text here    
    return data



##### MAIN SETTING
OUT_ITERATION               = 5

data_mode                   = 'METABRIC' #METABRIC, SYNTHETIC
seed                        = 1234

EVAL_TIMES                  = [12, 24, 36] # evalution times (for C-index and Brier-Score)


##### IMPORT DATASET
'''
    num_Category            = max event/censoring time * 1.2 (to make enough time horizon)
    num_Event               = number of evetns i.e. len(np.unique(label))-1
    max_length              = maximum number of measurements
    x_dim                   = data dimension including delta (num_features)
    mask1, mask2            = used for cause-specific network (FCNet structure)
'''
if data_mode == 'SYNTHETIC':
    (x_dim), (data, time, label), (mask1, mask2) = impt.import_dataset_SYNTHETIC(norm_mode = 'standard')
    EVAL_TIMES  = [12, 24, 36]
elif data_mode == 'METABRIC':
    (x_dim), (data, time, label), (mask1, mask2) = impt.import_dataset_METABRIC(norm_mode = 'standard')
    EVAL_TIMES  = [144, 288, 432]
else:
    print('ERROR:  DATA_MODE NOT FOUND !!!')

_, num_Event, num_Category  = np.shape(mask1)  # dim of mask1: [subj, Num_Event, Num_Category]



in_path = data_mode + '/results_TF_CTD/'

if not os.path.exists(in_path):
    os.makedirs(in_path)


FINAL_CTD = np.zeros([num_Event, OUT_ITERATION])


for out_itr in range(OUT_ITERATION):
    in_hypfile = in_path + '/itr_' + str(out_itr) + '/hyperparameters_log.txt'
    in_parser = load_logging(in_hypfile)


    ##### HYPER-PARAMETERS
    mb_size                     = in_parser['mb_size']

    iteration                   = in_parser['iteration']

    keep_prob                   = in_parser['keep_prob']
    lr_train                    = in_parser['lr_train']

    h_dim_shared                = in_parser['h_dim_shared']
    h_dim_CS                    = in_parser['h_dim_CS']
    num_layers_shared           = in_parser['num_layers_shared']
    num_layers_CS               = in_parser['num_layers_CS']

    if in_parser['active_fn'] == 'relu':
        active_fn                = tf.nn.relu
    elif in_parser['active_fn'] == 'elu':
        active_fn                = tf.nn.elu
    elif in_parser['active_fn'] == 'tanh':
        active_fn                = tf.nn.tanh
    else:
        print('Error!')


    initial_W                   = tf.contrib.layers.xavier_initializer()

    alpha                       = in_parser['alpha']  #for log-likelihood loss
    beta                        = in_parser['beta']  #for ranking loss
    gamma                       = in_parser['gamma']  #for RNN-prediction loss
    parameter_name              = 'a' + str('%02.0f' %(10*alpha)) + 'b' + str('%02.0f' %(10*beta)) + 'c' + str('%02.0f' %(10*gamma))


    ##### MAKE DICTIONARIES
    # INPUT DIMENSIONS
    input_dims                  = { 'x_dim'         : x_dim,
                                    'num_Event'     : num_Event,
                                    'num_Category'  : num_Category}

    # NETWORK HYPER-PARMETERS
    network_settings            = { 'h_dim_shared'         : h_dim_shared,
                                    'h_dim_CS'          : h_dim_CS,
                                    'num_layers_shared'    : num_layers_shared,
                                    'num_layers_CS'    : num_layers_CS,
                                    'active_fn'      : active_fn,
                                    'initial_W'         : initial_W }


    # for out_itr in range(OUT_ITERATION):
    print ('ITR: ' + str(out_itr+1) + ' DATA MODE: ' + data_mode + ' (a:' + str(alpha) + ' b:' + str(beta) + ' c:' + str(gamma) + ')' )
    ##### CREATE DEEPFHT NETWORK
    tf.reset_default_graph()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    model = Model_DeepHit(sess, "DeepHit", input_dims, network_settings)
    saver = tf.train.Saver()

    sess.run(tf.global_variables_initializer())

    ### TRAINING-TESTING SPLIT
    (tr_data,te_data, tr_time,te_time, tr_label,te_label, 
     tr_mask1,te_mask1, tr_mask2,te_mask2)  = train_test_split(data, time, label, mask1, mask2, test_size=0.20, random_state=seed+out_itr) 

    (tr_data,va_data, tr_time,va_time, tr_label,va_label, 
     tr_mask1,va_mask1, tr_mask2,va_mask2)  = train_test_split(tr_data, tr_time, tr_label, tr_mask1, tr_mask2, test_size=0.20, random_state=seed+out_itr) 
    
    ##### PREDICTION & EVALUATION
    saver.restore(sess, in_path + '/itr_' + str(out_itr) + '/models/model_itr_' + str(out_itr))

    ### PREDICTION
    pred = model.predict(te_data)
    
    ### EVALUATION: Antolini C^td
    result_ctd = np.zeros([num_Event])

    for k in range(num_Event):
        result_ctd[k] = ctd_index(
            pred,
            te_time[:,0],
            te_label[:,0],
            event=k+1
        )

        if not np.isfinite(result_ctd[k]) or result_ctd[k] < 0:
            raise RuntimeError(
                "Invalid test C^td for event {}: {}".format(
                    k+1, result_ctd[k]
                )
            )

    FINAL_CTD[:, out_itr] = result_ctd

    ### SAVE / PRINT C^td
    df_ctd = pd.DataFrame(
        result_ctd.reshape(-1, 1),
        index=['Event_' + str(k+1) for k in range(num_Event)],
        columns=['Ctd']
    )
    df_ctd.to_csv(
        in_path + '/result_CTD_itr' + str(out_itr) + '.csv'
    )

    print('========================================================')
    print('ITR: ' + str(out_itr+1) + ' DATA MODE: ' + data_mode)
    print('- C^td: ')
    print(df_ctd)
    print('========================================================')


### FINAL MEAN/STD
ctd_mean = np.mean(FINAL_CTD, axis=1)
ctd_std  = np.std(FINAL_CTD, axis=1)

df_ctd_final = pd.DataFrame(
    {
        'mean_Ctd': ctd_mean,
        'std_Ctd': ctd_std
    },
    index=['Event_' + str(k+1) for k in range(num_Event)]
)

df_ctd_final.to_csv(
    in_path + '/result_CTD_FINAL.csv'
)

print('========================================================')
print('- FINAL C^td: ')
print(df_ctd_final)
print('========================================================')
