import sys
batch = int(sys.argv[1])
lr = float(sys.argv[2])
opt_name = sys.argv[3]
views = str(sys.argv[4])

if views == '1':
    view = '4ch'
elif views == '2':
    view = '2ch'
elif views == '3':
    view = 'plax'
else:
    print('error')
    

import pandas as pd
import numpy as np
import gc
import os
import tensorflow as tf
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'False'
os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'

import random

SEED=1
def set_seeds(seed=SEED):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)
    

def set_global_determinism(seed=SEED):
    set_seeds(seed=seed)

    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.keras.backend.set_floatx('float32')

set_global_determinism(seed=SEED)


import keras
from tensorflow.keras import layers
from vit import create_vit_classifier
from transformer import transformer_model
from tensorflow.keras.optimizers import Adam, RMSprop, SGD, Adadelta, Adagrad, Adamax, Nadam, Ftrl, Lion, Adafactor, AdamW
import h5py
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_curve, auc

seq, w, h, c = 50, 96, 96, 1


def get_model():
    inputs = layers.Input(shape=(seq, w, h, c))
    vit = create_vit_classifier()
    feat = layers.TimeDistributed(vit)(inputs)

    transformer_encoder = transformer_model((seq, 512), 256, 4, 512, 4, [1024, 1024], dropout=0.1, mlp_dropout=0.1)

    feat = transformer_encoder(feat)
    model = keras.Model(inputs, feat)
    
    return model
    


def mcc(tn, fp, fn, tp):
    mcc = ((tp * tn) - (fp * fn)) / ((((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) ) ** 0.5) + 0.0000000000000000001)
    return mcc

def normmcc(mcc_value):
    return (mcc_value+1)/2





strategy = tf.distribute.MirroredStrategy()
print("Number of devices: {}".format(strategy.num_replicas_in_sync))
with strategy.scope():


    h5f = h5py.File('data/'+view+'_85t_50x96x96x1.h5', 'r')
    X = h5f['x'][:]
    Y = h5f['y'][:]
    h5f.close()
    Y[Y == 2] = 1


    kf = StratifiedKFold(n_splits=20, random_state=SEED, shuffle=True)
    for fold, (train_index, test_index) in enumerate(kf.split(X, Y)):
        x_train, y_train = X[train_index], Y[train_index]
        x_test, y_test = X[test_index], Y[test_index]

        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, stratify=y_train, test_size=0.1, random_state=SEED)


        x_train = tf.convert_to_tensor(x_train, np.float32)
        x_val = tf.convert_to_tensor(x_val, np.float32)
        x_test = tf.convert_to_tensor(x_test, np.float32)
        



        model = get_model()

        if(opt_name == "Lion"):
            opt = Lion
        elif(opt_name ==  "Adafactor"):
            opt = Adafactor
        elif(opt_name ==  "SGD"):
            opt = SGD
        elif(opt_name ==  "Adam"):
            opt = Adam
        elif(opt_name ==  "RMSprop"):
            opt = RMSprop
        elif(opt_name ==  "Adadelta"):
            opt = Adadelta
        elif(opt_name ==  "Adagrad"):
            opt = Adagrad
        elif(opt_name ==  "Adamax"):
            opt = Adamax
        elif(opt_name ==  "Nadam"):
            opt = Nadam
        elif(opt_name ==  "Ftrl"):
            opt = Ftrl
        else:
            opt = AdamW
            
        if(opt == SGD):
            model.compile(
                optimizer=opt(learning_rate = lr, momentum=0.9), # Use momentum when using SGD
                loss=keras.losses.BinaryCrossentropy(),
                metrics=['accuracy'],
            ) 
            
        else:
            model.compile(
                optimizer=opt(learning_rate = lr),
                loss=keras.losses.BinaryCrossentropy(),
                metrics=['accuracy'],
            )

        my_callbacks = [
            
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=15),
            keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5),
            keras.callbacks.TensorBoard(log_dir='./logs')
        ]
        model.fit(
            x = x_train,
            y = y_train,
            batch_size = batch,
            epochs = 100,
            verbose = 1,
            callbacks = my_callbacks,
            validation_data = (x_val, y_val)
        )

        y_pred = model.predict(x_test)
        pred = []
        for i in range(len(y_pred)):
            if y_pred[i] > 0.5:
                pred.append(1)
            else:
                pred.append(0)
        

        print("Fold: "+str(fold)+", Batch: "+str(batch)+", LR: "+str(lr)+", Opt: "+str(opt))
        cm = confusion_matrix(y_test, pred)
        print(cm)
        tn, fp, fn, tp = cm.ravel()
        mcc_value = mcc(tn, fp, fn, tp)

        f1 = f1_score(y_test, pred)
        acc = accuracy_score(y_test, pred) 
        prec = precision_score(y_test, pred) 
        rec = recall_score(y_test, pred, pos_label=1)
        spec = recall_score(y_test, pred, pos_label=0)
        fpr, tpr, thresholds = roc_curve(y_test, pred, pos_label=1)
        auc_score = auc(fpr, tpr)

        h5f = h5py.File('predict/view_'+view+'_fold_'+str(fold)+'_f1_'+str("{:.2f}".format(f1*100))+'_'+str(batch)+'_'+str(lr)+'_'+str(opt_name)+'.h5', 'w')
        h5f.create_dataset('y_test', data=y_test)
        h5f.create_dataset('y_pred', data=pred)
        h5f.create_dataset('y_prob', data=y_pred)
        h5f.close()

        del y_pred
        del pred
        del my_callbacks

        print('Test F1:', f1)
                        


        del model
        gc.collect()
        
        df = pd.read_excel(view+'_result.xlsx')
        print(df)
        idx = len(df.index)

        df2 = pd.DataFrame({
        "Fold": [str(fold)],
        "Fname": ['model/fold_'+str(fold)+'_f1_'+str("{:.2f}".format(f1*100))+'_'+str(batch)+'_'+str(lr)+'_'+str(opt_name)+'.h5'],
        "batch": [str(batch)],
        "lr": [str(lr)],
        "opt": [str(opt_name)],
        "Accuracy": [str("{:.2f}".format(acc*100))],
        "Precision": [str("{:.2f}".format(prec*100))],
        "Recall": [str("{:.2f}".format(rec*100))],
        "F1": [str("{:.2f}".format(f1*100))],
        "Specificity": [str("{:.2f}".format(spec*100))],
        "AUC": [str("{:.2f}".format(auc_score*100))],
        "Cm": [str(cm)],
        "MCC": ["{:.2f}".format(mcc_value*100)],
        "NormMCC": ["{:.2f}".format(normmcc(mcc_value)*100)]
        })

        df = pd.concat([df, df2])
        print(df)
        df.to_excel(view+'_result.xlsx', index=False)

