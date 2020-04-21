import sys
sys.path.insert(0, "/lustrehome/adipilato/venv_CNNDoublets_new/CNN_CMS-Tracker")
import argparse
import datetime
import json
import tempfile
import os
import cnn_models as cnm

import tensorflow as tf
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard

from keras.models import Model
from keras import optimizers
from keras.constraints import max_norm
from keras.callbacks import Callback

import numpy as np
import itertools
import pickle
from random import shuffle
from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import roc_auc_score

from dataset import *

from keras import backend as K

import tensorflow.contrib.tensorrt as trt

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=int, default=0)
    main_args = parser.parse_args()
    

    remote_data = "/lustre/cms/store/user/adiflori/ConvPixels/TTBar_13TeV_PU_35/bal_data/"

    debug = False

    remote_dirs = os.listdir(remote_data)
    remote_dirs = [remote_data + d + "/" for d in remote_dirs if os.path.isdir(remote_data + d) and "ten_l" not in d]

    shuffle(remote_dirs)

    train_dirs = remote_dirs[:int(len(remote_dirs)*0.8)] if not debug else remote_dirs[:1]
    test_dirs = remote_dirs[int(len(remote_dirs)*0.8):int(len(remote_dirs)*0.9)] if not debug else remote_dirs[:1]
    val_dirs = remote_dirs[int(len(remote_dirs)*0.9):] if not debug else remote_dirs[:1]

    train_files = []
    val_files = []
    test_files = []
    print train_dirs
    MAX = 1
    for ff in train_dirs:
        train_files = train_files + [ff + "/" + f for f in os.listdir(ff) if "Step" in f][:MAX]
    for ff in val_dirs:
        val_files = val_files + [ff + "/" + f for f in os.listdir(ff) if "Step" in f][:MAX]
    for ff in test_dirs:
        test_files = test_files + [ff + "/" + f for f in os.listdir(ff) if "Step" in f][:MAX]
    
    train_files = train_files[:50]
    val_files = val_files[:20]
    test_files = test_files[:20]


    padshape = 16

    target_lab = "label"

    pdg_lab = "inTpPdgId"

    headLab = ["run","evt","lumi","PU","detSeqIn","detSeqOut","bSX","bSY","bSZ","bSdZ"]

    hitCoord = ["X","Y","Z","Phi","R"] #5

    hitDet = ["DetSeq","IsBarrel","Layer","Ladder","Side","Disk","Panel","Module","IsFlipped","Ax1","Ax2"] #11

    hitClust = ["ClustX","ClustY","ClustSize","ClustSizeX","ClustSizeY","PixelZero",
                "AvgCharge","OverFlowX","OverFlowY","Skew","IsBig","IsBad","IsEdge"] #13

    hitPixel = ["Pix" + str(el) for el in range(1, padshape*padshape + 1)]

    hitCharge = ["SumADC"]

    hitLabs = hitCoord + hitDet + hitClust + hitPixel + hitCharge

    inHitLabs = [ "in" + str(i) for i in hitLabs]
    outHitLabs = [ "out" + str(i) for i in hitLabs]

    inPixels = [ "in" + str(i) for i in hitPixel]
    outPixels = [ "out" + str(i) for i in hitPixel]


    particle = ["PId","TId","Px","Py","Pz","Pt","MT","ET","MSqr","PdgId",
                    "Charge","NTrackerHits","NTrackerLayers","Phi","Eta","Rapidity",
                    "VX","VY","VZ","DXY","DZ","BunchCrossing","IsChargeMatched",
                    "IsSigSimMatched","SharedFraction","NumAssocRecoTracks"]

    hitFeatures = hitCoord + hitDet + hitClust + hitCharge # 5 + 12 + 13 + 1 = 30

    inParticle = [ "inTp" + str(i) for i in particle]
    outParticle = [ "outTp" + str(i) for i in particle]

    inHitFeature  = [ "in" + str(i) for i in hitFeatures]
    outHitFeature = [ "out" + str(i) for i in hitFeatures]

    particleLabs = ["label","intersect","particles"] + inParticle +  outParticle

    differences = ["deltaA", "deltaADC", "deltaS", "deltaR", "deltaPhi", "deltaZ", "ZZero"]

    featureLabs = inHitFeature + outHitFeature + differences

    dataLab = headLab + inHitLabs + outHitLabs + differences + particleLabs + ["dummyFlag"]

    layer_ids = [0, 1, 2, 3, 14, 15, 16, 29, 30, 31]

    particle_ids = [-1.,11.,13.,15.,22.,111.,211.,311.,321.,2212.,2112.,3122.,223.]

    main_pdgs = [11.,13.,211.,321.,2212.]

    layer_ids = [0, 1, 2, 3, 14, 15, 16, 29, 30, 31]

    particle_ids = [-1.,11.,13.,15.,22.,111.,211.,311.,321.,2212.,2112.,3122.,223.]

    main_pdgs = [11.,13.,211.,321.,2212.]

    allLayerPixels = []

    for i in range(10):
        thisPixels = [ h + "_in_" + str(i) for h in hitPixel]
        allLayerPixels = allLayerPixels + thisPixels
    for i in range(10):
        thisPixels = [ h + "_out_" + str(i) for h in hitPixel]
        allLayerPixels = allLayerPixels + thisPixels
    #print '\n\n*******AllLayerPixels **********\n\n', allLayerPixels

    il1Pix = ['inPix41', 'inPix57', 'inPix72', 'inPix73', 'inPix74', 'inPix88', 'inPix89', 'inPix90', 'inPix104', 'inPix105', 'inPix106', 'inPix119', 'inPix120', 'inPix121', 'inPix122', 'inPix123', 'inPix135', 'inPix136', 'inPix137', 'inPix138', 'inPix139', 'inPix140', 'inPix151', 'inPix152', 'inPix153', 'inPix154', 'inPix155', 'inPix168', 'inPix169', 'inPix170', 'inPix184', 'inPix185', 'inPix186', 'inPix200', 'inPix201', 'inPix202', 'inPix217', 'inPix233']

    ol1Pix = ['outPix89', 'outPix104', 'outPix105', 'outPix106', 'outPix119', 'outPix120', 'outPix121', 'outPix122', 'outPix123', 'outPix134', 'outPix135', 'outPix136', 'outPix137', 'outPix138', 'outPix139', 'outPix151', 'outPix152', 'outPix153', 'outPix154', 'outPix155', 'outPix168', 'outPix169', 'outPix170', 'outPix185']

    inHitCoords = [ "in" + str(i) for i in hitCoord]
    outHitCoords = [ "out" + str(i) for i in hitCoord]
    inHitDet = [ "in" + str(i) for i in hitDet]
    outHitDet = [ "out" + str(i) for i in hitDet]
    inHitClust = [ "in" + str(i) for i in hitClust]
    outHitClust = [ "out" + str(i) for i in hitClust]


    val_data = Dataset(val_files)
    test_data = Dataset(test_files)
    train_data = Dataset(train_files)

    X_val_hit, X_val_info, y_val = val_data.get_layer_map_data()
    X_test_hit, X_test_info, y_test = test_data.get_layer_map_data()
    X_train_hit, X_train_info, y_train = train_data.get_layer_map_data()

    X_val_hit = X_val_hit.transpose(0,3,1,2)
    #X_val_hit.shape

    X_test_hit = X_test_hit.transpose(0,3,1,2)
    #X_test_hit.shape

    X_train_hit = X_train_hit.transpose(0,3,1,2)
    #X_train_hit.shape

    #print "X_train_info shape: ", X_train_info.shape
    #print "X_val_info shape: ", X_val_info.shape
    #print "X_test_info shape: ", X_test_info.shape
    

    model = cnm.build_model(main_args.model, n_channels=X_train_hit.shape[1], labels=X_train_info.shape[1])
    print('Testing Model Number: {}\n'.format(main_args.model))
    print model.summary()

    callbacks = [
            EarlyStopping(monitor='val_loss', patience=20, verbose=1, restore_best_weights=True),
            #ModelCheckpoint(fname + "_last.h5", save_best_only=True,
                            #save_weights_only=True),
                 TensorBoard(log_dir='./',
                 write_graph=True, write_images=True)
                 #roc_callback(training_data=(train_input_list,y),validation_data=(val_input_list,y_val))
        ]


    history = model.fit([X_train_hit, X_train_info], y_train, batch_size=1024, epochs=50, shuffle=True, validation_data=([X_val_hit, X_val_info], y_val), callbacks=callbacks, verbose=1)
    print '\n'

    #history_save = pd.DataFrame(history.history).to_hdf("pixel_and_labels_final_noBatchNorm_history.h5", "history", append=False)
    #print history.history

    def max_binary_accuracy(y_true, y_pred,n=50):

        thresholds = np.linspace(0.0,1.0,num=n)
        accmax = 0
        for t in thresholds:
            acc = np.mean(((y_pred[:,0] > t).astype(float)==y_true[:,0]).astype(float))
            if acc > accmax:
                tmax = t
                accmax = acc
        return accmax,tmax

    
    loss_test, acc_test = model.evaluate([X_test_hit, X_test_info], y_test, batch_size=1024, verbose=1)
    test_pred = model.predict([X_test_hit, X_test_info], verbose=1)
    test_roc = roc_auc_score(y_test, test_pred)
    test_acc,t_test = max_binary_accuracy(y_test,test_pred,n=1000)
    print('Test loss / test AUC       = {:.4f} / {:.4f} '.format(loss_test,test_roc))
    print('Test acc /  acc max (@t)   = {:.4f} / {:.4f} ({:.3f})'.format(acc_test,test_acc,t_test))

    loss_val, acc_val = model.evaluate([X_val_hit, X_val_info], y_val, batch_size=1024, verbose=1)
    val_pred = model.predict([X_val_hit, X_val_info], verbose=1)
    val_roc = roc_auc_score(y_val, val_pred)
    val_acc,t_val = max_binary_accuracy(y_val,val_pred,n=1000)
    print('Val loss / val AUC       = {:.4f} / {:.4f} '.format(loss_val,val_roc))
    print('Val acc /  acc max (@t)   = {:.4f} / {:.4f} ({:.3f})'.format(acc_val,val_acc,t_val))

    loss_train, acc_train = model.evaluate([X_train_hit, X_train_info], y_train, batch_size=1024, verbose=1)
    train_pred = model.predict([X_train_hit, X_train_info], verbose=1)
    train_roc = roc_auc_score(y_train, train_pred)
    train_acc,t_train = max_binary_accuracy(y_train,train_pred,n=1000)
    print('Train loss / train AUC       = {:.4f} / {:.4f} '.format(loss_train,train_roc))
    print('Train acc /  acc max (@t)   = {:.4f} / {:.4f} ({:.3f})'.format(acc_train,train_acc,t_train))

    print '\n'

    '''
    f=open('/lustrehome/adipilato/venv_CNNDoublets_new/CNN_CMS-Tracker/test_info/pixel_and_labels_final_noBatchNorm.txt', 'a+')
    f.write('Model Number: {}\n'.format(main_args.model))
    f.write('Test loss / test AUC       = {:.4f} / {:.4f} \n'.format(loss_test,test_roc))
    f.write('Test acc /  acc max (@t)   = {:.4f} / {:.4f} ({:.3f}) \n'.format(acc_test,test_acc,t_test))
    f.write('Val loss / val AUC       = {:.4f} / {:.4f} \n'.format(loss_val,val_roc))
    f.write('Val acc /  acc max (@t)   = {:.4f} / {:.4f} ({:.3f}) \n'.format(acc_val,val_acc,t_val))
    f.write('Train loss / train AUC       = {:.4f} / {:.4f} \n'.format(loss_train,train_roc))
    f.write('Train acc /  acc max (@t)   = {:.4f} / {:.4f} ({:.3f}) \n'.format(acc_train,train_acc,t_train))
    f.write('\n\n')
    f.close()


    print 'X Train Data Shape: ', X_train_hit.shape
    print 'X Val Data Shape: ', X_val_hit.shape
    print 'X Test Data Shape: ', X_test_hit.shape
    print 'X Train Info Shape: ', X_train_info.shape
    print 'X Val Info Shape: ', X_val_info.shape
    print 'X Test Info Shape: ', X_test_info.shape


    data_linear_train = X_train_hit.reshape(X_train_hit.shape[0],-1)
    data_linear_val   = X_val_hit.reshape(X_val_hit.shape[0],-1)
    data_linear_test  = X_test_hit.reshape(X_test_hit.shape[0],-1)

    print 'Data Linear Train Shape: ', data_linear_train.shape
    print 'Data Linear Val Shape: ', data_linear_val.shape
    print 'Data Linear Test Shape: ', data_linear_test.shape

    #data_linear[0][0:256].reshape(16,16)
    #data_linear_val[0][0:256].reshape(16,16)
    #data_linear_test[0][0:256].reshape(16,16)
    #print 'Data Linear Train Shape: ', data_linear.shape
    #print 'Data Linear Val Shape: ', data_linear_val.shape
    #print 'Data Linear Test Shape: ', data_linear_test.shape


    data_linear_train_df = pd.DataFrame(data_linear_train,columns=allLayerPixels)
    info_train_df = pd.DataFrame(X_train_info, dtype=float, columns=featureLabs)
    #data_linear_train_df["y0"] = y_train[:,0]
    #data_linear_train_df["y1"] = y_train[:,1]
    #data_linear_train_df.head()
    data_linear_test_df = pd.DataFrame(data_linear_test,columns=allLayerPixels)
    info_test_df = pd.DataFrame(X_test_info, dtype=float, columns=featureLabs)
    #data_linear_test_df["y0"] = y_test[:,0]
    #data_linear_test_df["y1"] = y_test[:,1]
    #data_linear_test_df.head()
    data_linear_val_df = pd.DataFrame(data_linear_val,columns=allLayerPixels)
    info_val_df = pd.DataFrame(X_val_info, dtype=float, columns=featureLabs)
    #data_linear_val_df["y0"] = y_val[:,0]
    #data_linear_val_df["y1"] = y_val[:,1]
    #data_linear_val_df.head()

    label_train_df = pd.DataFrame(y_train, columns=('y0', 'y1'))
    label_val_df = pd.DataFrame(y_val, columns=('y0', 'y1'))
    label_test_df = pd.DataFrame(y_test, columns=('y0', 'y1'))

    data_linear_val_df.to_hdf("pixel_and_labels_val.h5","data",append=False)
    data_linear_test_df.to_hdf("pixel_and_labels_test.h5","data",append=False)
    data_linear_train_df.to_hdf("pixel_and_labels_train.h5","data",append=False)

    info_val_df.to_hdf("pixel_and_labels_val.h5","features",append=False)
    info_test_df.to_hdf("pixel_and_labels_test.h5","features",append=False)
    info_train_df.to_hdf("pixel_and_labels_train.h5","features",append=False)

    label_train_df.to_hdf("pixel_and_labels_train.h5","labels",append=False)
    label_val_df.to_hdf("pixel_and_labels_val.h5","labels",append=False)
    label_test_df.to_hdf("pixel_and_labels_test.h5","labels",append=False)

    print 'Data files created succesfully!'
    print '\n'
    '''
    
    def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
        from tensorflow.python.framework.graph_util import convert_variables_to_constants
        graph = session.graph
        with graph.as_default():
            freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
            output_names = output_names or []
            output_names += [v.op.name for v in tf.global_variables()]
            input_graph_def = graph.as_graph_def()
            if clear_devices:
                for node in input_graph_def.node:
                    node.device = ""
            frozen_graph = convert_variables_to_constants(session, input_graph_def,
                                                          output_names, freeze_var_names)
            return frozen_graph
  
  

    #model.save_weights("pixel_and_labels_final.h5", overwrite=True)
    #model.save_weights("pixel_and_labels_newTest.h5", overwrite=True)
    frozen_graph = freeze_session(K.get_session(), output_names=[out.op.name for out in model.outputs])
    tf.train.write_graph(frozen_graph, "./", "pixel_and_labels_newTest.pb", as_text=False)

    '''
    print 'Loading the model for inference...'
    #model = pixel_only_model(n_channels=X_train_hit.shape[1])
    model = cnm.build_model(main_args.model)
    model.load_weights("pixel_and_labels_final_noBatchNorm.h5")
    test_df = pd.read_hdf("pixel_and_labels_test.h5", "data").values.reshape(-1,20,16,16)
    test_features_df = pd.read_hdf("pixel_and_labels_test.h5", "features").values.reshape(-1,67)

    y_test = pd.read_hdf("pixel_and_labels_test.h5", "labels").values.reshape(-1,2)

    print 'X Test Data Shape: ', test_df.shape
    print 'X Test Info Shape: ', test_features_df.shape
    print 'Y Test Shape: ', y_test.shape



    loss, acc = model.evaluate([test_df, test_features_df], y_test, batch_size=1024)
    test_pred = model.predict([test_df, test_features_df])
    test_roc = roc_auc_score(y_test, test_pred)
    test_acc,t_test = max_binary_accuracy(y_test,test_pred,n=1000)
    print('Test loss / test AUC       = {:.4f} / {:.4f} '.format(loss,test_roc))
    print('Test acc /  acc max (@t)   = {:.4f} / {:.4f} ({:.3f})'.format(acc,test_acc,t_test))
    '''
    
    print '***** Model working correctly! *****'
