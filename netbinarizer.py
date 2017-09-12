#!/usr/bin/python
# encoding=utf8
import time
import tensorflow as tf
import numpy as np
import h5py
#######################     h5py READER     ###################################
binary = False
modify = True
if(modify):
    if (binary):
        f = h5py.File('C:/Users/yash/.keras/models/binarycompress.h5', mode = 'r+')
    else: 
        f = h5py.File('C:/Users/yash/.keras/models/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5', mode = 'r+')
    keyholder = list(h5py.AttributeManager.keys(f))
    layerstringlist = []
    for i in range(len((keyholder))):
        for j in range(len((keyholder[i]))):
            try:
                string = str(keyholder[i]) + "/" + str(list(f.get(keyholder[i]))[j])
                layerstringlist.append(string)
            except:
                pass
    print("=======BIAS MOD=========")
    for i in range(1, 26, 2):
        print(i)
        print(layerstringlist[i], '  ' , i)
        layer = h5py.AttributeManager.get(f, key = str(layerstringlist[i]))
        np_layer = np.asarray(list(layer))
        Alk = (abs(np_layer).sum())/(np_layer.shape[0])
        np_layer = 0*(np.sign(np_layer))
        del f[layerstringlist[i]]
        f[layerstringlist[i]] = np_layer
    print("=======WEIGHT MOD=========")
    for i in range(2, 25, 2):
        print(i)
        print(layerstringlist[i], '  ' , i)
        layer = h5py.AttributeManager.get(f, key = str(layerstringlist[i]))
        np_layer = np.asarray(list(layer))
        for b in range(np_layer.shape[2]):
            for c in range(np_layer.shape[3]):
                        Alk = (abs(np_layer[:, :, b, c]).sum())/(np_layer.shape[1]*np_layer.shape[0])
                        np_layer[:,:,b,c] = Alk*(np.sign(np_layer[:,:,b,c]))
        del f[layerstringlist[i]]
        f[layerstringlist[i]] = np_layer
    print("=======FC WEIGHT MOD=========")
    for i in range(26, 31, 2):
        print(i)
        print(layerstringlist[i], '  ' , i)
        layer = h5py.AttributeManager.get(f, key = str(layerstringlist[i]))
        np_layer = np.asarray(list(layer))
        Alk = (abs(np_layer.sum())/(np_layer.shape[1]*np_layer.shape[0]))
        np_layer = Alk*(np.sign(np_layer))
        del f[layerstringlist[i]]
        f[layerstringlist[i]] = np_layer
    print("=======FC BIAS MOD=========")
    for i in range(27, 32, 2):    
        print(i)
        print(layerstringlist[i], '  ' , i)
        layer = h5py.AttributeManager.get(f, key = str(layerstringlist[i]))
        np_layer = np.asarray(list(layer))
        Alk = (abs(np_layer).sum())/(np_layer.shape[0])
        np_layer = Alk*(np.sign(np_layer))
        del f[layerstringlist[i]]
        f[layerstringlist[i]] = np_layer
    f.close()
    
    print("Array testing Initiating, Stop if you like")
else:
    ################################## ARRAY TESTER ###############################
    f = h5py.File('C:/Users/yash/.keras/models/vgg16_weights_tf_dim_ordering_tf_kernels_2.h5', mode = 'r+')
    ab = list(h5py.AttributeManager.keys(f))
    layerstringlisttest = []
    for i in range(len(list(h5py.AttributeManager.keys(f)))):
        print("    Parent layer")
        print("          ",  ab[i])
        print("                        Children:")
        for j in range(len(list(ab[i]))):
            try:
                string = str(ab[i]) + "/" + str(list(f.get(ab[i]))[j])
                print(string)
                layerstringlisttest.append(string)
                print("               ", string, "     ", np.asarray(list(f.get(string))).shape)
            except:
                pass
    print('===============WEIGHTS===============')
    print(list(f[layerstringlist[0]][:,:,2,1]))
    print(list(f[layerstringlist[2]][:,:,2,6]))
    print(list(f[layerstringlist[2]][:,:,2,6]))
    print(list(f[layerstringlist[2]][:,:,1,9]))
    time.sleep(10)
    print(list(f[layerstringlist[8]][:,:,5,4]))
    print(list(f[layerstringlist[8]][:,:,2,6]))
    time.sleep(10)
    print(list(f[layerstringlist[22]][:,:,2,6]))
    print(list(f[layerstringlist[22]][:,:,1,9]))
    time.sleep(10)
    print('===============BIAS & FC===============')
    print(list(f[layerstringlist[3]]))
    time.sleep(10)
    print(list(f[layerstringlist[7]]))
    time.sleep(10)
    print(list(f[layerstringlist[21]]))
    time.sleep(10)
    print(list(f[layerstringlist[28]]))
    time.sleep(10)
    print(list(f[layerstringlist[31]]))
