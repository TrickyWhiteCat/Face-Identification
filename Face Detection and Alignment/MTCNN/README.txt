
Prerequistes:
You should first download WIDER Face(for face detection) and LFW data set(for face alignmet)
Requirement:
Tensorflow 1.4.0
Python 2.6
Cuda 8.0

How to train
Step 1.Download Wider Face Training part only from Official Website , unzip to replace WIDER_train and put it into prepare_data folder.
Step 2.Download landmark training data from here,unzip and put them into prepare_data folder.
Step 3.Run prepare_data/gen_12net_data.py to generate training data(Face Detection Part) for PNet.
Step 4.Run gen_landmark_aug_12.py to generate training data(Face Landmark Detection Part) for PNet.
Step 5.Run gen_imglist_pnet.py to merge two parts of training data.
Step 6.Run gen_PNet_tfrecords.py to generate tfrecord for PNet.
Step 7.After training PNet, run gen_hard_example to generate training data(Face Detection Part) for RNet.
Step 8.Run gen_landmark_aug_24.py to generate training data(Face Landmark Detection Part) for RNet.
Step 9.Run gen_imglist_rnet.py to merge two parts of training data.
Step 10.Run gen_RNet_tfrecords.py to generate tfrecords for RNet.(you should run this script four times to generate tfrecords of neg,pos,part and landmark respectively)
Step 11.After training RNet, run gen_hard_example to generate training data(Face Detection Part) for ONet.
Step 12.Run gen_landmark_aug_48.py to generate training data(Face Landmark Detection Part) for ONet.
Step 13.Run gen_imglist_onet.py to merge two parts of training data.
Step 14.Run gen_ONet_tfrecords.py to generate tfrecords for ONet.(you should run this script four times to generate tfrecords of neg,pos,part and landmark respectively)