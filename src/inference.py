# This script deploys a trained model for performance evaluation. 

import numpy as np
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
import os
from us_datagenerator import testdatagen
from datetime import datetime
from utilities import IOU

#We created 4 folders to document the the good and bas results and also seperate the prediction and the ground truth
good_prediction = "/path/to/correctly_predicted_images"
good_actual= "/ground_truth_for_good_predictions/"
bad_prediction = "/path/forpredicted images/prediction"
bad_actual = "/path_where/ground_truth/will_be_saved"

def draw_img(X, Y, T,listname, threshold):# Y= prediction & T = Target
    true_positive=0
    for i in range(0,X.shape[0]):
        img = X[i].reshape(512,512,1)
        img=img*255
        img_pred = np.concatenate((img,img,img),axis=-1)
        img_act = np.concatenate((img,img,img),axis=-1)
        cv2.rectangle(img_pred,(int(Y[i,0]), int(Y[i,1])), (int(Y[i,2]), int(Y[i,3])), (0,0,255), 2)
        cv2.rectangle(img_act,(int(T[i,0]), int(T[i,1])), (int(T[i,2]), int(T[i,3])), (255,0,0), 2)
        pred_mid = int((int(Y[i,3]) + int(Y[i,1]))/2)
        tar_mid = int((int(T[i,3]) + int(T[i,1]))/2)
        distance = int(abs(pred_mid-tar_mid))
        iou = IOU(Y[i,:],T[i,:])
        if distance<threshold and iou >=0.5:
        #if distance<=threshold:
            true_positive+=1
            # uncommenting the two  line below  would save the images
            #cv2.imwrite(os.path.join(good_prediction,listname[i]),img_pred)
            #cv2.imwrite(os.path.join(good_actual,listname[i]),img_act)
        else:
            k = 0
            # uncommenting the two  line below  would save the images
            #cv2.imwrite(os.path.join(bad_prediction,listname[i]),img_pred)
            #cv2.imwrite(os.path.join(bad_actual,listname[i]),img_act)
    return true_positive

def infer(testgenerator, batch_size = 4, threshold = 28):
    with tf.Session() as sess:
        imported_meta = tf.train.import_meta_graph("./Model_97/modelfinal.meta") # Giving path to where the model has be saved
        imported_meta.restore(sess,"./Model_97/modelfinal") # for more information http://stackabuse.com/tensorflow-save-and-restore-models/
        total_count = 0
        true_positive = 0
        batches = 1221
        activation = tf.get_collection("prediction")[0]
        start = datetime.now()
        for i in range(0, batches):
            Xtest , Ytest , imlist = next(testgenerator)
            result = np.zeros((len(Xtest), 4))
            for j in range(0, len(Xtest)//batch_size):
                Xtest_b = Xtest[j*batch_size:(j+1)*batch_size]
                Ytest_b = Ytest[j*batch_size:(j+1)*batch_size]
                result[j*batch_size:(j+1)*batch_size] = sess.run(activation, feed_dict={"Placeholder:0": Xtest_b})
            true_positive+=draw_img(Xtest,result,Ytest,imlist, threshold)
            total_count=(i+1)*len(Xtest)
            print(true_positive,total_count)
            accuracy = (true_positive/total_count)*100
            print("batch %d of 1221 , accuracy: %.2f percent" %(i, accuracy))
        end = datetime.now()
        duration = (end - start).total_seconds()

        print("Frame rate ", (batches*16)/duration)


def main():
    testgenerator = testdatagen()
    batch_size = 4 # Selecting the batch size, larger batch size would give memory error as the RAM available is 8GB
    threshold = 28 # Pixel threshold 28 corresponds to 2 mm in US images , for more information see report
    infer(testgenerator, batch_size, threshold)

if __name__ == "__main__":
    main()
