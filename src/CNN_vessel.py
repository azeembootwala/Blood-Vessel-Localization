# The folowing script is the main training deep learning script of the project
# This file trains a deep neural network. The Architecture used is VGG for more information visit https://arxiv.org/abs/1409.1556
# At the end of the training, it shall test the model on validation images and give you the accuracy.
# However the accuracy and frame rate is again calculated in inference.py
# The code is written in a class format. Each class has its conctructor which takes in operation parameters
# Each class, has its own forward function. The classes are made according to the Architecture of the VGG network
# Conv2poolLayers have two conv operations followed by pooling
# Conv3poolLayers have theee conv operation followed by pooling
# HiddenLayers have parameters that has FC (fully connected) layer operations
# Finally class CNN combiles all the combination of classes according to the Architecture
# Author :- Azeem Bootwala
# Date :- January, 2018
import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle
from util_vgg import init_filter,init_weights_bias,y2ind, error_rate
from us_datagenerator import traindatagen , testdatagen
import cv2
import os

good_path = "/path/to/folder/for/good_results"
bad_path = "/path/to/folder/for/poor_results"

def draw_img(X, Y, T,listname):# Y= prediction & T = Target
    true_positive=0
    for i in range(0,X.shape[0]):
        img = X[i].reshape(512,512,1)
        img*=255
        cv2.rectangle(img ,(int(Y[i,0]), int(Y[i,1])), (int(Y[i,2]), int(Y[i,3])), (255,0,0), 2)
        cv2.rectangle(img ,(int(T[i,0]), int(T[i,1])), (int(T[i,2]), int(T[i,3])), (255,0,0), 1)
        pred_mid = int((int(Y[i,3]) + int(Y[i,1]))/2)
        tar_mid = int((int(T[i,3]) + int(T[i,1]))/2)
        distance = int(np.sqrt((pred_mid-tar_mid)**2))

        if distance<28:
            true_positive+=1
            #cv2.imwrite(os.path.join(good_path,listname[i]),img)
        else:
            K = 0
            #cv2.imwrite(os.path.join(bad_path,listname[i]),img)

    return true_positive


class Conv2poolLayers(object):
    def __init__(self,mi1,mo1,mi2,mo2,f_in=3,f_out=3,pool_size=(2,2)):
        self.shape1 = (f_in, f_out, mi1, mo1)
        self.shape2 = (f_in , f_out, mi2, mo2)
        W1, b1 = init_filter(self.shape1,pool_size)
        W2, b2 = init_filter(self.shape2,pool_size)
        self.W1 = tf.Variable(W1.astype(np.float32))
        self.b1 = tf.Variable(b1.astype(np.float32))
        self.W2 = tf.Variable(W2.astype(np.float32))
        self.b2 = tf.Variable(b2.astype(np.float32))
        self.params = [self.W1,self.b1,self.W2,self.b2]

    def forward(self, X):
        conv_out  = tf.nn.conv2d(X , self.W1 , strides = [1,1,1,1],padding="SAME")
        conv_out  = tf.nn.bias_add(conv_out,self.b1)
        conv_out = tf.nn.relu(conv_out) # added later
        conv2_out = tf.nn.conv2d(conv_out, self.W2, strides=[1,1,1,1],padding="SAME")
        conv2_out = tf.nn.bias_add(conv2_out,self.b2)
        conv2_out = tf.nn.relu(conv2_out) # added later
        pool_out  = tf.nn.max_pool(conv2_out, ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")
        return pool_out

class Conv3poolLayers(object):
    def __init__(self, mi1,mo1,mi2,mo2,mi3,mo3,f_w=3,f_h=3,pool_size=(2,2)):
        self.shape1 = (f_w , f_h , mi1, mo1)
        self.shape2 = (f_w , f_h , mi2, mo2)
        self.shape3 = (f_w , f_h , mi3 ,mo3)
        W1, b1 = init_filter(self.shape1,pool_size)
        W2, b2 = init_filter(self.shape2,pool_size)
        W3, b3 = init_filter(self.shape3, pool_size)
        self.W1 = tf.Variable(W1.astype(np.float32))
        self.b1 = tf.Variable(b1.astype(np.float32))
        self.W2 = tf.Variable(W2.astype(np.float32))
        self.b2 = tf.Variable(b2.astype(np.float32))
        self.W3 = tf.Variable(W3.astype(np.float32))
        self.b3 = tf.Variable(b3.astype(np.float32))
        self.params = [self.W1, self.b1 , self.W2 , self.b2,self.W3 , self.b3]

    def forward(self,X):
        conv_out  = tf.nn.conv2d(X , self.W1 , strides = [1,1,1,1],padding="SAME")
        conv_out  = tf.nn.bias_add(conv_out,self.b1)
        conv_out  = tf.nn.relu(conv_out) # Added later
        conv2_out = tf.nn.conv2d(conv_out, self.W2, strides=[1,1,1,1],padding="SAME")
        conv2_out = tf.nn.bias_add(conv2_out,self.b2)
        conv2_out = tf.nn.relu(conv2_out) # Added later
        conv3_out = tf.nn.conv2d(conv2_out,self.W3 , strides=[1,1,1,1], padding="SAME")
        conv3_out = tf.nn.bias_add(conv3_out,self.b3)
        conv3_out = tf.nn.relu(conv3_out)# added later
        pool_out = tf.nn.max_pool(conv3_out,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")
        return pool_out

class HiddenLayers(object):
    def __init__(self, M1 , M2 ):
        W , b = init_weights_bias(M1 , M2)
        self.W = tf.Variable(W.astype(np.float32))
        self.b = tf.Variable(b.astype(np.float32))
        self.params=[self.W, self.b]

    def forward(self,X):
        return tf.nn.relu(tf.matmul(X,self.W)+self.b)

class CNN(object):
    def __init__(self, conv2layersizes , conv3layersizes , hiddenlayersizes):
        self.fw = 3
        self.fh = 3
        self.conv2layersizes = conv2layersizes
        self.conv3layersizes = conv3layersizes
        self.hiddenlayersizes = hiddenlayersizes
    def fit(self,traingenerator,testgenerator,learning_rate =10e-7,mu = 0.9, decay = 0.99,reg=0 ,batch_size=4,show_fig = True):

        dw = 512 # These two variables represent the input size of the image
        dh = 512

        self.convlayer = []

        for mi1 , mo1, mi2, mo2 in  self.conv2layersizes:
            c2obj = Conv2poolLayers(mi1, mo1 , mi2, mo2)
            self.convlayer.append(c2obj)
            dw = dw//2
            dh = dh//2
        for mi1, mo1 , mi2, mo2 , mi3 , mo3 in self.conv3layersizes:
            c3obj = Conv3poolLayers(mi1, mo1 , mi2 , mo2 , mi3 , mo3)
            self.convlayer.append(c3obj)
            dw = dw//2
            dh = dh//2

        self.hiddenlayer = []
        M1 = self.convlayer[-1].shape3[-1] * dw * dh
        for M2 in self.hiddenlayersizes:
            h = HiddenLayers(M1 , M2)
            self.hiddenlayer.append(h)
            M1 = M2
        W , b = init_weights_bias(M1 , 4)
        self.W = tf.Variable(W.astype(np.float32))
        self.b = tf.Variable(b.astype(np.float32))
        self.params = [self.W , self.b]
        for c in self.convlayer:
            self.params += c.params
        for h in self.hiddenlayer:
            self.params+=h.params

        tfX = tf.placeholder(tf.float32,shape=(None,512,512,1))
        tfT = tf.placeholder(tf.float32,shape=(None , 4))

        act = self.forward(tfX)

        rcost = reg*sum([tf.nn.l2_loss(p) for p in self.params])
        #cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=act, labels=tfT))  + rcost
        cost = tf.reduce_sum(tf.losses.mean_squared_error(labels=tfT, predictions=act)) + rcost

        training_op = tf.train.AdamOptimizer(learning_rate).minimize(cost)

        prediction_op = self.predict(tfX)

        tf.add_to_collection("prediction", prediction_op) # This landmark initializations are needed to extract information in the inference.py script

        saver = tf.train.Saver()  # We use this to initialize the tensorflow saver class


        N = 16 # No of train & test bach images loaded
        n_batches = N // batch_size
        max_iter = 10 # Number of epochs # default:- 10

        init = tf.global_variables_initializer()

        LL = []
        with tf.Session() as session:
            session.run(init)
            for g in range(0,max_iter):

                for i in range(0,6901):# looping over all images default 6901
                    Xtrain , Ytrain = next(traingenerator) # initialized a gererator function to give us images

                    train_cost=0
                    for j in range(0, n_batches):
                        Xbatch = Xtrain[j*batch_size:(j+1)*batch_size]
                        Ybatch = Ytrain[j*batch_size:(j+1)*batch_size]

                        session.run(training_op, feed_dict={tfX:Xbatch , tfT:Ybatch})
                        train_cost+=session.run(cost, feed_dict={tfX:Xbatch,tfT:Ybatch})
                    train_cost = train_cost/16
                    print("Epoch %d of %d Iteration %d of 6900 , cost: %f"%(g,max_iter,i,train_cost))

            tot_count = 0
            true_positive=0
            for p in range(0,1221):# default 1221
                Xtest, Ytest,imlist = next(testgenerator)
                prediction = np.zeros((len(Xtest),4))
                c=0
                for k in range(0,len(Xtest)//batch_size):
                    Xtest_b = Xtest[k*batch_size:(k+1)*batch_size]
                    Ytest_b = Ytest[k*batch_size:(k+1)*batch_size]
                    c+= session.run(cost, feed_dict={tfX:Xtest_b, tfT:Ytest_b})
                    prediction[k*batch_size:(k+1)*batch_size] = session.run(prediction_op, feed_dict={tfX:Xtest_b})

                c = c/16
                LL.append(c)

                os.chdir("/path/you/want_to/save_your_trainded_model/") # We change the path to where we want the model to be stored
                true_positive+=draw_img(Xtest,prediction,Ytest,imlist)
                tot_count =(p+1)*len(Xtest)
                print(true_positive,tot_count)
                accuracy = (true_positive/tot_count)*100
                print("batch %d of 1221 cost: %d, accuracy: %d percent" %(p,c, accuracy))
            saver.save(session,"modelfinal")

    def forward(self, X):
        Z = X
        for c in self.convlayer:
            Z = c.forward(Z)
        tf.add_to_collection("con",Z)
        Z_shape= Z.get_shape().as_list() # We reshape the output of last conv layers for FC layers
        Z = tf.reshape(Z, [-1 , np.prod(Z_shape[1:])])
        for h in self.hiddenlayer:
            Z = h.forward(Z)
        tf.add_to_collection("FC_layer", Z)
        return tf.matmul(Z, self.W) + self.b
    def predict(self,X):
        X = self.forward(X)
        return X



def main():
    model = CNN([(1,64,64,64),(64,128,128,128)],
                [(128,256,256,256,256,256),(256,512,512,512,512,512),(512, 512, 512, 512, 512, 512),
                (512,512,512,512,512,512)],
                [4096,4096])
    traingenerator = traindatagen() # Initializing objects for python generators
    testgenerator = testdatagen()

    model.fit(traingenerator,testgenerator)

if __name__ =="__main__":
    main()
