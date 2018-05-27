# Here a few modifications of the original code were made , to implement batch normalization and dropout
# This script is not so important as , they are some experiments with the neural network with average results 
import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from util_vgg import init_filter,init_weights_bias,y2ind, error_rate
from us_datagen import traindatagen , testdatagen
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
            #cv2.imwrite(os.path.join(good_path,listname[i]),img) # uncommenting this line would save the images
        else:
            K = 0
            #cv2.imwrite(os.path.join(bad_path,listname[i]),img) # uncommenting this line would save the images

    return true_positive


class Conv2poolLayers(object):
    def __init__(self,mi1,mo1,mi2,mo2,f_in=3,f_out=3,pool_size=(2,2)):
        self.shape1 = (f_in, f_out, mi1, mo1)
        self.shape2 = (f_in , f_out, mi2, mo2)
        W1, b1 = init_filter(self.shape1,pool_size)
        W2, b2 = init_filter(self.shape2,pool_size)
        self.W1 = tf.Variable(W1.astype(np.float32))
        self.beta1 = tf.Variable(np.zeros(mo1).astype(np.float32))
        self.gamma1 = tf.Variable(np.ones(mo1).astype(np.float32))
        self.run_mean1 = tf.Variable(np.zeros(mo1).astype(np.float32),trainable=False)
        self.run_var1 = tf.Variable(np.zeros(mo1).astype(np.float32),trainable=False)
        #self.b1 = tf.Variable(b1.astype(np.float32))
        self.W2 = tf.Variable(W2.astype(np.float32))
        self.beta2 = tf.Variable(np.zeros(mo2).astype(np.float32))
        self.gamma2 = tf.Variable(np.ones(mo2).astype(np.float32))
        self.run_mean2 = tf.Variable(np.zeros(mo2).astype(np.float32),trainable=False)
        self.run_var2 = tf.Variable(np.zeros(mo2).astype(np.float32),trainable=False)
        #self.b2 = tf.Variable(b2.astype(np.float32))
        #self.params = [self.W1,self.b1,self.W2,self.b2]

    def forward(self, X,TASK,decay=0.9):
        conv_out = tf.nn.conv2d(X , self.W1 , strides = [1,1,1,1],padding="SAME")
        if TASK =="train":
            batch_mean1 , batch_var1 = tf.nn.moments(conv_out,axes=[0,1,2])
            update_rn_mean1 = tf.assign(self.run_mean1,self.run_mean1*decay+batch_mean1*(1-decay))
            update_rn_var1 = tf.assign(self.run_var1, self.run_var1*decay+batch_var1*(1-decay))
            with tf.control_dependencies([update_rn_mean1, update_rn_var1]):
                conv_out = tf.nn.batch_normalization(conv_out,batch_mean1,batch_var1,self.beta1,self.gamma1,1e-4)
            conv_out = tf.nn.relu(conv_out)
            conv2_out = tf.nn.conv2d(conv_out, self.W2, strides=[1,1,1,1],padding="SAME")
            batch_mean2 , batch_var2 = tf.nn.moments(conv2_out,axes=[0,1,2])
            update_rn_mean2 = tf.assign(self.run_mean2,self.run_mean2*decay+batch_mean2*(1-decay))
            update_rn_var2 = tf.assign(self.run_var2, self.run_var2*decay+batch_var2*(1-decay))
            with tf.control_dependencies([update_rn_mean2, update_rn_var2]):
                conv2_out = tf.nn.batch_normalization(conv2_out,batch_mean2,batch_var2,self.beta2,self.gamma2,1e-4)
            conv2_out = tf.nn.relu(conv2_out)
            pool_out  = tf.nn.max_pool(conv2_out, ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")
        else:
            conv_out = tf.nn.batch_normalization(conv_out,self.run_mean1,self.run_var1,self.beta1,self.gamma1,1e-4)
            conv_out = tf.nn.relu(conv_out)
            conv2_out = tf.nn.conv2d(conv_out, self.W2, strides=[1,1,1,1],padding="SAME")
            conv2_out = tf.nn.batch_normalization(conv2_out,self.run_mean2,self.run_var2,self.beta2,self.gamma2,1e-4)
            conv2_out = tf.nn.relu(conv2_out)
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
        #self.b1 = tf.Variable(b1.astype(np.float32))
        self.beta1 = tf.Variable(np.zeros(mo1).astype(np.float32))
        self.gamma1 = tf.Variable(np.ones(mo1).astype(np.float32))
        self.run_mean1 = tf.Variable(np.zeros(mo1).astype(np.float32),trainable=False)
        self.run_var1 = tf.Variable(np.zeros(mo1).astype(np.float32),trainable=False)
        self.W2 = tf.Variable(W2.astype(np.float32))
        self.beta2 = tf.Variable(np.zeros(mo2).astype(np.float32))
        self.gamma2 = tf.Variable(np.ones(mo2).astype(np.float32))
        self.run_mean2 = tf.Variable(np.zeros(mo2).astype(np.float32),trainable=False)
        self.run_var2 = tf.Variable(np.zeros(mo2).astype(np.float32),trainable=False)
        self.W3 = tf.Variable(W3.astype(np.float32))
        self.beta3 = tf.Variable(np.zeros(mo3).astype(np.float32))
        self.gamma3 = tf.Variable(np.ones(mo3).astype(np.float32))
        self.run_mean3 = tf.Variable(np.zeros(mo3).astype(np.float32),trainable=False)
        self.run_var3 = tf.Variable(np.zeros(mo3).astype(np.float32),trainable=False)

    def forward(self,X,TASK,decay=0.9):
        conv_out  = tf.nn.conv2d(X , self.W1 , strides = [1,1,1,1],padding="SAME")
        if TASK=="train":
            batch_mean1 , batch_var1 = tf.nn.moments(conv_out,axes=[0,1,2])
            update_rn_mean1 = tf.assign(self.run_mean1,self.run_mean1*decay+batch_mean1*(1-decay))
            update_rn_var1 = tf.assign(self.run_var1, self.run_var1*decay+batch_var1*(1-decay))
            with tf.control_dependencies([update_rn_mean1, update_rn_var1]):
                conv_out = tf.nn.batch_normalization(conv_out,batch_mean1,batch_var1,self.beta1,self.gamma1,1e-4)
            conv_out  = tf.nn.relu(conv_out)
            conv2_out = tf.nn.conv2d(conv_out, self.W2, strides=[1,1,1,1],padding="SAME")
            batch_mean2 , batch_var2 = tf.nn.moments(conv2_out,axes=[0,1,2])
            update_rn_mean2 = tf.assign(self.run_mean2,self.run_mean2*decay+batch_mean2*(1-decay))
            update_rn_var2 = tf.assign(self.run_var2, self.run_var2*decay+batch_var2*(1-decay))
            with tf.control_dependencies([update_rn_mean2, update_rn_var2]):
                conv2_out = tf.nn.batch_normalization(conv2_out,batch_mean2,batch_var2,self.beta2,self.gamma2,1e-4)

            conv2_out = tf.nn.relu(conv2_out)
            conv3_out = tf.nn.conv2d(conv2_out,self.W3 , strides=[1,1,1,1], padding="SAME")
            batch_mean3 , batch_var3 = tf.nn.moments(conv3_out,axes=[0,1,2])
            update_rn_mean3 = tf.assign(self.run_mean3,self.run_mean3*decay+batch_mean3*(1-decay))
            update_rn_var3 = tf.assign(self.run_var3, self.run_var3*decay+batch_var3*(1-decay))
            with tf.control_dependencies([update_rn_mean3, update_rn_var3]):
                conv3_out = tf.nn.batch_normalization(conv3_out,batch_mean3,batch_var3,self.beta3,self.gamma3,1e-4)

            conv3_out = tf.nn.relu(conv3_out)
            pool_out = tf.nn.max_pool(conv3_out,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")
        else:
            conv_out = tf.nn.batch_normalization(conv_out,self.run_mean1,self.run_var1,self.beta1,self.gamma1,1e-4)
            conv_out  = tf.nn.relu(conv_out)
            conv2_out = tf.nn.conv2d(conv_out, self.W2, strides=[1,1,1,1],padding="SAME")
            conv2_out = tf.nn.batch_normalization(conv2_out,self.run_mean2,self.run_var2,self.beta2,self.gamma2,1e-4)
            conv2_out = tf.nn.relu(conv2_out)
            conv3_out = tf.nn.conv2d(conv2_out,self.W3 , strides=[1,1,1,1], padding="SAME")
            conv3_out = tf.nn.batch_normalization(conv3_out,self.run_mean3,self.run_var3,self.beta3,self.gamma3,1e-4)
            conv3_out = tf.nn.relu(conv3_out)
            pool_out = tf.nn.max_pool(conv3_out,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")
        return pool_out

class HiddenLayers(object):
    def __init__(self, M1 , M2 ):
        W , b = init_weights_bias(M1 , M2)
        self.W = tf.Variable(W.astype(np.float32))
        self.gamma = tf.Variable(np.ones(M2).astype(np.float32))
        self.beta = tf.Variable(np.zeros(M2).astype(np.float32))
        #For testing
        self.running_mean = tf.Variable(np.zeros(M2).astype(np.float32),trainable=False)
        self.running_variance = tf.Variable(np.zeros(M2).astype(np.float32),trainable=False)

    def forward(self,X,TASK,decay=0.9):
        out=tf.matmul(X,self.W)
        if TASK=="train":
            batch_mean, batch_variance = tf.nn.moments(out,axes=[0])
            update_rn_mean = tf.assign(self.running_mean,decay*self.running_mean+(1-decay)*batch_mean)
            update_rn_var = tf.assign(self.running_variance, decay*self.running_variance+(1-decay)*batch_variance)
            with tf.control_dependencies([update_rn_mean,update_rn_var]):
                out = tf.nn.batch_normalization(out,batch_mean,batch_variance,self.beta, self.gamma,1e-4)
        else:
            out = tf.nn.batch_normalization(out, self.running_mean, self.running_variance,self.beta, self.gamma,1e-4)
        return tf.nn.relu(out)

class CNN(object):
    def __init__(self, conv2layersizes , conv3layersizes , hiddenlayersizes, dropout_rate):
        self.fw = 3
        self.fh = 3
        self.conv2layersizes = conv2layersizes
        self.conv3layersizes = conv3layersizes
        self.hiddenlayersizes = hiddenlayersizes
        self.dropout_rate = dropout_rate
    def fit(self,traingenerator,testgenerator,learning_rate =10e-7,mu = 0.9, decay = 0.99,reg=0 ,batch_size=4,show_fig = True):

        dw = 512
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
        #self.params = [self.W , self.b]
        #for c in self.convlayer:
        #    self.params += c.params
        #for h in self.hiddenlayer:
        #    self.params+=h.params

        tfX = tf.placeholder(tf.float32,shape=(None, 512, 512, 1))
        tfT = tf.placeholder(tf.float32,shape=(None , 4))

        act = self.forward(tfX,"train")

        #rcost = reg*sum([tf.nn.l2_loss(p) for p in self.params])
        #cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=act, labels=tfT))  + rcost
        cost = tf.reduce_sum(tf.losses.mean_squared_error(labels=tfT, predictions=act)) #+ rcost

        training_op = tf.train.AdamOptimizer(learning_rate).minimize(cost)
        #training_op = tf.train.RMSPropOptimizer(learning_rate, decay, mu).minimize(cost)

        prediction_op = self.predict(tfX,"test")

        tf.add_to_collection("prediction", prediction_op)

        saver = tf.train.Saver()


        N = 16 # No of train & test bach images loaded
        n_batches = N // batch_size
        max_iter = 10 # default 10

        init = tf.global_variables_initializer()
        p =0

        LL = []
        acc = []
        with tf.Session() as session:
            session.run(init)
            for g in range(0,max_iter):
                for i in range(0, 6901):# default 6901
                    Xtrain , Ytrain = next(traingenerator)
                    train_cost=0
                    for j in range(0, n_batches):
                        Xbatch = Xtrain[j*batch_size:(j+1)*batch_size]
                        Ybatch = Ytrain[j*batch_size:(j+1)*batch_size]

                        session.run(training_op, feed_dict={tfX:Xbatch , tfT:Ybatch})
                        train_cost+=session.run(cost, feed_dict={tfX:Xbatch,tfT:Ybatch})
                    train_cost = train_cost/16
                    print("Epoch %d of %d Iteration %d of 6900 , cost: %f"%(g,max_iter-1,i,train_cost))
                    if i % 100 == 0:
                        for k in range(0,3):
                            Xtest, Ytest,imlist = next(testgenerator)
                            tot_count = 0
                            true_positive=0
                            prediction = np.zeros((len(Xtest),4))
                            c=0
                            for k in range(0,len(Xtest)//batch_size):
                                Xtest_b = Xtest[k*batch_size:(k+1)*batch_size]
                                Ytest_b = Ytest[k*batch_size:(k+1)*batch_size]
                                c+= session.run(cost, feed_dict={tfX:Xtest_b, tfT:Ytest_b})
                                prediction[k*batch_size:(k+1)*batch_size] = session.run(prediction_op, feed_dict={tfX:Xtest_b})

                            c = c/16
                            LL.append(c)
                            true_positive+=draw_img(Xtest,prediction,Ytest,imlist)
                            tot_count =len(Xtest)
                            p+=1
                            print(true_positive,tot_count)
                            accuracy = (true_positive/tot_count)*100
                            acc.append(accuracy)
                            print("batch %d of 1221 cost: %d, accuracy: %d percent" %(p,c, accuracy))
            os.chdir("/path/to/saving/models/")

            saver.save(session,"modelfinal_bn")
            plt.plot(LL, label="cost")
            plt.legend()
            plt.show()
            plt.plot(acc, label="accuracy")
            plt.legend()
            plt.show()


    def forward(self, X,TASK):
        Z = X
        for c in self.convlayer:
            Z = c.forward(Z,TASK)
            tf.add_to_collection("final_convlayer",Z)
        Z_shape= Z.get_shape().as_list()
        Z = tf.reshape(Z, [-1 , np.prod(Z_shape[1:])])
        Z = tf.nn.dropout(Z,self.dropout_rate[0])
        for h,p in zip(self.hiddenlayer, self.dropout_rate[1:]):
            Z = h.forward(Z,TASK)
            Z = tf.nn.dropout(Z,p)
        return tf.matmul(Z, self.W) + self.b
    def predict(self,X,TASK):
        X = self.forward(X,TASK)
        return X



def main():
    model = CNN([(1,64,64,64),(64,128,128,128)],
                [(128,256,256,256,256,256),(256,512,512,512,512,512),(512, 512, 512, 512, 512, 512),
                (512,512,512,512,512,512)],
                [4096,4096],[0.8,0.5,0.5])
    traingenerator = traindatagen()
    testgenerator = testdatagen()

    model.fit(traingenerator,testgenerator)

if __name__ =="__main__":
    main()
