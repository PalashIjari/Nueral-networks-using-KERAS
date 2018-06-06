# -*- coding: utf-8 -*-
"""
Created on Tue May  1 01:06:36 2018

@author: Palash
"""



import numpy
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
K.set_image_dim_ordering('th')

seed = 7
numpy.random.seed(seed)




# load data
(X_train, y_train), (X_test, y_test) = mnist.load_data()
# reshape to be [samples][pixels][width][height]
X_train = X_train.reshape(X_train.shape[0], 1, 28, 28).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 1, 28, 28).astype('float32')




# normalize inputs from 0-255 to 0-1
X_train = X_train / 255
X_test = X_test / 255
# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]




def baseline_model():
	# create model
	model = Sequential()
	model.add(Conv2D(32, (5, 5), input_shape=(1, 28, 28), activation='relu'))
	model.add(MaxPooling2D(pool_size=(4, 4)))
	model.add(Dropout(0.25))
	model.add(Flatten())
	model.add(Dense(128, activation='relu'))
	model.add(Dense(num_classes, activation='softmax'))
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

# build the model
model = baseline_model()
# Fit the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=200, verbose=2)
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("CNN Error: %.2f%%" % (100-scores[1]*100))



#Train on 60000 samples, validate on 10000 samples
#Epoch 1/10
#
# - 243s - loss: 0.2345 - acc: 0.9334 - val_loss: 0.0844 - val_acc: 0.9732
#Epoch 2/10
# - 233s - loss: 0.0752 - acc: 0.9776 - val_loss: 0.0476 - val_acc: 0.9838
#Epoch 3/10
# - 210s - loss: 0.0541 - acc: 0.9836 - val_loss: 0.0440 - val_acc: 0.9863
#Epoch 4/10
# - 203s - loss: 0.0408 - acc: 0.9875 - val_loss: 0.0399 - val_acc: 0.9874
#Epoch 5/10
# - 203s - loss: 0.0338 - acc: 0.9896 - val_loss: 0.0344 - val_acc: 0.9882
#Epoch 6/10
# - 200s - loss: 0.0278 - acc: 0.9913 - val_loss: 0.0320 - val_acc: 0.9890
#Epoch 7/10
# - 240s - loss: 0.0236 - acc: 0.9927 - val_loss: 0.0358 - val_acc: 0.9884
#Epoch 8/10
# - 207s - loss: 0.0206 - acc: 0.9936 - val_loss: 0.0335 - val_acc: 0.9886
#Epoch 9/10
# - 209s - loss: 0.0170 - acc: 0.9944 - val_loss: 0.0314 - val_acc: 0.9898
#Epoch 10/10
# - 202s - loss: 0.0148 - acc: 0.9957 - val_loss: 0.0328 - val_acc: 0.9896
#CNN Error: 1.04%


#training with pool size 3x3
#Train on 60000 samples, validate on 10000 samples
#Epoch 1/10
# - 195s - loss: 0.2763 - acc: 0.9224 - val_loss: 0.0873 - val_acc: 0.9744
#Epoch 2/10
# - 193s - loss: 0.0852 - acc: 0.9746 - val_loss: 0.0501 - val_acc: 0.9841
#Epoch 3/10
# - 174s - loss: 0.0627 - acc: 0.9808 - val_loss: 0.0488 - val_acc: 0.9844
#Epoch 4/10
# - 173s - loss: 0.0494 - acc: 0.9845 - val_loss: 0.0380 - val_acc: 0.9876
#Epoch 5/10
# - 174s - loss: 0.0420 - acc: 0.9873 - val_loss: 0.0351 - val_acc: 0.9883
#Epoch 6/10
# - 175s - loss: 0.0352 - acc: 0.9891 - val_loss: 0.0302 - val_acc: 0.9903
#Epoch 7/10
# - 174s - loss: 0.0318 - acc: 0.9904 - val_loss: 0.0303 - val_acc: 0.9899
#Epoch 8/10
# - 193s - loss: 0.0280 - acc: 0.9914 - val_loss: 0.0275 - val_acc: 0.9909
#Epoch 9/10
# - 184s - loss: 0.0243 - acc: 0.9923 - val_loss: 0.0283 - val_acc: 0.9906
#Epoch 10/10
# - 174s - loss: 0.0211 - acc: 0.9934 - val_loss: 0.0265 - val_acc: 0.9912
#CNN Error: 0.88%


#activation with tanh and pool size 3x3
#Train on 60000 samples, validate on 10000 samples
#Epoch 1/10
# - 181s - loss: 0.2625 - acc: 0.9257 - val_loss: 0.0882 - val_acc: 0.9738
#Epoch 2/10
# - 193s - loss: 0.0812 - acc: 0.9761 - val_loss: 0.0538 - val_acc: 0.9831
#Epoch 3/10
# - 182s - loss: 0.0582 - acc: 0.9827 - val_loss: 0.0483 - val_acc: 0.9846
#Epoch 4/10
# - 180s - loss: 0.0465 - acc: 0.9854 - val_loss: 0.0411 - val_acc: 0.9865
#Epoch 5/10
# - 179s - loss: 0.0374 - acc: 0.9887 - val_loss: 0.0391 - val_acc: 0.9869
#Epoch 6/10
# - 178s - loss: 0.0307 - acc: 0.9907 - val_loss: 0.0368 - val_acc: 0.9886
#Epoch 7/10
# - 179s - loss: 0.0257 - acc: 0.9923 - val_loss: 0.0351 - val_acc: 0.9888
#Epoch 8/10
# - 191s - loss: 0.0224 - acc: 0.9933 - val_loss: 0.0330 - val_acc: 0.9893
#Epoch 9/10
# - 188s - loss: 0.0190 - acc: 0.9941 - val_loss: 0.0302 - val_acc: 0.9907
#Epoch 10/10
# - 181s - loss: 0.0153 - acc: 0.9958 - val_loss: 0.0311 - val_acc: 0.9900
#CNN Error: 1.00%


#activation function used is sigmoid
#Train on 60000 samples, validate on 10000 samples
#Epoch 1/10
# - 175s - loss: 1.2668 - acc: 0.6144 - val_loss: 0.4398 - val_acc: 0.8897
#Epoch 2/10
# - 176s - loss: 0.3834 - acc: 0.8909 - val_loss: 0.2592 - val_acc: 0.9260
#Epoch 3/10
# - 191s - loss: 0.2672 - acc: 0.9229 - val_loss: 0.1886 - val_acc: 0.9436
#Epoch 4/10
# - 180s - loss: 0.2031 - acc: 0.9399 - val_loss: 0.1487 - val_acc: 0.9579
#Epoch 5/10
# - 175s - loss: 0.1655 - acc: 0.9503 - val_loss: 0.1154 - val_acc: 0.9661
#Epoch 6/10
# - 173s - loss: 0.1394 - acc: 0.9589 - val_loss: 0.0975 - val_acc: 0.9713
#Epoch 7/10
# - 173s - loss: 0.1212 - acc: 0.9645 - val_loss: 0.0830 - val_acc: 0.9742
#Epoch 8/10
# - 174s - loss: 0.1089 - acc: 0.9677 - val_loss: 0.0736 - val_acc: 0.9774
#Epoch 9/10
# - 183s - loss: 0.0951 - acc: 0.9716 - val_loss: 0.0696 - val_acc: 0.9782
#Epoch 10/10
# - 185s - loss: 0.0875 - acc: 0.9736 - val_loss: 0.0635 - val_acc: 0.9804
#CNN Error: 1.96%


#training with 5 epochs
#Train on 60000 samples, validate on 10000 samples
#Epoch 1/5
# - 186s - loss: 0.2754 - acc: 0.9222 - val_loss: 0.0868 - val_acc: 0.9741
#Epoch 2/5
# - 189s - loss: 0.0849 - acc: 0.9745 - val_loss: 0.0503 - val_acc: 0.9836
#Epoch 3/5
# - 179s - loss: 0.0621 - acc: 0.9811 - val_loss: 0.0482 - val_acc: 0.9837
#Epoch 4/5
# - 183s - loss: 0.0488 - acc: 0.9847 - val_loss: 0.0371 - val_acc: 0.9871
#Epoch 5/5
# - 178s - loss: 0.0413 - acc: 0.9873 - val_loss: 0.0345 - val_acc: 0.9883
#CNN Error: 1.17%

#training with 7 epocs
#Train on 60000 samples, validate on 10000 samples
#Epoch 1/7
# - 184s - loss: 0.2752 - acc: 0.9226 - val_loss: 0.0874 - val_acc: 0.9739
#Epoch 2/7
# - 198s - loss: 0.0847 - acc: 0.9748 - val_loss: 0.0503 - val_acc: 0.9842
#Epoch 3/7
# - 246s - loss: 0.0623 - acc: 0.9808 - val_loss: 0.0483 - val_acc: 0.9848
#Epoch 4/7
# - 272s - loss: 0.0490 - acc: 0.9849 - val_loss: 0.0375 - val_acc: 0.9876
#Epoch 5/7
# - 263s - loss: 0.0416 - acc: 0.9876 - val_loss: 0.0346 - val_acc: 0.9888
#Epoch 6/7
# - 249s - loss: 0.0351 - acc: 0.9892 - val_loss: 0.0302 - val_acc: 0.9900
#Epoch 7/7
# - 249s - loss: 0.0316 - acc: 0.9903 - val_loss: 0.0293 - val_acc: 0.9904
#CNN Error: 0.96%


#with 9 epochs
#Train on 60000 samples, validate on 10000 samples
#Epoch 1/9
# - 178s - loss: 0.2757 - acc: 0.9224 - val_loss: 0.0871 - val_acc: 0.9743
#Epoch 2/9
# - 191s - loss: 0.0848 - acc: 0.9747 - val_loss: 0.0504 - val_acc: 0.9845
#Epoch 3/9
# - 193s - loss: 0.0623 - acc: 0.9811 - val_loss: 0.0488 - val_acc: 0.9845
#Epoch 4/9
# - 178s - loss: 0.0490 - acc: 0.9848 - val_loss: 0.0380 - val_acc: 0.9875
#Epoch 5/9
# - 177s - loss: 0.0419 - acc: 0.9873 - val_loss: 0.0359 - val_acc: 0.9885
#Epoch 6/9
# - 185s - loss: 0.0353 - acc: 0.9891 - val_loss: 0.0308 - val_acc: 0.9904
#Epoch 7/9
# - 183s - loss: 0.0316 - acc: 0.9904 - val_loss: 0.0302 - val_acc: 0.9898
#Epoch 8/9
# - 192s - loss: 0.0277 - acc: 0.9916 - val_loss: 0.0279 - val_acc: 0.9910
#Epoch 9/9
# - 202s - loss: 0.0244 - acc: 0.9923 - val_loss: 0.0282 - val_acc: 0.9906
#CNN Error: 0.94%

#with 11 epocs and 3x3
#Train on 60000 samples, validate on 10000 samples
#Epoch 1/11
# - 189s - loss: 0.2746 - acc: 0.9229 - val_loss: 0.0875 - val_acc: 0.9740
#Epoch 2/11
# - 200s - loss: 0.0845 - acc: 0.9749 - val_loss: 0.0497 - val_acc: 0.9838
#Epoch 3/11
# - 193s - loss: 0.0620 - acc: 0.9811 - val_loss: 0.0484 - val_acc: 0.9847
#Epoch 4/11
# - 187s - loss: 0.0487 - acc: 0.9851 - val_loss: 0.0376 - val_acc: 0.9880
#Epoch 5/11
# - 177s - loss: 0.0414 - acc: 0.9875 - val_loss: 0.0351 - val_acc: 0.9884
#Epoch 6/11
# - 181s - loss: 0.0349 - acc: 0.9892 - val_loss: 0.0305 - val_acc: 0.9901
#Epoch 7/11
# - 176s - loss: 0.0316 - acc: 0.9905 - val_loss: 0.0295 - val_acc: 0.9900
#Epoch 8/11
# - 186s - loss: 0.0273 - acc: 0.9917 - val_loss: 0.0271 - val_acc: 0.9908
#Epoch 9/11
# - 186s - loss: 0.0240 - acc: 0.9926 - val_loss: 0.0281 - val_acc: 0.9903
#Epoch 10/11
# - 184s - loss: 0.0208 - acc: 0.9933 - val_loss: 0.0273 - val_acc: 0.9907
#Epoch 11/11
# - 183s - loss: 0.0181 - acc: 0.9941 - val_loss: 0.0276 - val_acc: 0.9905
#CNN Error: 0.95%

#with batch size 100
#runfile('C:/studyy time/!!!masters!!/669-Dmachine learning/Project2/keras.py', wdir='C:/studyy time/!!!masters!!/669-Dmachine learning/Project2')
#Train on 60000 samples, validate on 10000 samples
#Epoch 1/11
# - 189s - loss: 0.2183 - acc: 0.9361 - val_loss: 0.0785 - val_acc: 0.9751
#Epoch 2/11
# - 184s - loss: 0.0728 - acc: 0.9775 - val_loss: 0.0416 - val_acc: 0.9868
#Epoch 3/11
# - 182s - loss: 0.0516 - acc: 0.9837 - val_loss: 0.0390 - val_acc: 0.9867
#Epoch 4/11
# - 200s - loss: 0.0409 - acc: 0.9875 - val_loss: 0.0309 - val_acc: 0.9893
#Epoch 5/11
# - 186s - loss: 0.0333 - acc: 0.9899 - val_loss: 0.0300 - val_acc: 0.9891
#Epoch 6/11
# - 187s - loss: 0.0279 - acc: 0.9908 - val_loss: 0.0317 - val_acc: 0.9897
#Epoch 7/11
# - 186s - loss: 0.0251 - acc: 0.9920 - val_loss: 0.0338 - val_acc: 0.9887
#Epoch 8/11
# - 174s - loss: 0.0210 - acc: 0.9931 - val_loss: 0.0271 - val_acc: 0.9911
#Epoch 9/11
# - 181s - loss: 0.0183 - acc: 0.9940 - val_loss: 0.0256 - val_acc: 0.9915
#Epoch 10/11
# - 191s - loss: 0.0167 - acc: 0.9948 - val_loss: 0.0277 - val_acc: 0.9903
#Epoch 11/11
# - 176s - loss: 0.0133 - acc: 0.9955 - val_loss: 0.0300 - val_acc: 0.9898
#CNN Error: 1.02%


#with dropout 10%
#runfile('C:/studyy time/!!!masters!!/669-Dmachine learning/Project2/keras.py', wdir='C:/studyy time/!!!masters!!/669-Dmachine learning/Project2')
#Train on 60000 samples, validate on 10000 samples
#Epoch 1/10
# - 174s - loss: 0.2748 - acc: 0.9230 - val_loss: 0.0892 - val_acc: 0.9727
#Epoch 2/10
# - 174s - loss: 0.0798 - acc: 0.9762 - val_loss: 0.0520 - val_acc: 0.9831
#Epoch 3/10
# - 174s - loss: 0.0577 - acc: 0.9822 - val_loss: 0.0467 - val_acc: 0.9851
#Epoch 4/10
# - 177s - loss: 0.0449 - acc: 0.9863 - val_loss: 0.0410 - val_acc: 0.9864
#Epoch 5/10
# - 191s - loss: 0.0372 - acc: 0.9887 - val_loss: 0.0362 - val_acc: 0.9872
#Epoch 6/10
# - 172s - loss: 0.0308 - acc: 0.9905 - val_loss: 0.0324 - val_acc: 0.9891
#Epoch 7/10
# - 173s - loss: 0.0280 - acc: 0.9916 - val_loss: 0.0326 - val_acc: 0.9886
#Epoch 8/10
# - 172s - loss: 0.0241 - acc: 0.9922 - val_loss: 0.0287 - val_acc: 0.9910
#Epoch 9/10
# - 172s - loss: 0.0208 - acc: 0.9934 - val_loss: 0.0274 - val_acc: 0.9915
#Epoch 10/10
# - 173s - loss: 0.0169 - acc: 0.9945 - val_loss: 0.0288 - val_acc: 0.9904
#CNN Error: 0.96%

#with dropout 15%
#runfile('C:/studyy time/!!!masters!!/669-Dmachine learning/Project2/keras.py', wdir='C:/studyy time/!!!masters!!/669-Dmachine learning/Project2')
#Train on 60000 samples, validate on 10000 samples
#Epoch 1/10
# - 172s - loss: 0.2715 - acc: 0.9241 - val_loss: 0.0871 - val_acc: 0.9728
#Epoch 2/10
# - 172s - loss: 0.0817 - acc: 0.9755 - val_loss: 0.0501 - val_acc: 0.9844
#Epoch 3/10
# - 172s - loss: 0.0598 - acc: 0.9815 - val_loss: 0.0464 - val_acc: 0.9852
#Epoch 4/10
# - 183s - loss: 0.0473 - acc: 0.9855 - val_loss: 0.0386 - val_acc: 0.9869
#Epoch 5/10
# - 183s - loss: 0.0396 - acc: 0.9881 - val_loss: 0.0340 - val_acc: 0.9881
#Epoch 6/10
# - 171s - loss: 0.0329 - acc: 0.9899 - val_loss: 0.0332 - val_acc: 0.9894
#Epoch 7/10
# - 171s - loss: 0.0297 - acc: 0.9909 - val_loss: 0.0327 - val_acc: 0.9896
#Epoch 8/10
# - 171s - loss: 0.0257 - acc: 0.9921 - val_loss: 0.0291 - val_acc: 0.9903
#Epoch 9/10
# - 171s - loss: 0.0225 - acc: 0.9926 - val_loss: 0.0274 - val_acc: 0.9908
#Epoch 10/10
# - 176s - loss: 0.0188 - acc: 0.9940 - val_loss: 0.0270 - val_acc: 0.9900
#CNN Error: 1.00%

#with dropout of 25%
#Train on 60000 samples, validate on 10000 samples
#Epoch 1/10
# - 173s - loss: 0.2817 - acc: 0.9196 - val_loss: 0.0879 - val_acc: 0.9738
#Epoch 2/10
# - 172s - loss: 0.0884 - acc: 0.9736 - val_loss: 0.0499 - val_acc: 0.9843
#Epoch 3/10
# - 172s - loss: 0.0644 - acc: 0.9806 - val_loss: 0.0475 - val_acc: 0.9843
#Epoch 4/10
# - 172s - loss: 0.0511 - acc: 0.9838 - val_loss: 0.0385 - val_acc: 0.9870
#Epoch 5/10
# - 172s - loss: 0.0429 - acc: 0.9865 - val_loss: 0.0345 - val_acc: 0.9882
#Epoch 6/10
# - 192s - loss: 0.0364 - acc: 0.9884 - val_loss: 0.0302 - val_acc: 0.9896
#Epoch 7/10
# - 177s - loss: 0.0329 - acc: 0.9898 - val_loss: 0.0287 - val_acc: 0.9899
#Epoch 8/10
# - 172s - loss: 0.0285 - acc: 0.9908 - val_loss: 0.0263 - val_acc: 0.9917
#Epoch 9/10
# - 172s - loss: 0.0248 - acc: 0.9925 - val_loss: 0.0278 - val_acc: 0.9903
#Epoch 10/10
# - 171s - loss: 0.0221 - acc: 0.9927 - val_loss: 0.0264 - val_acc: 0.9908
#CNN Error: 0.92%


#with elu
#Epoch 1/10
# - 181s - loss: 0.2579 - acc: 0.9254 - val_loss: 0.0854 - val_acc: 0.9739
#Epoch 2/10
# - 199s - loss: 0.0850 - acc: 0.9746 - val_loss: 0.0520 - val_acc: 0.9834
#Epoch 3/10
# - 176s - loss: 0.0642 - acc: 0.9805 - val_loss: 0.0485 - val_acc: 0.9838
#Epoch 4/10
# - 175s - loss: 0.0534 - acc: 0.9830 - val_loss: 0.0419 - val_acc: 0.9853
#Epoch 5/10
# - 175s - loss: 0.0440 - acc: 0.9860 - val_loss: 0.0385 - val_acc: 0.9869
#Epoch 6/10
# - 175s - loss: 0.0371 - acc: 0.9884 - val_loss: 0.0363 - val_acc: 0.9889
#Epoch 7/10
# - 175s - loss: 0.0320 - acc: 0.9900 - val_loss: 0.0332 - val_acc: 0.9889
#Epoch 8/10
# - 196s - loss: 0.0278 - acc: 0.9912 - val_loss: 0.0307 - val_acc: 0.9903
#Epoch 9/10
# - 179s - loss: 0.0244 - acc: 0.9920 - val_loss: 0.0292 - val_acc: 0.9899
#Epoch 10/10
# - 174s - loss: 0.0207 - acc: 0.9931 - val_loss: 0.0287 - val_acc: 0.9905
#CNN Error: 0.95%

#with softplus
#Train on 60000 samples, validate on 10000 samples
#Epoch 1/10
# - 183s - loss: 1.2096 - acc: 0.5818 - val_loss: 0.2897 - val_acc: 0.9138
#Epoch 2/10
# - 181s - loss: 0.3084 - acc: 0.9041 - val_loss: 0.1713 - val_acc: 0.9488
#Epoch 3/10
# - 181s - loss: 0.2075 - acc: 0.9356 - val_loss: 0.1203 - val_acc: 0.9623
#Epoch 4/10
# - 181s - loss: 0.1580 - acc: 0.9514 - val_loss: 0.0909 - val_acc: 0.9724
#Epoch 5/10
# - 195s - loss: 0.1311 - acc: 0.9579 - val_loss: 0.0798 - val_acc: 0.9744
#Epoch 6/10
# - 193s - loss: 0.1153 - acc: 0.9635 - val_loss: 0.0656 - val_acc: 0.9802
#Epoch 7/10
# - 180s - loss: 0.0986 - acc: 0.9695 - val_loss: 0.0557 - val_acc: 0.9823
#Epoch 8/10
# - 181s - loss: 0.0904 - acc: 0.9715 - val_loss: 0.0497 - val_acc: 0.9828
#Epoch 9/10
# - 180s - loss: 0.0822 - acc: 0.9743 - val_loss: 0.0550 - val_acc: 0.9814
#Epoch 10/10
# - 181s - loss: 0.0781 - acc: 0.9751 - val_loss: 0.0476 - val_acc: 0.9854
#CNN Error: 1.46%

#stride 4x4
#Train on 60000 samples, validate on 10000 samples
#Epoch 1/10
# - 171s - loss: 0.3398 - acc: 0.9014 - val_loss: 0.0953 - val_acc: 0.9723
#Epoch 2/10
# - 167s - loss: 0.1074 - acc: 0.9676 - val_loss: 0.0584 - val_acc: 0.9833
#Epoch 3/10
# - 170s - loss: 0.0785 - acc: 0.9754 - val_loss: 0.0486 - val_acc: 0.9839
#Epoch 4/10
# - 176s - loss: 0.0649 - acc: 0.9796 - val_loss: 0.0412 - val_acc: 0.9873
#Epoch 5/10
# - 188s - loss: 0.0537 - acc: 0.9832 - val_loss: 0.0343 - val_acc: 0.9881
#Epoch 6/10
# - 169s - loss: 0.0467 - acc: 0.9852 - val_loss: 0.0320 - val_acc: 0.9890
#Epoch 7/10
# - 173s - loss: 0.0419 - acc: 0.9871 - val_loss: 0.0319 - val_acc: 0.9896
#Epoch 8/10
# - 173s - loss: 0.0374 - acc: 0.9884 - val_loss: 0.0275 - val_acc: 0.9906
#Epoch 9/10
# - 171s - loss: 0.0340 - acc: 0.9894 - val_loss: 0.0258 - val_acc: 0.9915
#Epoch 10/10
# - 181s - loss: 0.0310 - acc: 0.9900 - val_loss: 0.0263 - val_acc: 0.9908
#CNN Error: 0.92%

