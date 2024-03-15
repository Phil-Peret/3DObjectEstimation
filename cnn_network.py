import matplotlib.pyplot as plt
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import pickle
import numpy as np
from tensorflow import keras
from keras import layers
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from scipy.spatial.transform import Rotation as R
from keras import backend as K
import rotation6d 
from keras import optimizers



def geodistic_distance_matrix_loss(y_true, y_pred):
    y_pred_m=tf.reshape(rotation6d.tf_rotation6d_to_matrix(y_pred), [-1,3,3])
    y_true_m=tf.reshape(rotation6d.tf_rotation6d_to_matrix(y_true), [-1,3,3])
    y_true_m_transposed = tf.transpose(y_true_m, perm=[0, 2, 1])

    # Calcolo della traccia della differenza tra le matrici
    trace_diff = tf.linalg.trace(tf.matmul(y_true_m_transposed, y_pred_m))

    # Clipping
    trace_diff = tf.clip_by_value(trace_diff, -1.0, 3.0)

    # Calcolo dell'angolo differenziale
    angle_diff = tf.math.acos(tf.clip_by_value(0.5 * (trace_diff - 1.0), -1.0, 1.0))

    # Calcolo della distanza geodetica media nel batch
    loss = tf.reduce_mean(angle_diff)
    return loss


#file = open('/content/drive/MyDrive/test6d_wo_z.pickle', 'rb')
file = open('test6d_wo_z.pickle', 'rb')
data = pickle.load(file)
file.close()
IMG_SHAPE = (100, 100, 3)

X0 = []
X1 = []
X2 = []
X3 = []
X4 = []
X5 = []
y0 = []


for element in data:
    X0.append(element[0][0])
    X1.append(element[0][1])
    X2.append(element[0][2])
    X3.append(element[0][3])
    X4.append(element[0][4])
    X5.append(element[0][5])
    y0.append(element[1])


X0 = np.asarray(X0)
X1 = np.asarray(X1)
X2 = np.asarray(X2)
X3 = np.asarray(X3)
X4 = np.asarray(X4)
X5 = np.asarray(X5)
y0 = np.asarray(y0)
#y0 = np.reshape(y0,(y0.shape[0],-1))

AUTOTUNE = tf.data.AUTOTUNE

X0_train, X0_test, X1_train, X1_test, X2_train, X2_test, X3_train, X3_test, X4_train, X4_test,\
X5_train, X5_test, y0_train, y0_test = train_test_split(X0,X1,X2,X3,X4,X5,y0,test_size=0.1)
#X0_train, X0_test, y0_train, y0_test= train_test_split(X0,y0,test_size=0.1)


def build_siamese_model(input_shape):
    input_0 = layers.Input(input_shape)
    resizing = layers.Resizing(100,100, "bilinear")(input_0)
    rescaling = layers.Rescaling(scale=1./255)(resizing)
    conv_0 = layers.Conv2D(4, 3, padding='same', activation='relu')(rescaling)
    max_0 = layers.MaxPooling2D()(conv_0)
    conv_1 = layers.Conv2D(8, 3, padding='same', activation='relu')(max_0)
    max_1 = layers.MaxPooling2D()(conv_1)
    conv_2 = layers.Conv2D(16, 3, padding='same', activation='relu')(max_1)
    max_2 = layers.MaxPooling2D()(conv_2)
    #conv_3 = layers.Conv2D(6, 3, padding='same', activation='relu')(max_2)
    #max_3 = layers.MaxPooling2D()(conv_3)
    #conv_2 = layers.Conv2D(256, 3, padding='same', activation='relu')(max_1)
    #max_2 = layers.MaxPooling2D()(conv_2)
    #global_avg = layers.GlobalAveragePooling2D()(max_1)
    flatten = layers.Flatten()(max_2)
    model_6d = keras.Model(inputs=input_0,outputs=flatten)
    return model_6d
#6d loss


feature_extractor=build_siamese_model(IMG_SHAPE)
img_0 = layers.Input(shape=IMG_SHAPE)
img_1 = layers.Input(shape=IMG_SHAPE)
img_2 = layers.Input(shape=IMG_SHAPE)
img_3 = layers.Input(shape=IMG_SHAPE)
img_4 = layers.Input(shape=IMG_SHAPE)
img_5 = layers.Input(shape=IMG_SHAPE)
feature_0 = feature_extractor(img_0)
feature_1 = feature_extractor(img_1)
feature_2 = feature_extractor(img_2)
feature_3 = feature_extractor(img_3)
feature_4 = feature_extractor(img_4)
feature_5 = feature_extractor(img_5)


concat = layers.concatenate([feature_0, feature_1, feature_2, feature_3, feature_4, feature_5])
dense_0 = layers.Dense(256, activation='relu')(concat)
dropout_0 = layers.Dropout(0.5)(dense_0)
dense_1 = layers.Dense(128, activation='relu')(dropout_0)
dropout_1 = layers.Dropout(0.35)(dense_1)
dense_2 = layers.Dense(64, activation='relu')(dropout_1)
dropout_2 = layers.Dropout(0.2)(dense_2)
dense_3 = layers.Dense(32, activation='relu')(dropout_2)
output = layers.Dense(6, activation='tanh', name='6d_rotation')(dense_3)

model_6d = keras.Model(inputs=[img_0, img_1, img_2, img_3, img_4, img_5], outputs=output)

custom_optimizer = optimizers.Adam(learning_rate=0.001)

model_6d.compile(optimizer='adam',
            loss = geodistic_distance_matrix_loss)

print(model_6d.summary())

history_6d = model_6d.fit([X0_train, X1_train, X2_train, X3_train, X4_train, X5_train], y0_train, batch_size=10, \
                        epochs=100, validation_data=([X0_test, X1_test, X2_test, X3_test, X4_test, X5_test],y0_test))

model_6d.save('trained_orientation6d.keras')

figure, loss = plt.subplots(2,1)
loss[0].plot(history_6d.epoch, history_6d.history["loss"], 'g', label='geodistic distance')
loss[0].set_title('Training')
loss[0].set_xlabel('Epochs')
loss[0].set_ylabel('Loss value')
loss[0].legend()
loss[1].plot(history_6d.epoch, history_6d.history["val_loss"], 'g', label='geodistic distance')
loss[1].set_title('Test')
loss[1].set_xlabel('Epochs')
loss[1].set_ylabel('Loss value')
loss[1].legend()
plt.show()
