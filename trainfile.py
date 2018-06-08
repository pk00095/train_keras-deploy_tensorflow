import tensorflow as tf
import os
from tfutils import Recorder
from tensorflow.python.keras.models import Model, load_model,Sequential
from tensorflow.python.keras import optimizers
from tensorflow.python.keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from tensorflow.python.keras.applications.mobilenet import MobileNet
from tensorflow.python.keras.utils import CustomObjectScope
from tensorflow.python.keras.backend import relu 

tf.logging.set_verbosity(tf.logging.INFO)

def relu6(x):
  return relu(x, max_value=6)


with CustomObjectScope({'relu6': relu6}):
  base_model = MobileNet(weights='imagenet', include_top=False,input_shape=(224, 224, 3))
  mid_start = 5;
  all_layers =base_model.layers
  for i in range(0 , mid_start):
     print(i)
     # all_layers[i].add(Flatten())
     all_layers[i].trainable = False;

  #x=base_model.output
  #x=Flatten()(x)
  #x=Dense(128,activation='relu')(x)
  #x=Dropout(0.5)(x)
  #pred = Dense(2,activation='softmax')(x)
  #print base_model.summary()


  model = Sequential()
  model.add(base_model)
  model.add(Flatten())
  model.add(Dense(128, activation='relu'))
  model.add(Dense(1, activation='softmax'))
  #conv_base.trainable = False
  model.compile(loss='binary_crossentropy',
              optimizer=tf.keras.optimizers.RMSprop(lr=2e-5),
              metrics=['acc'])


  model_dir = os.path.join(os.getcwd(), "models_catvsdog")
  if not os.path.exists(model_dir):
    os.makedirs(model_dir)

  est_catvsdog = tf.keras.estimator.model_to_estimator(keras_model=model,model_dir=model_dir)
  #exit()

  rec=Recorder()

  train_spec = tf.estimator.TrainSpec(input_fn =  lambda : rec.imgs_input_fn('/home/gmind/graymatics/development/Pratik_dev/tensorflow_experiments/output/cars.tfrecords',
                                                                   height = 224,
                                                                   width = 224,
                                                                   shuffle=True,
                                                                   repeat_count=5,
                                                                   batch_size=32), 
                                                                   max_steps=200)

  eval_spec = tf.estimator.EvalSpec(input_fn=lambda: rec.imgs_input_fn('/home/gmind/graymatics/development/Pratik_dev/tensorflow_experiments/output/cars.tfrecords',
                                                                 height = 224,
                                                                 width = 224,
                                                                 shuffle=False,
                                                                 batch_size=1))


  tf.estimator.train_and_evaluate(est_catvsdog, train_spec, eval_spec)
  print('done')
