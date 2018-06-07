import tensorflow as tf
import sys
import time
from PIL import Image
import numpy as np

class Recorder:

 def wrap_int64(self,value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

 def wrap_bytes(self,value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


 def convert(self,image_paths,labels,out_path,size=None):
    '''Args:
    image_paths : List of file-paths for the images
    labels : class-labels for images(vector of size len(image_paths)X1)
    out_path : Destination of TFRecords output file
    size : expected images size
    '''
    
    num_images = len(image_paths)
    
    with tf.python_io.TFRecordWriter(out_path) as writer :

       for i,(path,label) in enumerate(zip(image_paths,labels)):

           sys.stdout.write('\rProcessed :: {} outof ::{}'.format(i+1,num_images))
           sys.stdout.flush()
           #time.sleep(1)

           #img = cv2.imread(path)
           #extension = '.'+path.split('/')[-1].split('.')[-1]
           #img = cv2.resize(img,size,interpolation=cv2.INTER_AREA)
           # convert the image to raw Bytes
           #image_bytes = cv2.imencode(extension,img)[1].tostring()
           img = Image.open(path)
           img = np.array(img.resize((224,224)))
           image_bytes = img.tostring()
           # dictionary to store image data and image label
           data={
               'image':self.wrap_bytes(image_bytes),
               'label':self.wrap_int64(label)
                  }

           # Wrap the data as Tensorflow Feature.
           feature = tf.train.Features(feature=data)

           # Wrap again as a Tensorflow Example.
           example = tf.train.Example(features=feature)

           # Serialize the data
           serialized = example.SerializeToString()

           # Write the serialized 
           writer.write(serialized)


 def imgs_input_fn(self,filenames,height,width,shuffle=False,repeat_count=1,batch_size=32):

    def _parse_function(serialized,height=224,width=224):
       features = \
       {
          'image' : tf.FixedLenFeature([],tf.string),
          'label' : tf.FixedLenFeature([], tf.int64)
       }

       parsed_example = tf.parse_single_example(serialized=serialized,
                                                features=features)

       image_shape = tf.stack([height,width,3])
       image_raw = parsed_example['image']
       label = tf.cast(parsed_example['label'],tf.float32)

       # decode the raw bytes so it becomes a tensor with type

       image = tf.decode_raw(image_raw,tf.uint8)
       image = tf.cast(image,tf.float32)
       image = tf.reshape(image, image_shape)
       image = tf.subtract(image, 116.779)
       #d = dict(zip(['image_name.jpg'],[image])) , [label]
       d=image,label
       return d
    
    dataset = tf.data.TFRecordDataset(filenames=filenames)
    # Parse the serialised data to TFRecords files.
    # returns Tensorflow tensors for the image and labels.
    dataset = dataset.map(_parse_function)

    if shuffle:
        dataset = dataset.shuffle(buffer_size = 256)
    
    dataset = dataset.repeat(repeat_count) # Repeat the dataset this time
    batch_dataset = dataset.batch(32)    # Batch Size
    iterator = batch_dataset.make_one_shot_iterator()   # Make an iterator
    batch_features,batch_labels = iterator.get_next()  # Tensors to get next batch of image and their labels
    
    return batch_features, batch_labels



# Referred from https://www.dlology.com/blog/how-to-leverage-tensorflows-tfrecord-to-train-keras-model/?t=152826568954#rating-13
