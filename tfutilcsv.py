import tensorflow as tf
import pandas as pd
import numpy as np

class Recorder:

 def wrap_int64(self,value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

 def wrap_bytes(self,value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

 def _floats_feature(value):
   return tf.train.Feature(float_list=tf.train.FloatList(value=value))


 def convert(self,outpath='./csv.tfrecord',filepath=None):
    '''Args:
    image_paths : List of file-paths for the images
    labels : class-labels for images(vector of size len(image_paths)X1)
    out_path : Destination of TFRecords output file
    size : expected images size
    '''
    
    data = pd.read_csv(filepath)
    
    with tf.python_io.TFRecordWriter(outpath) as writer :

       for index, rows in data.iterrows():

           features, label = np.array(rows[:-1]),rows['label']

           DataBytes = features.tostring()
           # dictionary to store image data and image label
           data={
               'features':self.wrap_bytes(DataBytes),
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


 def imgs_input_fn(self,filenames,columns,shuffle=False,repeat_count=1,batch_size=32):

    def _parse_function(serialized):
       feat = \
       {
          'features' : tf.FixedLenFeature([],tf.string),
          'label' : tf.FixedLenFeature([], tf.int64)
       }

       parsed_example = tf.parse_single_example(serialized=serialized,
                                                features=feat)

       
       data = tf.decode_raw(parsed_example['features'],tf.int64)
       features = tf.convert_to_tensor(data)
       label = tf.cast(parsed_example['label'],tf.float32)

       # decode the raw bytes so it becomes a tensor with type

  
       d={'input':data},label
       return d
    
    dataset = tf.data.TFRecordDataset(filenames=filenames)
    # Parse the serialised data to TFRecords files.
    # returns Tensorflow tensors for the image and labels.
    dataset = dataset.map(_parse_function)

    if shuffle:
        dataset = dataset.shuffle(buffer_size = 256)
    
    dataset = dataset.repeat(repeat_count) # Repeat the dataset this time
    batch_dataset = dataset.batch(batch_size)    # Batch Size
    iterator = batch_dataset.make_one_shot_iterator()   # Make an iterator
    batch_features,batch_labels = iterator.get_next()  # Tensors to get next batch of image and their labels
    
    return batch_features, batch_labels



# Referred from https://www.dlology.com/blog/how-to-leverage-tensorflows-tfrecord-to-train-keras-model/?t=152826568954#rating-13

