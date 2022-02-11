# import the necessary packages
from keras.preprocessing.image import img_to_array

class ImageToArrayPreprocessor:
     def __init__(self, dataFormat=None):
          # store the image data format
          self.dataFormat = dataFormat

     def preprocess(self, image):
          # apply the Keras utility function that correctly rearranges
          # the dimensions of the image
          image = img_to_array(image, data_format=self.dataFormat) 
          # now normalize the image data to the range [0, 1]
          return image.astype("float") / 255.0