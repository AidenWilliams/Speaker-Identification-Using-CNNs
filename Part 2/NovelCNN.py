# import the necessary packages
from keras.models import Sequential                         # these 2 are added for regularization
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, BatchNormalization, Dropout
from keras import backend as K


class NovelCNN:
    @staticmethod
    def build(width, height, depth, classes):
        # initialize the model along with the input shape to be
        # "channels last"
        model = Sequential()
        inputShape = (height, width, depth)

        # if we are using "channels first", update the input shape
        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)

        model.add(Conv2D(input_shape=inputShape,filters=64,kernel_size=(3,3),padding="same", activation="relu"))
        model.add(MaxPool2D(pool_size=(2,2),strides=(2,2))) # 'raw' = 73%
        model.add(BatchNormalization())

        model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
        model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
        model.add(BatchNormalization())

        model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
        model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
        model.add(BatchNormalization())

        model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
        model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
        model.add(BatchNormalization()) # batch norms = 83%

        model.add(Flatten())
        model.add(Dense(units=4096,activation="relu")) 
        model.add(Dropout(0.3)) # 0.3 in both dropouts = 90%
        model.add(Dense(units=4096,activation="relu"))
        model.add(Dropout(0.3))
        model.add(Dense(units=classes, activation="softmax"))
        
        # return the constructed network architecture
        model.summary()
        return model