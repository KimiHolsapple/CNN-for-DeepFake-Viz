# SOURCES:
#  Feature Map https://towardsdatascience.com/convolutional-neural-network-feature-map-and-filter-visualization-f75012a5a49c
#  CAM https://www.youtube.com/watch?v=4v9usdvGU50
#  CNN MesoNet taken from https://www.youtube.com/watch?v=kYeLBZMTLjk and https://arxiv.org/pdf/1809.00888.pdf

'''
CNN FAKE IMAGE VIZ
Creates feature map and CAm viz for deep fake and real images.
Required Packages and installations
    pip install numpy
    pip install matplotlib
    pip install tensorflow
    pip install keras-vis
    pip install scipy.ndimage
    pip install PIL
    pip install sys
'''

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tensorflow.keras.layers import Input, Dense, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Dropout, LeakyReLU
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from matplotlib.backends.backend_pdf import PdfPages
from scipy.ndimage import zoom
import sys


LAYER_NAME = "conv2d_3"
image_dimensions = {'height':256, 'width':256, 'channels': 3}

class Classifier:
    def __init__():
        self.model = 0

    def predict(self, x):
        return self.model.predict(x)

    def fit(self, x, y):
        return self.model.train_on_batch(x,y)

    def get_accuracy(self, x, y):
        return self.model.test_on_batch(x, y)

    def get_summary(self):
        self.model.summary()

    def get_layer_names(self):
        layer_names = []
        for layers in self.model.layers:
            layer_names.append(layers._name)
        return (layer_names)

    def get_layers(self):
        layer = self.model.layers
        return layer

    def load(self, path):
        self.model.load_weights(path)

    def get_weight(self):
        weight_list = []
        for layer in self.model.layers:
            weights = layer.get_weights()
            weight_list.append(weights)
        mat = np.array(weight_list)
        return mat
    
    def get_layer_by_name(self, name):
        layer = self.model.get_layer(name)
        return layer
    
    def get_last_layer_weight(self):
        last_bias = self.model.layers[-1].get_weights()
        return last_bias

    def vis_model(self):
        layer_out = [layer.output for layer in self.model.layers[1:]]
        model_vis = Model(inputs=self.model.input, outputs=layer_out)
        return model_vis

class Meso4(Classifier):
    def __init__(self, learning_rate = 0.001):
        self.model = self.init_model()
        optimizer = Adam(lr = learning_rate)
        self.model.compile(optimizer = optimizer,
                           loss = 'mean_squared_error',
                           metrics = ['accuracy'])
    
    def init_model(self): 
        x = Input(shape = (image_dimensions['height'],
                           image_dimensions['width'],
                           image_dimensions['channels']))
        
        x1 = Conv2D(8, (3, 3), padding='same', activation = 'relu')(x)
        x1 = BatchNormalization()(x1)
        x1 = MaxPooling2D(pool_size=(2, 2), padding='same')(x1)
        
        x2 = Conv2D(8, (5, 5), padding='same', activation = 'relu')(x1)
        x2 = BatchNormalization()(x2)
        x2 = MaxPooling2D(pool_size=(2, 2), padding='same')(x2)
        
        x3 = Conv2D(16, (5, 5), padding='same', activation = 'relu')(x2)
        x3 = BatchNormalization()(x3)
        x3 = MaxPooling2D(pool_size=(2, 2), padding='same')(x3)
        
        x4 = Conv2D(16, (5, 5), padding='same', activation = 'relu')(x3)
        x4 = BatchNormalization()(x4)
        x4 = MaxPooling2D(pool_size=(4, 4), padding='same')(x4)
        
        y = Flatten()(x4)
        y = Dropout(0.5)(y)
        y = Dense(16)(y)
        y = LeakyReLU(alpha=0.1)(y)
        y = Dropout(0.5)(y)
        y = Dense(1, activation = 'sigmoid')(y)

        return Model(inputs = x, outputs = y)

# **********FUNCTIONS*************

'''
    gen_feature_map(pdf, X):
    This is a vis function that produces the feature map vis for all layers of the CNN
    It uses Viridis mapping in order to gain insight to what features are being activated.
        inputs: X - transformed image to be placed into model
                pdf - pdf file that will capture all outfput figures of feature map viz
    src @ https://towardsdatascience.com/convolutional-neural-network-feature-map-and-filter-visualization-f75012a5a49c
'''
def gen_feature_map(pdf, X):
    successive_feature_maps = meso.vis_model().predict(X)
    layer_names = meso.get_layer_names()

    for layer_name, feature_map in zip(layer_names, successive_feature_maps):
        # print(feature_map.shape)
        # print(len(feature_map.shape))
        if len(feature_map.shape) == 4:
            n_features = feature_map.shape[-1] 
            size       = feature_map.shape[ 1]  # feature map shape (1, size, size, n_features)
            display_grid = np.zeros((size, size * n_features))

            for i in range(n_features):
                x  = feature_map[0, :, :, i]
                x -= x.mean()
                x /= x.std ()
                x *=  64
                x += 128
                x  = np.clip(x, 0, 255).astype('uint8')
                # Tile each filter into a horizontal grid
                display_grid[:, i * size : (i + 1) * size] = x

                scale = 20. / n_features
                plt.figure( figsize=(scale * n_features, scale) )
                plt.title ( layer_name )
                plt.grid  ( False )
                plt.imshow( display_grid, aspect='auto', cmap='viridis' )
            pdf.savefig()

#************ MAIN ************

pdf = PdfPages('viz_ouput.pdf')
img_path = "DeepFake/127_8.jpg"
dir_path = "/Users/kimigrace/Desktop/CSE161/final_project"

meso = Meso4()
meso.load('Meso4_DF')       # load using pretrained weights

print(".............The Model has been loaded\n")
print("    ---- Menu ----    ")
print("Enter 1 to produce viz for the sample image\n")
print("      2 to run on your own image.\n")

menu_option = input()
print("User selected " + menu_option)
if (int(menu_option) == 1):
    dataGenerator = ImageDataGenerator(rescale=1.0/255)  # cale between 0 and 1 to reduce complexity
    
    dir_path = input("Please input the path to the directory where the DeepFake and Real Datasets are in: ")
    generator = dataGenerator.flow_from_directory(
        dir_path,
        target_size = (256, 256),
        batch_size=1,
        class_mode='binary')
    
    index = next(generator.index_generator)
    print(generator.class_indices)				# Checks class assignments
    # X is the image and y stands for the label that image belongs to
    X, y = generator._get_batches_of_transformed_samples(index)
    image_name = generator.filenames[index[0]]
    img_path = image_name

else:
    img_path = input("Please enter the path to the image you would like to analyze: ")
    print("\nThe file path you entered was " + img_path)
    img = load_img(img_path, target_size=(256, 256))
    x   = img_to_array(img)                           
    x   = x.reshape((1,) + x.shape)
    x /= 255.0
    X = x
    print(".............Image has successfully loaded \n")
    print("Real images are encoded with 1 and Fake images encoded with a 0")
    y_classifier = input("Please inidcate if this image is real or fake by inputting either 1 or 0: ")
    print("\nClassifier entered was " + y_classifier)
    if (int(y_classifier) == 0 or int(y_classifier) == 1 ):
        y = [int(y_classifier)]
    else:
        print("Incorrect input ")
        sys.exit()
    

print(".............Generating Model")
prediction = meso.predict(X)[0][0]

print(f"Predicted likelihood: {prediction:.4f}")
print(f"Actual label: {int(y[0])}")

print(f"\nCorrect prediction: {round(prediction)==y[0]}")

firstPage = plt.figure(figsize=(8.5,11))
firstPage.clf()
txt = 'Feature Map visulization \n' + " predicted " + str(prediction) + "\n With prediction " + str(round(prediction)==y[0])
firstPage.text(0.5,0.5,txt, transform=firstPage.transFigure, size=24, ha="center")
pdf.savefig()
plt.close()

plt.imshow(np.squeeze(X))
pdf.savefig()

meso.get_summary()
# layer_names = meso.get_layer_names()
# print(layer_names)  # we have layers at 1,4,7,10

##--------DATA VIZ ASPECT____________
'''
    Vis Aspect from @ https://www.youtube.com/watch?v=i3qjgJgQqgg
    Feature maps from @ https://towardsdatascience.com/convolutional-neural-network-feature-map-and-filter-visualization-f75012a5a49c
'''

# ***Feature Map
gen_feature_map(pdf, X)

# ***CAM
firstPage = plt.figure(figsize=(8.5,11))
firstPage.clf()
txt = 'CAM visulization'
firstPage.text(0.5,0.5,txt, transform=firstPage.transFigure, size=24, ha="center")
pdf.savefig()
plt.close()

# ____________________________________

# src https://www.youtube.com/watch?v=4v9usdvGU50

final_conv = meso.get_layer_by_name(LAYER_NAME)
W = final_conv.get_weights()[0]

img = load_img(img_path, target_size=(256, 256))
x_b   = img_to_array(img)                           
x_b   = x_b.reshape((1,) + x_b.shape)
x_b /= 255.0
X_b = x_b

fmaps = meso.predict(X_b)[0]

predictions = meso.predict(X_b)[0][0]

probs = meso.predict(X_b)
pred = np.argmax(probs[0])

w = W[:, pred]
list_fmap = [fmaps]*16
mat = np.array(list_fmap)
for i in range(len(w[0])):
    w[0][i]*=fmaps
cam = w[0]

cam = zoom(cam, (16,16), order=1)

plt.figure(figsize=(8.5,11))
plt.imshow(img,alpha=0.8)
plt.imshow(cam, cmap='jet', alpha=0.4)
pdf.savefig()

# ____________________________________

# Close PDF
pdf.close()
