from zipfile import ZipFile
import numpy as np
import tensorflow as tf
import pandas as pd
import os
import gc
import cv2
import keras
from skimage import feature
from skimage import color
from skimage import filters

from keras.models import Model
from keras.utils import Sequence
from keras import Input
from keras.preprocessing import image
import keras.applications.xception as xception
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.preprocessing import image
from keras.layers import Lambda, Concatenate
from keras.layers import Dense
from tensorflow.python.keras.initializers import Identity
from matplotlib import pyplot as plt


class data_sequence(Sequence):
    def __init__(self, x_set, batch_size, zip_path):
        self.x = x_set
        self.batch_size = batch_size
        self.zip_path = zip_path

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size : (idx + 1) * self.batch_size]
        image_hashes = batch_x.image.values
        images = np.zeros((self.batch_size, 299, 299, 3), dtype=np.float32)
        size = -1 * np.ones(shape=(self.batch_size,1), dtype=np.float32)
        avg_red = -1 * np.ones(shape=(self.batch_size,1), dtype=np.float32)
        avg_green = -1 * np.ones(shape=(self.batch_size,1), dtype=np.float32)
        avg_blue = -1 * np.ones(shape=(self.batch_size,1), dtype=np.float32)
        std_red = -1 * np.ones(shape=(self.batch_size,1), dtype=np.float32)
        std_green = -1 * np.ones(shape=(self.batch_size,1), dtype=np.float32)
        std_blue = -1 * np.ones(shape=(self.batch_size,1), dtype=np.float32)
        apw = -1 * np.ones(shape=(self.batch_size,1), dtype=np.float32)
        blurriness = -1 * np.ones(shape=(self.batch_size,1), dtype=np.float32)


        with ZipFile(self.zip_path) as im_zip:
            for i,hash in enumerate(image_hashes):
                try:
                    stats = im_zip.getinfo(hash + '.jpg')
                    size[i] = stats.file_size
                    file = im_zip.open(hash + '.jpg')

                    img = image.load_img(file, target_size=(299,299))
                    img = image.img_to_array(img)

                    average_color = [img[:, :, i].mean() for i in range(3)]
                    std_color = [img[:, :, i].std() for i in range(3)]
                    avg_red[i] = average_color[0]
                    avg_green[i] = average_color[1]
                    avg_blue[i] = average_color[2]
                    std_red[i] = std_color[0]
                    std_green[i] = std_color[1]
                    std_blue[i] = std_color[2]
                    im = color.rgb2gray(img / 255)
                    apw[i] = self.average_pixel_width(im)
                    blurriness[i] = filters.laplace(im).var()

                    images[i] = xception.preprocess_input(img)
                except KeyError:
                    print('Error loading image: %d' % hash)
        return {'inp':images, 'feat_in': np.concatenate([size, avg_red, avg_green, avg_blue,
                                                         std_red, std_green, std_blue, apw, blurriness], axis=1)}

    def average_pixel_width(self, img):
        edges_sigma1 = feature.canny(img, sigma=3)
        apw = (float(np.sum(edges_sigma1)) / (224 * 224))
        return apw * 100



train_df = pd.read_csv('../data/train.csv')
zip_path = '../data/train_jpg_0.zip'
files_in_zip = ZipFile(zip_path).namelist()
item_ids = list(map(lambda s: os.path.splitext(s)[0], files_in_zip))
train_df = train_df.loc[train_df['image'].isin(item_ids)]
train_df = train_df[:20]
dfs = np.array_split(train_df, 10)
del train_df
del files_in_zip, item_ids
gc.collect()

xception_model = xception.Xception(weights='imagenet')
inception_model = InceptionV3(weights='imagenet')
inception_resnet_model = InceptionResNetV2(weights='imagenet')

input = Input(shape=[299,299,3], name='inp')
feat_in = Input(shape=[9], dtype=np.float32, name='feat_in')

x = xception_model(input)
y = inception_model(input)
z = inception_resnet_model(input)

feat_out = Dense(9, use_bias=False, kernel_initializer=keras.initializers.Identity(), name='Identity', trainable=False)(feat_in)

model = Model([input, feat_in],
              [x, y, z, feat_out])
batch_size = 32

for i,df in enumerate(dfs):
    gc.collect()
    data_gen = data_sequence(x_set=df, batch_size=batch_size, zip_path=zip_path)
    preds = model.predict_generator(data_gen,
                            max_queue_size=8, workers=4, use_multiprocessing=True,
                            verbose=1)
    im_feat = preds[-1]
    preds = preds[:-1]
    im_features = pd.DataFrame(im_feat, columns=['size', 'avg_red', 'avg_green', 'avg_blue',
                                                 'std_red', 'std_green', 'std_blue', 'awp', 'blurriness'])
    print(im_features.head())

    #TODO Merge im_features, then concat predictions
    preds = [xception.decode_predictions(preds[0], top=1), xception.decode_predictions(preds[1], top=1),
              xception.decode_predictions(preds[2], top=1)]
    print(preds)
    preds = np.squeeze(np.stack(preds, axis=0))

    print(preds.shape)
    im_predictions_1 = pd.DataFrame(np.transpose(preds[:,:,1]), columns=['xception', 'inception', 'inception_resnet'])
    im_predictions_2 = pd.DataFrame(np.transpose(preds[:,:,2]), columns=['xception_score', 'inception_score', 'inception_resnet_score'])
    im_preds = pd.concat([im_predictions_1, im_predictions_2, im_features], axis=1)
    im_preds.to_csv('../data/image_features_%d.csv' % i)

    #im_features = pd.concat([im_features, im_predictions_1, im_predictions_2], axis=1)
    #print(im_features[:10])