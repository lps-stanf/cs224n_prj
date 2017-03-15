import argparse
import json
import random
import h5py
import numpy as np
import tensorflow as tf
from keras.applications.vgg16 import VGG16
from keras.engine import Input
from keras.layers import GlobalMaxPooling2D, GRU, Dense, Activation, Embedding, TimeDistributed, RepeatVector
from keras.models import Sequential, Merge, Model


def create_image_model(images_shape, repeat_count):
    inputs = Input(shape=images_shape)
    vgg_model = VGG16(weights='imagenet', include_top=False, input_tensor=inputs)

    x = vgg_model(inputs)
    x = GlobalMaxPooling2D()(x)
    x = RepeatVector(repeat_count)(x)
    return Model(inputs, x, 'image_model')


def create_sentence_model(dict_size, sentence_len):
    sentence_model = Sequential()
    # + 1 to respect masking
    sentence_model.add(Embedding(dict_size + 1, 512, input_length=sentence_len, mask_zero=True))
    sentence_model.add(GRU(output_dim=128, return_sequences=True))
    sentence_model.add(TimeDistributed(Dense(128)))
    return sentence_model


def create_model(images_shape, dict_size, sentence_len):
    # input (None, 224, 224, 3), outputs (None, sentence_len, 512)
    image_model = create_image_model(images_shape, sentence_len)

    # outputs (None, sentence_len, 128)
    sentence_model = create_sentence_model(dict_size, sentence_len)

    combined_model = Sequential()
    combined_model.add(Merge([image_model, sentence_model], mode='concat', concat_axis=-1))

    combined_model.add(GRU(256, return_sequences=False))

    combined_model.add(Dense(dict_size))
    combined_model.add(Activation('softmax'))

    combined_model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    return combined_model


def train_model(h5_data_file, dict_size):
    images_dset = h5_data_file['images']
    sent_to_img_dset = h5_data_file['sentences_to_img']
    sentences_dset = h5_data_file['sentences']

    sentence_len = len(sentences_dset[0])
    image_shape =images_dset.shape[1:]

    model = create_model(image_shape, dict_size, sentence_len)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed',
                        default=42, type=int)
    parser.add_argument('--id_to_word_file',
                        default='output/id_to_word.json')
    parser.add_argument('--preprocessed_file',
                        default='output/preprocessed.h5')

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.set_random_seed(args.seed)

    with open(args.id_to_word_file, 'r') as f:
        dict_size = len(json.load(f))

    with h5py.File('output/preprocessed.h5', 'r') as h5_data_file:
        train_model(h5_data_file, dict_size)