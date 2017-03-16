import argparse
import json
import random
import os

import h5py
import keras
import numpy as np
import tensorflow as tf
from keras.applications.vgg16 import VGG16
from keras.engine import Input
from keras.layers import GlobalMaxPooling2D, GRU, Dense, Activation, Embedding, TimeDistributed, RepeatVector
from keras.models import Sequential, Merge, Model

from model_checkpoints import MyModelCheckpoint


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

    # input words are 1-indexed and 0 index is used for masking!
    # but result words are 0-indexed and will go into [0, ..., dict_size-1] !!!

    combined_model.compile(loss='sparse_categorical_crossentropy', optimizer='rmsprop')

    return combined_model


def prepare_batch(sentence_len, sentences_dset, sentences_len_dset, sent_to_img_dset, images_dset, batch_size=50):
    num_sentences = sentences_dset.shape[0]
    while 1:
        indices = np.random.randint(num_sentences, size=batch_size)

        # todo: move generation of partial sentences to preprocessing?
        # select partial sentence end point for each sentence
        # sentence has 2 additional tokens: one at the beginning and one at the end
        # so we can select partial sentence length be from 1 up to (1 + original sentence len)
        # (in the last case we predict the <END> label)
        sentences_len_data = [sentences_len_dset[ind] for ind in indices]
        partial_lengths = [1 + np.random.randint(0, high=sent_len + 1) for sent_len in sentences_len_data]

        sentences_data = []
        truth_data = []
        for index_num, ind in enumerate(indices):
            new_elem = np.zeros(sentence_len)
            cur_partial_len = partial_lengths[index_num]
            new_elem[:cur_partial_len] = sentences_dset[ind][:cur_partial_len]
            sentences_data.append(new_elem)

            # input words are 1-indexed and 0 index is used for masking!
            # but result words are 0-indexed and will go into [0, ..., dict_size-1] !!!
            truth_data.append(sentences_dset[ind][cur_partial_len] - 1)

        images_data = np.array([images_dset[sent_to_img_dset[ind]] for ind in indices])

        yield [images_data, np.array(sentences_data)], np.array(truth_data)


def train_model(h5_data_file, dict_size, weight_save_period, samples_per_epoch, num_epoch, batch_size):
    images_dset = h5_data_file['images']
    sent_to_img_dset = h5_data_file['sentences_to_img']
    sentences_dset = h5_data_file['sentences']
    sentences_len_dset = h5_data_file['sentences_len']

    sentence_len = len(sentences_dset[0])
    image_shape = images_dset.shape[1:]

    model = create_model(image_shape, dict_size, sentence_len)

    tb = keras.callbacks.TensorBoard(log_dir="model_output", histogram_freq=1, write_images=True, write_graph=True)
    cp = MyModelCheckpoint("model_output", "weights", weight_save_period)

    model.fit_generator(generator=prepare_batch(sentence_len, sentences_dset, sentences_len_dset, sent_to_img_dset,
                                                images_dset, batch_size),
                        samples_per_epoch=samples_per_epoch, nb_epoch=num_epoch, callbacks=[tb, cp])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed',
                        default=42, type=int)
    parser.add_argument('--cuda_devices',
                        default=None)
    parser.add_argument('--id_to_word_file',
                        default='output/id_to_word.json')
    parser.add_argument('--preprocessed_file',
                        default='output/preprocessed.h5')
    parser.add_argument('--weight_save_epoch_period',
                        default=1, type=int)
    parser.add_argument('--batch_size',
                        default=50, type=int)
    parser.add_argument('--samples_per_epoch',
                        default=1000, type=int)
    parser.add_argument('--num_epoch',
                        default=100, type=int)

    args = parser.parse_args()

    if args.cuda_devices is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_devices

    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.set_random_seed(args.seed)

    with open(args.id_to_word_file, 'r') as f:
        dict_size = len(json.load(f))

    with h5py.File('output/preprocessed.h5', 'r') as h5_data_file:
        train_model(h5_data_file, dict_size, args.weight_save_epoch_period, args.samples_per_epoch, args.num_epoch,
                    args.batch_size)
