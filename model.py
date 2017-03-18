import argparse
import json
import random
import os
import datetime
import h5py
import keras
import numpy as np
import tensorflow as tf

from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.applications.resnet50 import ResNet50
from keras.applications.inception_v3 import InceptionV3
from keras.applications.resnet50 import ResNet50

from keras.engine import Input
from keras.layers import GlobalMaxPooling2D, GRU, Dense, Activation, Embedding, TimeDistributed, RepeatVector
from keras.models import Sequential, Merge, Model

from model_checkpoints import MyModelCheckpoint
from settings_keeper import SettingsKeeper

adam = keras.optimizers.Adam(lr=0.0002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
nadam = keras.optimizers.Nadam(lr=0.0005, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)


def create_image_model(images_shape, repeat_count):
    inputs = Input(shape=images_shape)

    #    visual_model = VGG16(weights='imagenet', include_top = False, input_tensor = inputs)
    visual_model = ResNet50(weights='imagenet', include_top=False, input_tensor=inputs)

    x = visual_model(inputs)
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


def create_model(images_shape, dict_size, sentence_len, optimizer = nadam):
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

    combined_model.compile(loss='sparse_categorical_crossentropy', optimizer=nadam)

    return combined_model


def prepare_batch(sentences_dset, sentences_next_dset, sent_to_img_dset, images_dset, batch_size):
    num_sentences = sentences_dset.shape[0]
    assert (num_sentences == sentences_next_dset.shape[0])

    while 1:
        indices = np.random.randint(num_sentences, size=batch_size)
        sentences_data = np.array([sentences_dset[ind] for ind in indices])
        images_data = np.array([images_dset[sent_to_img_dset[ind]] for ind in indices])

        # input words are 1-indexed and 0 index is used for masking!
        # but result words are 0-indexed and will go into [0, ..., dict_size-1] !!!
        truth_data = np.array([sentences_next_dset[ind] - 1 for ind in indices])

        yield [images_data, sentences_data], truth_data


def train_model(h5_images_train=None, h5_text_train=None, dict_size_train=None,
                weight_save_period=None, samples_per_epoch=None, num_epoch=None, batch_size=None,
                h5_images_val=None, h5_text_val=None, val_samples=None, start_weights_path=None, model_id=None):

    # Train
    images_train = h5_images_train['images']
    sent_to_img_train = h5_text_train['sentences_to_img']
    sentences_train = h5_text_train['sentences']
    sentences_next_train = h5_text_train['sentences_next']

    # Val
    if h5_images_val and h5_text_val and val_samples:
        images_val = h5_images_val['images']
        sent_to_img_val = h5_text_val['sentences_to_img']
        sentences_val = h5_text_val['sentences']
        sentences_next_val = h5_text_val['sentences_next']

        # initialize val generator
        val_stream = prepare_batch(sentences_val, sentences_next_val, sent_to_img_val, images_val, batch_size)
    else:
        val_stream = None
        val_samples = None

    sentence_len = len(sentences_train[0])
    image_shape = images_train.shape[1:]

    model = create_model(image_shape, dict_size_train, sentence_len)
    if start_weights_path is not None:
        model.load_weights(start_weights_path)
        print('Using start weights: "{}"'.format(start_weights_path))

    tb = keras.callbacks.TensorBoard(log_dir="model_output", histogram_freq=1, write_images=True, write_graph=True)
    cp = MyModelCheckpoint("model_output", "weights", weight_save_period, model_id=model_id)

    # Initialize train generator
    train_stream = prepare_batch(sentences_train, sentences_next_train, sent_to_img_train, images_train, batch_size)

    model.fit_generator(generator=train_stream,
                        samples_per_epoch=samples_per_epoch,
                        validation_data=val_stream,
                        nb_val_samples=val_samples,
                        nb_epoch=num_epoch,
                        callbacks=[tb, cp])


def main_func():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id',
                        default=datetime.datetime.now().isoformat(), type=str)

    args = parser.parse_args()

    settings_ini_section_list = ['model']
    settings = SettingsKeeper()
    settings.add_ini_file('settings.ini', settings_ini_section_list)
    if os.path.isfile('user_settings.ini'):
        settings.add_ini_file('user_settings.ini', settings_ini_section_list, False)
    settings.add_parsed_arguments(args)

    if settings.cuda_devices is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = settings.cuda_devices

    random.seed(settings.seed)
    np.random.seed(settings.seed)
    tf.set_random_seed(settings.seed)

    # Train data
    id_to_word_train = os.path.join(settings.preprocessed_train, 'id_to_word.json')
    with open(id_to_word_train, 'r') as f:
        dict_size_train = len(json.load(f))

    preprocessed_images_train = os.path.join(settings.preprocessed_train, 'preprocessed_images.h5')
    preprocessed_text_train = os.path.join(settings.preprocessed_train, 'preprocessed_text.h5')

    # Val data
    if settings.preprocessed_val is not None:
        preprocessed_images_val = os.path.join(settings.preprocessed_val, 'preprocessed_images.h5')
        preprocessed_text_val = os.path.join(settings.preprocessed_val, 'preprocessed_text.h5')

        h5_images_val = h5py.File(preprocessed_images_val, 'r')
        h5_text_val = h5py.File(preprocessed_text_val, 'r')
    else:
        h5_images_val, h5_text_val = None, None

    with h5py.File(preprocessed_images_train, 'r') as h5_images_train, \
            h5py.File(preprocessed_text_train, 'r') as h5_text_train:

        train_model(h5_images_train=h5_images_train, h5_text_train=h5_text_train, dict_size_train=dict_size_train,
                    # train data
                    h5_images_val=h5_images_val, h5_text_val=h5_text_val, val_samples=settings.samples_val,  # val data
                    weight_save_period=settings.weight_save_epoch_period,
                    samples_per_epoch=settings.samples_per_epoch,
                    num_epoch=settings.num_epoch,
                    batch_size=settings.batch_size,
                    start_weights_path=settings.start_weights_path,
                    model_id=settings.model_id)

    if h5_text_val and h5_images_val:
        h5_text_val.close()
        h5_images_val.close()


if __name__ == '__main__':
    main_func()
