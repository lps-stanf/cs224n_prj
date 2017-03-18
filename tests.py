import argparse
import json
import os

import h5py

from model import create_model
from preprocess import preprocess_image
import numpy as np
from keras.preprocessing import image

from settings_keeper import SettingsKeeper


def create_image_caption(model, image_filename, resolution, sentence_max_len, TokenBeginIndex, TokenEndIndex,
                         id_to_word_dict):
    preprocessed_image = preprocess_image(image_filename, resolution)
    # adding 1 to the beginning of the image shape so that model can accept it (making batch with one element)
    preprocessed_image = np.expand_dims(preprocessed_image, axis=0)

    current_sentence = np.zeros((1, sentence_max_len), dtype=np.int32)
    current_sentence[0, 0] = TokenBeginIndex
    cur_sent_len = 1

    while cur_sent_len < sentence_max_len:
        next_word = model.predict([preprocessed_image, current_sentence])
        # selecting the word with max probability
        next_word = np.argmax(next_word)
        # making the word 1-indexed as in dictionary
        next_word += 1
        current_sentence[0, cur_sent_len] = next_word
        cur_sent_len += 1
        if next_word == TokenEndIndex:
            break

    result_sentence = ""
    for cur_code in current_sentence[0]:
        result_sentence += id_to_word_dict[str(cur_code)]
        result_sentence += ' '
        if cur_code == TokenEndIndex:
            break
    result_sentence = result_sentence.strip()

    print('"{0}": "{1}"'.format(image_filename, result_sentence))


def find_token_index(id_to_word_dict, token):
    for k, v in id_to_word_dict.items():
        if v == token:
            return k
    else:
        raise "Didn't find required token"


def create_caption_for_path(source_path, model, model_resolution, sentence_max_len, TokenBeginIndex, TokenEndIndex,
                            id_to_word_dict):
    if os.path.isfile(source_path):
        create_image_caption(model, source_path, model_resolution, sentence_max_len, TokenBeginIndex, TokenEndIndex,
                             id_to_word_dict)
    else:
        images_list = image.list_pictures(source_path)
        for cur_image_path in images_list:
            create_image_caption(model, cur_image_path, model_resolution, sentence_max_len, TokenBeginIndex,
                                 TokenEndIndex, id_to_word_dict)


def perform_testing(settings, id_to_word_dict):
    from preprocess import TokenBegin, TokenEnd

    with h5py.File(settings.preprocessed_images_file, 'r') as h5_images_file:
        image_shape = h5_images_file['images'].shape[1:]

    with h5py.File(settings.preprocessed_text_file, 'r') as h5_text_file:
        sentence_max_len = len(h5_text_file['sentences'][0])

    dict_size = len(id_to_word_dict)

    TokenBeginIndex = find_token_index(id_to_word_dict, TokenBegin)
    TokenEndIndex = find_token_index(id_to_word_dict, TokenEnd)

    model = create_model(image_shape, dict_size, sentence_max_len, settings)
    model.load_weights(settings.weights_filename)

    create_caption_for_path(settings.test_source, model, image_shape[:2], sentence_max_len, TokenBeginIndex, TokenEndIndex,
                         id_to_word_dict)


def main_func():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights_filename', required=True)
    parser.add_argument('--cuda_devices',
                        default=None)
    parser.add_argument('--model',
                        default='default_model')

    args = parser.parse_args()

    settings_ini_section_list = ['tests', args.model]
    settings = SettingsKeeper()
    settings.add_ini_file('settings.ini', settings_ini_section_list)
    if os.path.isfile('user_settings.ini'):
        settings.add_ini_file('user_settings.ini', settings_ini_section_list, False)
    settings.add_parsed_arguments(args)

    if settings.cuda_devices is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = settings.cuda_devices

    with open(settings.id_to_word_file, 'r') as f:
        id_to_word_dict = json.load(f)
        id_to_word_dict = {int(k): v for k, v in id_to_word_dict.items()}
        perform_testing(settings, id_to_word_dict)


if __name__ == '__main__':
    main_func()
