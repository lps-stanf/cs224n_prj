import argparse
import json
import os

import h5py

from model import create_model
from preprocess import preprocess_image
import numpy as np


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
        result_sentence += id_to_word_dict[cur_code]
        result_sentence += ' '

    print('{0}: "{1}"', image_filename, result_sentence)


def find_token_index(id_to_word_dict, token):
    for k, v in id_to_word_dict.items():
        if v == token:
            return k
    else:
        raise "Didn't find required token"


def perform_testing(preprocessed_images_file, preprocessed_text_file, weights_filename, test_source, id_to_word_dict):
    from preprocess import TokenBegin, TokenEnd

    with h5py.File(args.preprocessed_images_file, 'r') as h5_images_file:
        image_shape = h5_images_file['images'].shape[1:]

    with h5py.File(args.preprocessed_text_file, 'r') as h5_text_file:
        sentence_max_len = len(h5_text_file['sentences'][0])

    dict_size = len(id_to_word_dict)

    TokenBeginIndex = find_token_index(id_to_word_dict, TokenBegin)
    TokenEndIndex = find_token_index(id_to_word_dict, TokenEnd)

    model = create_model(image_shape, dict_size, sentence_max_len)
    model.load_weights(weights_filename)

    create_image_caption(model, 'test.jpg', image_shape[:2], sentence_max_len, TokenBeginIndex, TokenEndIndex,
                         id_to_word_dict)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--preprocessed_text_file',
                        default='output/preprocessed_text.h5')
    parser.add_argument('--preprocessed_images_file',
                        default='output/preprocessed_images.h5')
    parser.add_argument('--weights_filename', required=True)
    parser.add_argument('--test_source',
                        default='test_images')
    parser.add_argument('--id_to_word_file',
                        default='output/id_to_word.json')
    parser.add_argument('--cuda_devices',
                        default=None)

    args = parser.parse_args()

    if args.cuda_devices is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_devices

    with open(args.id_to_word_file, 'r') as f:
        id_to_word_dict = json.load(f)
        id_to_word_dict = {int(k): v for k, v in id_to_word_dict.items()}
        perform_testing(args.preprocessed_images_file, args.preprocessed_text_file, args.weights_filename,
                        args.test_source, id_to_word_dict)
