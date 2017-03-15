import argparse
import json
import string
from collections import Counter
import h5py
import numpy as np
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input

# special tokens
TokenUnk = '<UNK>'
TokenBegin = '<BEGIN>'
TokenEnd = '<END>'


def words_preprocess(phrase):
    replacements = {
        u'½': u'half',
        u'—': u'-',
        u'™': u'',
        u'¢': u'cent',
        u'ç': u'c',
        u'û': u'u',
        u'é': u'e',
        u'°': u' degree',
        u'è': u'e',
        u'…': u'',
    }
    for k, v in replacements.items():
        phrase = phrase.replace(k, v)
    return str(phrase).lower().translate(str.maketrans('', '', string.punctuation)).split()


def get_filtered_images_info(captions_file, max_sentence_length):
    with open(captions_file) as f:
        captions_data = json.load(f)

    image_info_dict = {}

    # collect captions for the same id
    for cur_annotation in captions_data['annotations']:
        im_id = cur_annotation['image_id']
        split_cur_caption = words_preprocess(cur_annotation['caption'])

        # skipping too long or empty captions
        if 0 < max_sentence_length < len(split_cur_caption) or len(split_cur_caption) <= 0:
            continue

        if im_id not in image_info_dict:
            image_info_dict[im_id] = {'captions': []}
        image_info_dict[im_id]['captions'].append(split_cur_caption)

    # collect filenames of images with collected captions
    for cur_image_data in captions_data['images']:
        im_id = cur_image_data['id']
        if im_id in image_info_dict:
            image_info_dict[im_id]['file'] = cur_image_data['file_name']

    # filter out images with
    result = {k: v for k, v in image_info_dict.items() if 'file' in v}
    return result


def build_vocab(data, min_token_instances):
    token_counter = Counter()
    for _, img in data.items():
        for caption in img['captions']:
            token_counter.update(caption)
    vocab = set()
    for token, count in token_counter.items():
        if count >= min_token_instances:
            vocab.add(token)

    vocab.add(TokenUnk)
    vocab.add(TokenBegin)
    vocab.add(TokenEnd)

    return vocab


def build_vocab_to_id(vocab):
    word_to_id, id_to_word = {}, {}
    next_id = 1

    for word in vocab:
        word_to_id[word] = next_id
        id_to_word[next_id] = word
        next_id += 1

    return word_to_id, id_to_word


def encode_sentence(sent, word_to_id, max_sentence_len):
    result = []
    for word in sent:
        if word in word_to_id:
            result.append(word_to_id[word])
        else:
            result.append(word_to_id[TokenUnk])

    delta = max_sentence_len - len(result)
    assert delta >= 0

    # adding zeroes at the end for masking in the model
    result = [word_to_id[TokenBegin]] + result + [word_to_id[TokenEnd]] + [0] * delta
    return len(sent), result


def encode_images_captions(data, word_to_id, max_sentence_len):
    result = []
    for _, img_info in data.items():
        encoded_sentences = [encode_sentence(sent, word_to_id, max_sentence_len) for sent in
                             img_info['captions']]
        result.append((img_info['file'], encoded_sentences))
    return result


def add_images_data(h5_file, filenames, num_processed_images, images_folder):
    filenames_slice = filenames[:num_processed_images]

    image_dset = h5_file.create_dataset('images', shape=(num_processed_images, 224, 224, 3))
    for index, cur_filename in enumerate(filenames_slice):
        full_filename = images_folder + '/' + cur_filename
        img = image.load_img(full_filename, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        x = np.squeeze(x)
        image_dset[index, :, :, :] = x


def preprocess(args):
    filtered_images_info = get_filtered_images_info(args.captionsFile, args.max_sentence_length)
    vocab = build_vocab(filtered_images_info, args.min_token_instances)

    word_to_id, id_to_word = build_vocab_to_id(vocab)
    with open(args.output_dir + '/word_to_id.json', 'w') as f:
        json.dump(word_to_id, f)
    with open(args.output_dir + '/id_to_word.json', 'w') as f:
        json.dump(id_to_word, f)

    encoded_images_info = encode_images_captions(filtered_images_info, word_to_id, args.max_sentence_length)
    with open(args.output_dir + '/id_to_img.json', 'w') as f:
        json.dump({ind: val[0] for ind, val in enumerate(encoded_images_info)}, f)

    num_images = len(encoded_images_info)
    num_processed_images = num_images
    if args.max_images >= 0:
        num_processed_images = min(num_processed_images, args.max_images)

    # list of tuples (img_index, original_sentence_len, encoded sentence)
    sentences_data = [(img_ind, sent[0], sent[1]) for img_ind, img_data in enumerate(encoded_images_info) for sent in img_data[1]]

    sentence_to_img = np.asarray([x[0] for x in sentences_data], dtype=np.int32)
    sentences_len = np.array([x[1] for x in sentences_data], dtype=np.int32)
    sentences_array = np.array([x[2] for x in sentences_data], dtype=np.int32)
    with h5py.File(args.output_dir + '/preprocessed.h5', 'w') as h5_file:
        h5_file.create_dataset('sentences_to_img', data=sentence_to_img)
        h5_file.create_dataset('sentences_len', data=sentences_len)
        h5_file.create_dataset('sentences', data=sentences_array)

        add_images_data(h5_file, [x[0] for x in encoded_images_info], num_processed_images, args.images_folder)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--captionsFile',
                        default='data/captions_train2014.json')
    parser.add_argument('--max_sentence_length',
                        default=16, type=int)
    parser.add_argument('--min_token_instances',
                        default=15, type=int)
    parser.add_argument('--output_dir',
                        default='output')
    parser.add_argument('--max_images',
                        default=10, type=int)
    parser.add_argument('--images_folder',
                        default="data/train2014")

    args = parser.parse_args()
    preprocess(args)
