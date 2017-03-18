import argparse
import json
import string
import os
import random
from collections import Counter
import h5py
import numpy as np
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from threading import Thread, Lock
from queue import Queue, Empty
import tensorflow as tf

# special tokens
from settings_keeper import SettingsKeeper

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
    vocab = sorted(vocab)

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
    return len(sent) + 2, result


def encode_images_captions(data, word_to_id, max_sentence_len):
    result = []
    for _, img_info in data.items():
        partial_sentences = []
        for cur_sentence in img_info['captions']:
            encoded_len, encoded_sent = encode_sentence(cur_sentence, word_to_id, max_sentence_len)
            # generating all partial sentences and corresponding next words that we can use for prediction
            # the last sentence is the [<Begin>]
            while encoded_len > 1:
                encoded_len -= 1
                true_answer = encoded_sent[encoded_len]
                encoded_sent[encoded_len] = 0
                partial_sentences.append((list(encoded_sent), true_answer))

        result.append((img_info['file'], partial_sentences))
    return result


def preprocess_image(filename, target_size):
    img = image.load_img(filename, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    x = np.squeeze(x)
    return x


def build_initial_embedding_matrix(path_to_embeddings=None, word_2_id=None):
    f = open(path_to_embeddings, 'r')
    embeddings = list()
    new_word_2_id = dict()
    new_id_2_word = dict()
    next_id = 1

    # zero vector for masking
    embedding_len = int(path_to_embeddings.split('.')[-2].rstrip('d'))
    embeddings.append(np.zeros(embedding_len))

    # word vectors in word_2_id
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        if word in word_2_id:
            embeddings.append(coefs)
            new_word_2_id[word] = next_id
            new_id_2_word[next_id] = word
            next_id += 1

    # word vectors for tokens
    for token in [TokenBegin, TokenEnd, TokenUnk]:
        embeddings.append(np.random.uniform(-0.15, 0.15, embedding_len))
        new_word_2_id[token] = next_id
        new_id_2_word[next_id] = token
        next_id += 1

    return np.array(embeddings), new_id_2_word, new_word_2_id


def add_images_data(h5_file, filenames, num_processed_images, images_folder, image_work_threads_count):
    filenames_slice = filenames[:num_processed_images]

    image_dset = h5_file.create_dataset('images', shape=(num_processed_images, 224, 224, 3))

    num_workers = max(1, min(image_work_threads_count, num_processed_images / 10))
    lock = Lock()
    q = Queue()

    for index, cur_filename in enumerate(filenames_slice):
        fname = os.path.join(images_folder, cur_filename)
        q.put((index, fname))

    def worker():
        while True:
            try:
                index, filename = q.get(block=False)
            except Empty:
                break

            preprocessed_image = preprocess_image(filename, (224, 224))

            with lock:
                if index % 1000 == 0:
                    print('Writing image %d / %d' % (index, num_processed_images))
                image_dset[index, :, :, :] = preprocessed_image

            q.task_done()

    for worker_index in range(num_workers):
        t = Thread(target=worker)
        t.start()

    q.join()


def preprocess(config):
    filtered_images_info = get_filtered_images_info(config.captions_file, config.max_sentence_length)

    if config.train_data is not None:
        word_to_id = json.load(open(os.path.join(config.train_data, 'word_to_id.json')))
        id_to_word = json.load(open(os.path.join(config.train_data, 'id_to_word.json')))
    else:
        vocab = build_vocab(filtered_images_info, config.min_token_instances)
        word_to_id, id_to_word = build_vocab_to_id(vocab)

    with open(os.path.join(config.output_dir, 'word_to_id.json'), 'w') as f:
        json.dump(word_to_id, f)
    with open(os.path.join(config.output_dir, 'id_to_word.json'), 'w') as f:
        json.dump(id_to_word, f)

    if config.pretrained_word_embeddings is not None:
        # preparing numpy embedding matrix, saving it to output_dir
        emb_matrix, id_to_word, word_to_id = build_initial_embedding_matrix(
            path_to_embeddings=config.pretrained_word_embeddings, word_2_id=word_to_id)
        np.save(os.path.join(config.output_dir, 'initial_word_embeddings_matrix'), emb_matrix)

        # write new word_2_id and id_2_word
        with open(os.path.join(config.output_dir, 'word_to_id.json'), 'w') as f:
            json.dump(word_to_id, f)
        with open(os.path.join(config.output_dir, 'id_to_word.json'), 'w') as f:
            json.dump(id_to_word, f)

    encoded_images_info = encode_images_captions(filtered_images_info, word_to_id, config.max_sentence_length)
    with open(os.path.join(config.output_dir, 'id_to_img.json'), 'w') as f:
        json.dump({ind: val[0] for ind, val in enumerate(encoded_images_info)}, f)

    num_images = len(encoded_images_info)
    num_processed_images = num_images
    if config.max_images > 0:
        num_processed_images = min(num_processed_images, config.max_images)

    # list of tuples (img_index, encoded partial sentence, next word)
    sentences_data = [(img_ind, sent[0], sent[1]) for img_ind, img_data in enumerate(encoded_images_info) for sent in
                      img_data[1]]

    sentence_to_img = np.asarray([x[0] for x in sentences_data], dtype=np.int32)
    sentences_array = np.array([x[1] for x in sentences_data], dtype=np.int32)
    sentences_next = np.array([x[2] for x in sentences_data], dtype=np.int32)
    with h5py.File(os.path.join(config.output_dir, 'preprocessed_text.h5'), 'w') as h5_file:
        h5_file.create_dataset('sentences_to_img', data=sentence_to_img)
        h5_file.create_dataset('sentences_next', data=sentences_next)
        h5_file.create_dataset('sentences', data=sentences_array)

    with h5py.File(os.path.join(config.output_dir, 'preprocessed_images.h5'), 'w') as h5_file:
        add_images_data(h5_file, [x[0] for x in encoded_images_info], num_processed_images, config.images_folder,
                        config.image_work_threads)


def main_func():
    parser = argparse.ArgumentParser()

    parser.add_argument('--captions_file',
                        default='data/annotations/captions_train2014.json')
    parser.add_argument('--output_dir',
                        default='output_train')
    parser.add_argument('--images_folder',
                        default="data/train2014")
    parser.add_argument('--train_data',
                        default=None)
    parser.add_argument('--model',
                        default='default_model')

    args = parser.parse_args()

    settings_ini_section_list = ['preprocess', args.model]
    settings = SettingsKeeper()
    settings.add_ini_file('settings.ini', settings_ini_section_list)
    if os.path.isfile('user_settings.ini'):
        settings.add_ini_file('user_settings.ini', settings_ini_section_list, False)
    settings.add_parsed_arguments(args)

    random.seed(settings.seed)
    np.random.seed(settings.seed)
    tf.set_random_seed(settings.seed)

    preprocess(settings)


if __name__ == '__main__':
    main_func()
