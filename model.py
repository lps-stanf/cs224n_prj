import argparse
import json
import random
import string
from collections import Counter
import numpy as np
import tensorflow as tf
from keras.layers import Conv2D, Dense, Embedding, TimeDistributed, GRU, RepeatVector, Activation, GlobalMaxPooling2D, \
    MaxPooling2D
from keras.models import Sequential, Merge
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

    result = [word_to_id[TokenBegin]] + result + [word_to_id[TokenEnd]] + [0] * delta
    return result


def encode_images_info(data, word_to_id, max_sentence_len):
    result = []
    for _, img_info in data.items():
        encoded_sentences = [encode_sentence(sent, word_to_id, max_sentence_len) for sent in
                                    img_info['captions']]
        result.append((img_info['file'], encoded_sentences))
    return result


def preprocess(args):
    filtered_images_info = get_filtered_images_info(args.captionsFile, args.max_sentence_length)
    vocab = build_vocab(filtered_images_info, args.min_token_instances)

    word_to_id, id_to_word = build_vocab_to_id(vocab)
    with open(args.output_dir + '/word_to_id.json', 'w') as f:
        json.dump(word_to_id, f)
    with open(args.output_dir + '/id_to_word.json', 'w') as f:
        json.dump(id_to_word, f)

    encoded_images_info = encode_images_info(filtered_images_info, word_to_id, args.max_sentence_length)

    return encoded_images_info, word_to_id


def create_model(image_model_weights_path, dict_size, encoded_sent_len):
    image_model = Sequential()
    # image_model.add(Input(shape=(224, 224, 3)))

    # Block 1
    image_model.add(
        Conv2D(64, 3, 3, activation='relu', border_mode='same', name='block1_conv1', input_shape=(224, 224, 3)))
    image_model.add(Conv2D(64, 3, 3, activation='relu', border_mode='same', name='block1_conv2'))
    image_model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool'))

    # Block 2
    image_model.add(Conv2D(128, 3, 3, activation='relu', border_mode='same', name='block2_conv1'))
    image_model.add(Conv2D(128, 3, 3, activation='relu', border_mode='same', name='block2_conv2'))
    image_model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool'))

    # Block 3
    image_model.add(Conv2D(256, 3, 3, activation='relu', border_mode='same', name='block3_conv1'))
    image_model.add(Conv2D(256, 3, 3, activation='relu', border_mode='same', name='block3_conv2'))
    image_model.add(Conv2D(256, 3, 3, activation='relu', border_mode='same', name='block3_conv3'))
    image_model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool'))

    # Block 4
    image_model.add(Conv2D(512, 3, 3, activation='relu', border_mode='same', name='block4_conv1'))
    image_model.add(Conv2D(512, 3, 3, activation='relu', border_mode='same', name='block4_conv2'))
    image_model.add(Conv2D(512, 3, 3, activation='relu', border_mode='same', name='block4_conv3'))
    image_model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool'))

    # Block 5
    image_model.add(Conv2D(512, 3, 3, activation='relu', border_mode='same', name='block5_conv1'))
    image_model.add(Conv2D(512, 3, 3, activation='relu', border_mode='same', name='block5_conv2'))
    image_model.add(Conv2D(512, 3, 3, activation='relu', border_mode='same', name='block5_conv3'))
    image_model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool'))

    image_model.add(GlobalMaxPooling2D())
    # outputs (None, 512)

    image_model.load_weights(image_model_weights_path)

    # next, let's define a RNN model that encodes sequences of words
    # into sequences of 128-dimensional word vectors.
    language_model = Sequential()
    language_model.add(Embedding(dict_size + 1, 512, input_length=encoded_sent_len, mask_zero=True))
    language_model.add(GRU(output_dim=128, return_sequences=True))
    language_model.add(TimeDistributed(Dense(128)))

    # let's repeat the image vector to turn it into a sequence.
    image_model.add(RepeatVector(encoded_sent_len))

    # the output of both models will be tensors of shape (samples, max_caption_len, 128).
    # let's concatenate these 2 vector sequences.
    model = Sequential()
    model.add(Merge([image_model, language_model], mode='concat', concat_axis=-1))

    # let's encode this vector sequence into a single vector
    model.add(GRU(256, return_sequences=False))

    # which will be used to compute a probability
    # distribution over what the next word in the caption should be!
    model.add(Dense(dict_size))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    return model

def prepare_batch(images_info, image_batch_size = 10):
    while 1:
        indices = np.random.randint(0, len(images_info), image_batch_size)
        sample = [images_info[ind] for ind in indices]

        img_data = []
        for img_path,_ in sample:
            img = image.load_img(img_path, target_size=(224, 224))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            img_data.append(x)

        # todo!
        captions_data = []

        yield img_data, captions_data



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
    parser.add_argument('--seed', type=int,
                        default=42)
    parser.add_argument('--image_model_weights',
                        default='data/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5')

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.set_random_seed(args.seed)

    images_info, word_to_id = preprocess(args)

    prepare_batch(images_info)

    model = create_model(args.image_model_weights, len(word_to_id), args.max_sentence_length + 2)

    model.fit_generator(generator=prepare_batch(images_info), samples_per_epoch=10000, nb_epoch=10)
