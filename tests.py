import argparse
import json
import os
import random

import h5py
import numpy as np
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
import tensorflow as tf

from model import create_model
from preprocess import preprocess_image
from settings_keeper import SettingsKeeper


def get_text_box(text, text_border, font):
    t_w, t_h = font.getsize(text)
    return t_w + 2 * text_border, t_h + 2 * text_border


def fit_text_to_box(box_w, text, text_border, font):
    result_lines = []
    words = text.strip().split()
    while len(words) > 0:
        cur_line = [words.pop(0)]
        cur_line_merged = cur_line[0]
        cur_line_box = get_text_box(cur_line_merged, text_border, font)
        while len(words) > 0:
            test_line = list(cur_line)
            test_line.append(words[0])
            test_line_merged = ' '.join(test_line)
            test_line_box = get_text_box(test_line_merged, text_border, font)
            if test_line_box[0] <= box_w:
                cur_line = test_line
                cur_line_merged = test_line_merged
                cur_line_box = test_line_box
                words.pop(0)
            else:
                break

        result_lines.append((cur_line_merged, cur_line_box))
    return result_lines


def add_label_to_image(src_file, target_file, text, max_out_resolution):
    margin = 10
    font_size = 20
    text_line_border = 2

    with Image.open(src_file) as img:
        font = ImageFont.truetype("fonts/LibreBaskerville-Bold.ttf", font_size)

        if max_out_resolution is not None:
            img.thumbnail(max_out_resolution, Image.ANTIALIAS)

        fit_text = fit_text_to_box(img.width - 2 * margin, text, text_line_border, font)

        total_text_h = 0
        for text_line, text_line_box in fit_text:
            total_text_h += text_line_box[1]

        bottom_text_h = total_text_h + 2 * margin

        with Image.new(img.mode, (img.width, img.height + bottom_text_h), 'white') as new_image:
            new_image.paste(img, box=(0, 0))

            draw = ImageDraw.Draw(new_image)

            x = margin
            y = new_image.height - total_text_h - margin
            shadow_color = 'black'

            for text_line, text_line_box in fit_text:
                # thicker border
                # draw.text((x - 1, y - 1), text_line, font=font, fill=shadow_color)
                # draw.text((x + 1, y - 1), text_line, font=font, fill=shadow_color)
                # draw.text((x - 1, y + 1), text_line, font=font, fill=shadow_color)
                # draw.text((x + 1, y + 1), text_line, font=font, fill=shadow_color)

                draw.text((x, y), text_line, font=font, fill='black')
                y += text_line_box[1]

            new_image.save(target_file)


def create_image_caption(model, image_filename, resolution, sentence_max_len, TokenBeginIndex, TokenEndIndex,
                         id_to_word_dict, output_folder=None, max_out_resolution=None, printResult=True):
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
    result_words_list = []
    has_end_token = False
    for cur_code in current_sentence[0]:
        cur_word = id_to_word_dict[cur_code]
        result_sentence += cur_word
        result_sentence += ' '
        result_words_list.append(cur_word)
        if cur_code == TokenEndIndex:
            has_end_token = True
            break
    result_sentence = result_sentence.strip()

    if printResult:
        print('"{0}": "{1}"'.format(image_filename, result_sentence))

    if len(result_words_list) > 0:
        result_words_list.pop(0)
        if has_end_token:
            result_words_list.pop()

    if output_folder is not None:
        target_filename = os.path.basename(image_filename)
        target_filename = os.path.join(output_folder, target_filename)

        if len(result_words_list) > 0:
            result_words_list[-1] += '.'
            result_words_list[0] = result_words_list[0].title()
        add_label_to_image(image_filename, target_filename, ' '.join(result_words_list), max_out_resolution)

    return result_words_list


def find_token_index(id_to_word_dict, token):
    for k, v in id_to_word_dict.items():
        if v == token:
            return k
    else:
        raise Exception("Didn't find required token")


def get_image_files(source_dir):
    ext_list = ['.jpg', '.jpeg', '.bmp', '.png']
    files = os.listdir(source_dir)
    filtered_files = [os.path.join(source_dir, file) for file in files if os.path.splitext(file)[1] in ext_list]
    return filtered_files


def create_caption_for_path(source_path, model, model_resolution, sentence_max_len, TokenBeginIndex, TokenEndIndex,
                            id_to_word_dict, output_folder, max_out_resolution):
    if os.path.isfile(source_path):
        create_image_caption(model, source_path, model_resolution, sentence_max_len, TokenBeginIndex, TokenEndIndex,
                             id_to_word_dict, output_folder, max_out_resolution)
    else:
        images_list = get_image_files(source_path)
        for cur_image_path in images_list:
            create_image_caption(model, cur_image_path, model_resolution, sentence_max_len, TokenBeginIndex,
                                 TokenEndIndex, id_to_word_dict, output_folder, max_out_resolution)


def calculate_metrics(settings, model, id_to_word_dict, captions_data, coco_images_dir, coco_num_images, coco_out_path,
                      model_resolution, sentence_max_len, TokenBeginIndex, TokenEndIndex):
    images_data = captions_data['images']
    images_count = len(images_data)
    if coco_num_images >= images_count or coco_num_images <= 0:
        image_indices = range(images_count)
    else:
        image_indices = random.sample(range(images_count), coco_num_images)

    result_for_metrics = []
    for img_ind in image_indices:
        cur_image_path = os.path.join(coco_images_dir, images_data[img_ind]['file_name'])
        image_id = images_data[img_ind]['id']
        caption_prediction = create_image_caption(model, cur_image_path, model_resolution, sentence_max_len,
                                                  TokenBeginIndex, TokenEndIndex, id_to_word_dict, printResult=False)
        result_for_metrics.append({'image_id': image_id, 'caption': ' '.join(caption_prediction)})

    weight_filename = os.path.basename(settings.weights_filename)
    weight_filename = os.path.splitext(weight_filename)
    weight_filename = weight_filename[0]

    with open(os.path.join(coco_out_path, '{0}.pred'.format(weight_filename)), 'w') as out_file:
        json.dump(result_for_metrics, out_file)


def perform_testing(settings, id_to_word_dict, captions_data=None, coco_images_dir=None, coco_num_images=None,
                    coco_out_path=None):
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

    if captions_data is not None and coco_images_dir is not None and coco_num_images is not None \
            and coco_out_path is not None:
        calculate_metrics(settings, model, id_to_word_dict, captions_data, coco_images_dir, coco_num_images,
                          coco_out_path, image_shape[:2], sentence_max_len, TokenBeginIndex, TokenEndIndex)
    else:
        create_caption_for_path(settings.test_source, model, image_shape[:2], sentence_max_len, TokenBeginIndex,
                                TokenEndIndex, id_to_word_dict, settings.output_dir,
                                (settings.out_max_width, settings.out_max_height))


def main_func():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights_filename', required=True)
    parser.add_argument('--cuda_devices',
                        default=None)
    parser.add_argument('--model',
                        default='default_model')
    parser.add_argument('--output_dir',
                        default='test_images_captions')
    parser.add_argument('--test_source',
                        default='test_images')
    parser.add_argument('--out_max_width',
                        default=640, type=int)
    parser.add_argument('--out_max_height',
                        default=480, type=int)

    # param to switch to the metric generation
    parser.add_argument('--coco_params', nargs=4,
                        help='the first argument for parameter is the path to the COCO captions file\n'
                             'the second param is the path to corresponding COCO images folder\n'
                             'the third param is the num of images we use (if it is <= 0 then all images will be used)\n'
                             'the forth param is the output_folder\n'
                             'example: --coco_params data/annotations/captions_val2014.json data/val2014 1000 output_test'
                        )

    args = parser.parse_args()

    settings_ini_section_list = ['tests', args.model]
    settings = SettingsKeeper()
    settings.add_ini_file('settings.ini', settings_ini_section_list)
    if os.path.isfile('user_settings.ini'):
        settings.add_ini_file('user_settings.ini', settings_ini_section_list, False)
    settings.add_parsed_arguments(args)

    random.seed(settings.seed)
    np.random.seed(settings.seed)
    tf.set_random_seed(settings.seed)

    if settings.cuda_devices is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = settings.cuda_devices

    with open(settings.id_to_word_file, 'r') as f:
        id_to_word_dict = json.load(f)
        id_to_word_dict = {int(k): v for k, v in id_to_word_dict.items()}
        if args.coco_params is not None:
            captions_file, coco_images_dir, coco_num_images, coco_out_path = args.coco_params
            coco_num_images = int(coco_num_images)
            with open(captions_file, 'r') as cap_f:
                captions_data = json.load(cap_f)
        else:
            captions_data = None
            coco_images_dir = None
            coco_num_images = None
            coco_out_path = None

        perform_testing(settings, id_to_word_dict, captions_data, coco_images_dir, coco_num_images, coco_out_path)


if __name__ == '__main__':
    main_func()
