import os
import re
from typing import List

import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image


def mkdir_if_not_exists(*paths):
    for p in paths:
        if not os.path.exists(p):
            os.makedirs(p, exist_ok=True)


def check_slash(path):
    if path.endswith('/'):
        return path
    else:
        return path + '/'


def list_image_files(directory, sort_function=None):
    return list_files(directory, endings=('.png', '.bmp', '.jpeg'), sort_function=sort_function)


def list_files(directory, endings=None, sort_function=None, full_path=True):
    paths = []

    names = sorted(os.listdir(directory), key=sort_function)
    if endings is not None:
        names = filter(lambda x: x.endswith(endings), names)

    if full_path:
        for name in names:
            paths.append(check_slash(directory) + name)
        return paths
    else:
        return names


def list_dirs(directory, full_path=True):
    contents = list_files(directory=directory, full_path=True)
    dirs = list(filter(lambda x: os.path.isdir(x), contents))
    if full_path:
        return dirs
    else:
        return [file_name(x) for x in dirs]


def list_files_with_patterns(path, patterns, endings, regex=True, sort_function=None):
    path_lists = []
    files = list_files(path, endings=endings, sort_function=sort_function)

    for p in patterns:
        if regex:
            path_lists.append(list(filter(lambda x: p.match(x), files)))
        else:
            path_lists.append(list(filter(lambda x: p in x, files)))
    return path_lists


def scan_for_existing_folders(folder, name, delim='_'):
    """
    Check the contents of the specified folder for enumerated subfolders with specified name.
    Returns the path to the next folder in the enumeration.
    Naming scheme: 'name''delim'0, 'name''delim'1, 'name''delim'2, ...
    If no folder is present, the path to the first one will be returned.

    :param folder: Folder to search for subfolders.
    :param name: Name base of enumerated subfolders.
    :param delim: Delimiter between name and enumeration.
    :return: The path to the next folder in the eneumeration.
    """
    contents = list_dirs(folder, full_path=False)
    contents = list(filter(lambda x: x.startswith(name), contents))

    if len(contents) == 0:
        return folder + '/' + name + delim + str(0)

    contents = [int(x.replace(name + delim, '')) for x in contents]
    contents = list(sorted(contents))
    return folder + '/' + name + delim + str(contents[-1] + 1)


def file_name(path, delim='/'):
    split = path.split(delim)
    return split[-1]


def strip_ending(path_or_name: str):
    split = path_or_name.split('.')
    assert len(split) == 2
    return split[0]


def ending(path_or_name: str):
    split = path_or_name.split('.')
    assert len(split) == 2
    return '.' + split[1]


def folder(path_to_file, delim='/'):
    """
    Extract the folder path containing the file with specified path. Specified delimiter will be appended to the found
    folder path.

    :param path_to_file: Path to some file.
    :param delim: Path delimiter.
    :return: Path to folder containing specified file.
    """
    name = file_name(path=path_to_file, delim=delim)
    if '.' not in name:
        raise ValueError('Specified path: \'{}\' appears not to point to a file.'.format(path_to_file))
    split = path_to_file.split(delim)
    return delim.join(split[0:-1]) + delim


def filter_split(strings: List[str], patterns: List[str]):
    splits = []
    for pat in patterns:
        splits.append(list(filter(lambda x: pat in x, strings)))
    return tuple(splits)


def collect_files(root, regex):
    paths = []
    for path, subdirs, files in os.walk(root):
        for file in files:
            if re.match(regex, file):
                paths.append(check_slash(path.replace('\\', '/')) + file)
    return paths


def partition(lists, ratio, seed):
    x = [np.array(l) for l in lists]

    for arr in x:
        if arr.shape[0] != x[0].shape[0]:
            raise ValueError('All lists must have the same shape in first dimension!')

    if seed is not None:
        np.random.seed(seed)

    size = x[0].shape[0]

    idxs = np.arange(size)
    np.random.shuffle(idxs)

    split_idx = int(np.floor(size * (1 - ratio)))

    firsts = [arr[idxs[0:split_idx]] for arr in x]
    seconds = [arr[idxs[split_idx:size]] for arr in x]

    return firsts, seconds


def partition_train_test(lists, test_ratio=0.1, seed=None):
    firsts, seconds = partition(lists, ratio=test_ratio, seed=seed)
    return firsts, seconds


def partition_train_test_validation(lists, test_ratio=0.1, validation_ratio=0.2, seed=None):
    ref_ratio = test_ratio + validation_ratio
    train_partitions, tests_partitions = partition_train_test(lists, test_ratio=ref_ratio, seed=seed)

    validation_of_ref_ratio = validation_ratio / ref_ratio
    tests_partitions, val_partitions = partition_train_test(tests_partitions, test_ratio=validation_of_ref_ratio,
                                                            seed=seed)
    return train_partitions, tests_partitions, val_partitions


def create_iterators_switch(session, *iterators):
    if len(iterators) < 2:
        raise ValueError('Need to switch at least two datasets!')

    handle = tf.placeholder(tf.string, shape=[])
    chained_iterator = tf.data.Iterator.from_string_handle(handle, iterators[0].output_types,
                                                           iterators[0].output_shapes)
    accesses = list(map(lambda x: session.run(x.string_handle()), iterators))

    return handle, accesses, chained_iterator


def shuffle_lists(*lists, seed=None):
    size = len(lists[0])
    for l in lists:
        assert len(l) == size

    np.random.seed(seed)

    arrs = [np.array(x) for x in lists]
    s = np.arange(size)
    np.random.shuffle(s)
    arrs = [list(x[s]) for x in arrs]

    return tuple(arrs)


def is_list(thing):
    return isinstance(thing, list)


def plot_imgs(imgs, shape=None):
    if is_list(imgs):
        plots = imgs
    else:
        plots = [imgs]

    if shape is None:
        sh = (1, len(plots))
    else:
        if len(plots) > shape[0] * shape[1]:
            raise ValueError('Can\'t show more images than there are cells in shape {}'.format(repr(shape)))
        sh = shape

    row = 0
    col = 0
    for img in plots:

        plt.subplot2grid(sh, (row, col))

        if img is not None:
            plt.imshow(img)

        if col+1 == sh[1]:
            col = 0
            row += 1
        else:
            col += 1
    plt.show()


def validate_names(pngs, npzs, png_prefix, npz_prefix, pickles=None, pickle_prefix=None):
    print('Validating file names...')
    pngs = [strip_ending(file_name(x)) for x in pngs]
    pngs = [x.replace(png_prefix, '') for x in pngs]

    npzs = [strip_ending(file_name(x)) for x in npzs]
    npzs = [x.replace(npz_prefix, '') for x in npzs]

    if pickle_prefix is not None and pickles is not None:
        pickles = [strip_ending(file_name(x)) for x in pickles]
        pickles = [x.replace(pickle_prefix, '') for x in pickles]

        assert len(pngs) == len(npzs) == len(pickles)

        for p, n, pk in zip(pngs, npzs, pickles):
            if p != n != pk:
                raise ValueError(f'Wrong or missing files:\n {p} + vs. + {n} + vs. + {pk}')
    else:
        assert len(pngs) == len(npzs)

        for p, n in zip(pngs, npzs):
            if p != n:
                raise ValueError(f'Wrong or missing files:\n {p} + vs. + {n}')


def numbered_file_names_sort(x):
    """
    Key function for 'sorted'.\n
    Sorts a list of Strings according to al numbers contained in the String.\n
    All numbers will be simply concatenated.

    :param x: input of the key function
    :return: all numbers contained in x concatenated and converted to int
    """
    nums = [s for s in re.findall(r'\d+', x)]
    num = int(''.join(nums))
    return num


def blend_images(img1, img2, factor, mask=None):
    pil1 = Image.fromarray(img1)
    pil2 = Image.fromarray(img2)

    blended = np.array(Image.blend(pil1, pil2, factor))

    if mask is None:
        m = img2 == 0
    else:
        m = mask == 0
        if len(img1.shape) == 3:
            m = np.expand_dims(m, axis=-1)

    return (img1 * m) + (blended * np.logical_not(m))


def mask_to_color_image(mask, color=(255, 0, 0)):
    stack = []
    for channel_col in color:
        stack.append(mask * channel_col)
    return np.stack(stack, axis=-1)


def filter_components(masks: List[np.ndarray], size, mode='MIN', bg_cmp_idx=0):
    """
    Remove all connected components in specified binary images (inplace) that to not meet fulfill the specified size
    condition.

    :param masks: List of binary images to filter
    :param size: Size condition
    :param mode: One of
            'MIN' := all components smaller than size will be removed,
            'MAX' := all components larger than size will be removed
    :param bg_cmp_idx: Index of background component
    """
    for mask in masks:
        retval, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

        cmp_idxs_wo_bg = list(range(stats.shape[0]))
        cmp_idxs_wo_bg.remove(bg_cmp_idx)

        idxs_to_remove = []

        if retval < 2:
            return
        elif retval > 2:
            for i in cmp_idxs_wo_bg:
                comp_size = stats[i, cv2.CC_STAT_AREA]

                if mode is 'MIN':
                    if comp_size < size:
                        idxs_to_remove.append(i)
                elif mode is 'MAX':
                    if comp_size > size:
                        idxs_to_remove.append(i)

            for i in idxs_to_remove:
                mask[labels == i] = 0
