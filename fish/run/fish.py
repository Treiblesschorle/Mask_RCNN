import re

import numpy as np
from scipy.misc import imread

import mrcnn.model as modellib
from mrcnn import utils
from mrcnn.config import Config
import util.util as futil
import util.stitch as stitch
import util.mean_ap as ap
from util.util import plot_imgs

TILE_SIZE = (448, 448)


class FishConfig(Config):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    NAME = "fish"

    GPU_COUNT = 1
    IMAGES_PER_GPU = 3

    NUM_CLASSES = 1 + 1

    IMAGE_MIN_DIM = 448
    IMAGE_MAX_DIM = 448

    STEPS_PER_EPOCH = 1000


class FishDataset(utils.Dataset):
    """Generates the shapes synthetic dataset. The dataset consists of simple
    shapes (triangles, squares, circles) placed randomly on a blank surface.
    The images are generated on the fly. No file access required.
    """

    def load(self, directory):
        self.add_class("fish", 1, "fish")

        def numbered_file_names_sort(x):
            nums = [s for s in re.findall(r'\d+', x)]
            nums = ['1'] + nums
            num = int(''.join(nums))
            return num

        def enumerated_file_names_sort(x):
            nums = [int(s) for s in re.findall(r'\d+', x)]
            return tuple(nums)

        img_paths, masks_paths = futil.list_files_with_patterns(directory, patterns=('.png', '.npz'),
                                                                endings=('.png', '.npz'), regex=False,
                                                                sort_function=enumerated_file_names_sort)

        futil.validate_names(pngs=img_paths, npzs=masks_paths, png_prefix='img_', npz_prefix='masks_')

        for i, (path, mp) in enumerate(zip(img_paths, masks_paths)):
            self.add_image("fish", image_id=i, path=path, mask_path=mp)

    def load_image(self, image_id):
        info = self.image_info[image_id]
        img = imread(info['path'])
        # futil.plot_imgs(img)
        return img

    def image_reference(self, image_id):
        info = self.image_info[image_id]
        return info

    def load_mask(self, image_id):
        info = self.image_info[image_id]
        masks = np.load(info['mask_path'])['arr_0']
        class_ids = [1] * masks.shape[-1]
        # futil.plot_imgs([np.squeeze(x) for x in np.split(masks, masks.shape[-1], axis=-1)])
        return masks.astype(np.bool), np.array(class_ids, dtype=np.int32)


def train():
    """Train the model."""
    # Training dataset.
    dataset_train = FishDataset()
    dataset_train.load('D:/Dave/MRCNN/data/train')
    dataset_train.prepare()

    # Validation dataset
    dataset_val = FishDataset()
    dataset_val.load('D:/Dave/MRCNN/data/test')
    dataset_val.prepare()

    model_dir = 'D:/Dave/MRCNN/runs'
    futil.mkdir_if_not_exists(model_dir)
    config = FishConfig()
    model = modellib.MaskRCNN(mode="training", config=config,
                              model_dir=model_dir)
    model.load_weights(model.get_imagenet_weights(), by_name=True)

    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=30,
                layers='heads')


def image_and_labelling(imgs_list, masks_list, gt_masks_list, size):
    stitched_img = stitch.stitch_overlapping_tiles(imgs_list, size, 100)[0]

    if len(masks_list) > 0:
        extend_masks(masks_list)
        stitched_masks = stitch.stitch_overlapping_tiles(masks_list, size, 100)
        stitched_masks = np.max(np.stack(stitched_masks, axis=-1), axis=-1)
    else:
        stitched_masks = None

    stitched_gt_masks = stitch.stitch_overlapping_tiles(gt_masks_list, size, 100)

    for i, x in enumerate(stitched_gt_masks):
        x[:] = x * (i + 1)

    stitched_gt_masks = np.max(np.stack(stitched_gt_masks, axis=-1), axis=-1)
    stitched_gt_masks = np.squeeze(stitched_gt_masks)

    return stitched_img, stitched_gt_masks, stitched_masks


def extend_masks(masks_list):
    zero_tile = np.zeros(TILE_SIZE)
    max_instances = max([len(x) for x in masks_list])
    for ml in masks_list:
        if len(ml) != max_instances:
            ml.extend([zero_tile] * (max_instances - len(ml)))


def predict():
    class InferenceConfig(FishConfig):
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1

    inference_config = InferenceConfig()

    # Validation dataset
    dataset_val = FishDataset()
    dataset_val.load('D:/Dave/MRCNN/data/test')
    dataset_val.prepare()

    model_dir = 'D:/Dave/MRCNN/runs'
    # Recreate the model in inference mode
    model = modellib.MaskRCNN(mode="inference",
                              config=inference_config,
                              model_dir=model_dir)

    model_path = model.find_last()
    model.load_weights(model_path, by_name=True)

    img_num = 0
    tile_num = 0
    infos = []

    size_and_infos_per_img = []
    for info in dataset_val.image_info:

        nums = [int(s) for s in re.findall(r'\d+', info['mask_path'])]

        if nums[0] != img_num:
            if tile_num == 98:
                size = (2028, 2704)
            elif tile_num == 76:
                size = (1538, 2704)
            else:
                raise ValueError

            size_and_infos_per_img.append((size, infos.copy()))

            img_num = nums[0]
            infos.clear()

        infos.append(info)
        tile_num = nums[1]

    size_and_infos_per_img.append((size, infos.copy()))

    aps = []
    for size, infos in size_and_infos_per_img:
        inst_idx = 1
        imgs_list = []
        gt_masks_list = []
        masks_list = []
        for info in infos:
            # print(info)
            original_image, image_meta, gt_class_id, gt_bbox, gt_mask = modellib.load_image_gt(dataset_val,
                                                                                               inference_config,
                                                                                               info['id'],
                                                                                               use_mini_mask=False)
            imgs_list.append((original_image,))
            gt_arr = np.load(info['mask_path'])['arr_0']
            gt_arr = np.split(gt_arr, gt_arr.shape[-1], axis=-1)
            gt_masks_list.append(gt_arr)

            results = model.detect([original_image], verbose=0)
            r = results[0]
            masks = r['masks']
            if masks.shape[-1] != 0:
                scores = r['scores']
                instances = [np.squeeze(x) for x in np.split(masks, masks.shape[-1], axis=-1)]
                instances = [x for x, s in zip(instances, scores) if s > 0.9]

                tmp = []
                for inst in instances:
                    tmp.append(np.int32(inst) * inst_idx)
                    inst_idx += 1

                masks_list.append(tmp)
            else:
                masks_list.append([np.zeros(original_image.shape[0:2])])

        if len(masks_list) == 0:
            print('No instances detected.')
        stitched_img, stitched_gt_masks, stitched_masks = image_and_labelling(imgs_list, masks_list, gt_masks_list,
                                                                              size)
        if stitched_masks is None:
            aps.append(0.0)
            print(0.0)
        else:
            sgm_filter = np.uint8(stitched_gt_masks > 0)
            sm_filter = np.uint8(stitched_masks > 0)
            futil.filter_components([sgm_filter, sm_filter], mode='MIN', size=100)

            stitched_gt_masks = np.int32(stitched_gt_masks * sgm_filter)
            stitched_masks = np.int32(stitched_masks * sm_filter)

            mean_ap = ap.ap_dsb2018([stitched_masks], [stitched_gt_masks], compute_mean=True)
            aps.append(mean_ap[-1])
            print(mean_ap)

            # futil.plot_imgs([stitched_img, stitched_gt_masks, stitched_masks])
            # blend_pred = futil.blend_images(stitched_img, futil.mask_to_color_image(np.uint8(stitched_masks > 0)),
            #                                 factor=0.3)
            # blend_gt = futil.blend_images(stitched_img, futil.mask_to_color_image(np.uint8(stitched_gt_masks > 0)),
            #                               factor=0.3)
            # futil.plot_imgs([blend_pred, blend_gt])

    print(f'Mean ap: {sum(aps) / len(aps)}')


predict()
# train()
