from typing import List, Tuple
import math
import numpy as np


def stitch_overlapping_tiles(tiles: List[Tuple[np.ndarray, ...]], size: Tuple[int, int], border: int):
    """
    Stitches a sequence of overlapping tiles. The order of tiles must be row wise.

    :param tiles: The list of tiles to stitch. Each element of this list can be a tuple containing tiles of several
                  images/arrays.
    :param size: The size of the resulting stitched image.
    :param border: The size of the border/overlap of each tile. This amount will be cropped from each border of each
                   tile.
    :return: A list of stitched arrays/images.
    """
    # First, unpack tiles, i.e. creating lists of tiles for each array.
    tiles_per_img = []
    ts = tiles[0][0].shape
    cols = int(math.ceil(size[1] / (ts[1] - (2 * border))))

    num_tiles = len(tiles)
    assert num_tiles % cols == 0

    num_imgs = len(tiles[0])
    for i in range(num_imgs):
        curr = []
        for t in tiles:
            curr.append(t[i])
        tiles_per_img.append(curr)

    # Stitch the tiles, first stitch tiles vertically to obtain rows. Next, stitch rows.
    imgs = []
    for img_tiles in tiles_per_img:
        rows = []
        row_img = None
        for i, tile in enumerate(img_tiles):
            cropped_tile = tile[border:ts[0] - border, border:ts[1] - border, ...]

            if (i + 1) % cols == 0:
                row_img = np.concatenate([row_img, cropped_tile], axis=1)
                rows.append(row_img)
                row_img = None
            else:
                if row_img is None:
                    row_img = cropped_tile
                else:
                    row_img = np.concatenate([row_img, cropped_tile], axis=1)
        img = np.concatenate(rows, axis=0)

        # Crop the stitched image to the specified size. This will remove the black border if the tile size did not
        # evenly fit into the image.
        img = img[0:size[0], 0:size[1]]
        imgs.append(img)
    return imgs

