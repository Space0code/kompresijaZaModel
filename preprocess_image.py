import cv2
import numpy as np
import os
import tqdm


def find_almost_constant_pixels(directory: str):
    """
    Finds pixels that are almost constant in the dataset. This is useful for determining
    the area of interest.
    """
    # load all the images in the directory
    images = []
    for filename in tqdm.tqdm(os.listdir(directory)):
        if filename.lower().endswith(".jpg") and filename.upper().startswith("VIRB"):
            img = cv2.imread(os.path.join(directory, filename))
            images.append(np.array(img))
    mega_array = np.array(images)
    assert len(mega_array.shape) == 4 and mega_array.shape[3] == 3, f"Images must be RGB, but were {mega_array.shape}"
    # calculate the standard deviation of each pixel
    # Kljub numpy-ju to nekaj časa traja
    stds = np.std(mega_array, axis=0)
    # save stds as black-white image - for every channel:
    # morda skala ni smiselna, ampak zanekrat je ok
    for i in tqdm.trange(3):
        cv2.imwrite(os.path.join(directory, f"std_{i}.jpg"), stds[:, :, i])
    return stds, mega_array


def cut_image_into_subimages(image, subimage_size):
    """
    Cuts the image into subimages of size subimage_size.
    :param image: image to be cut
    :param subimage_size: the size of the subimages
    :return: a list of subimages
    """
    subimages = []
    for y in range(0, image.shape[0], subimage_size):
        for x in range(0, image.shape[1], subimage_size):
            subimages.append((x, y, image[y:y + subimage_size, x:x + subimage_size]))
            # cv2.imwrite(f"test_moje/kos_{x}_{y}_{subimage_size}.jpg", subimages[-1][-1])
    return subimages


def join_subimages(subimages, final_size):
    # Create an empty canvas for the final image
    data_type = subimages[0][-1].dtype
    final_image = np.zeros(final_size, dtype=data_type)

    # Iterate over the pieces and coordinates, and place each piece in the final image
    for x, y, piece in subimages:
        final_image[y: y + piece.shape[0], x: x + piece.shape[1]] = piece
    # cv2.imwrite("test_moje/izhod.jpg", final_image)
    return final_image


# pieces = cut_image_into_subimages(cv2.imread("620_VIRB/VIRB0018-3060.JPG"), 1000)
# image2d = (cv2.imread("620_VIRB/18_3061_c.png")[:, :, 0]).squeeze()
# pieces = cut_image_into_subimages(image2d, 250)
# join_subimages(pieces, image2d.shape)
# s, m = find_almost_constant_pixels("620_VIRB")
