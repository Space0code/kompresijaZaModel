import os
import tqdm
import random
import cv2 as cv
import numpy as np

"""
Slika je primerna, če na njej ni zamelgitev.
Primerne slike uredimo glede na to, koliko belega se najde na njej.
"""


def load_labels(path):
    """
    Reads labels file and loads image names from the lines of form

    image-name class coordindates

    (lines are separated by spaces)
    """
    images_with_labels = set()
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            image_name = line[:line.find(" ")].strip()
            images_with_labels.add(image_name)
    return images_with_labels


def process_one_directory(dir_path):
    people_labels = os.path.join(dir_path, "labels_people.txt")
    vehicles_labels = os.path.join(dir_path, "labels_vehicles.txt")
    all_images = set(x[:x.rfind(".")] for x in os.listdir(dir_path) if x.lower().endswith(".jpg"))
    bad_images = load_labels(people_labels) | load_labels(vehicles_labels)
    good_images = all_images - bad_images
    return {os.path.join(dir_path, img) + ".JPG" for img in good_images}, len(all_images), len(good_images)


def find_all_directories(root_dir):
    dirs = []
    for dir_name in os.listdir(root_dir):
        dir_path = os.path.join(root_dir, dir_name)
        label_file = os.path.join(dir_path, "labels_people.txt")
        if os.path.isdir(dir_path):
            if os.path.isfile(label_file):
                dirs.append(dir_path)
            try:
                dirs.extend(find_all_directories(dir_path))
            except PermissionError:
                print(f"Skipping {dir_path}")
    return dirs


def process_all_directories(root_dir):
    good_images = []
    total_total = 0
    total_good = 0
    for directory in tqdm.tqdm(find_all_directories(root_dir)):
        paths, n_all, n_good = process_one_directory(directory)
        good_images.extend(paths)
        total_total += n_all
        total_good += n_good
    print("Malo statistik:", total_total, total_good, total_good / total_total)
    random.seed(1234)
    good_images.sort()
    random.shuffle(good_images)
    return good_images


def run_on_good_images(good_image_paths):
    from inference_unet import main

    best_so_far = 0.0
    with open("image_scores.txt", "w") as f:
        for path, prob_map in main("", "", 'vgg16', "./models/model_unet_vgg_16_best.pt", None, 1000, 0.2, paths=good_image_paths):
            score = np.sum(prob_map)
            print(f"{path};{score}", file=f, flush=True)
            if score > best_so_far:
                best_so_far = score
                print(f"New best: {path};{score}")
                cv.imwrite(filename=os.path.join("best_scores", f'{path.stem}.jpg'), img=(prob_map * 255).astype(np.uint8))


if __name__ == "__main__":
    ok_images = process_all_directories("E:/")
    run_on_good_images(ok_images)

        
