"""
Program se izvaja v štirih korakih:
1) Model za prepoznavo razpok se požene na originalnih (OG) slikah.
2) OG slike stisnemo
    - za različne faktorje,
    - z dvema različnima programoma za kompresijo: 
        - eden (pcaColor) temelji na singularnem razcepu (SVD oz. PCA),
        - drugi (pil) temelji na ??? TODO.
3) Model za prepoznavo razpok poženemo nad kompresiranimi slikami. 
4) Rezultate (slike), ki jih izpljune model primerjamo z različnimi metrikami:
    - intersection over union,
    - (average) precision,
    - (average) recall,
    - mean squared error (mse).
    
    Rezultate shranimo v csv datoteko. :)

"""


#########
# imports
#########
import os
from pil_kompresor import compressImagePIL 
from pcaColor_kompresor import compressImagePCA
import argparse
import sys
import csv
import cv2
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from inference_unet import main
from metrics import *


# prestavimo argumente sem, ker bo treba tako ali tako to pognati preko ukazne vrstice
parser = argparse.ArgumentParser()
parser.add_argument('-img_dir',type=str, help='input dataset directory', default="./test_moje")
parser.add_argument('-model_path', type=str, help='trained model path', default="./models/model_unet_vgg_16_best.pt")
parser.add_argument('-model_type', type=str, choices=['vgg16', 'resnet101', 'resnet34'], default='vgg16')
parser.add_argument('-out_viz_dir', type=str, default='', required=False, help='visualization output dir')
parser.add_argument('-out_pred_dir', type=str, required=False,  help='prediction output dir', default="")
parser.add_argument('-threshold', type=float, default=0.2, help='threshold to cut off crack response')
parser.add_argument('-subimage_size', type=int, default=1000, help='cut image into smaller parts first to avoid dataloss when resizing')
parser.add_argument('-results_dir', type=str, required=True, help='where to store csv file with results of comparisons', default='./results')
parser.add_argument('-compressed_dir', type=str, required=True, help='where to store compressed images', default='./compressed')
args = parser.parse_args()


# csv datoteke se shranijo v results_dir
results_dir = args.results_dir
os.makedirs(results_dir, exist_ok=True)


# ustvari tabelo Q - kvalitete za stopnje stiskanja:
# odštejemo 1, 2, 4, 8 ... od popolne.
Q = [99, 95, 90, 75, 50, 25, 10]
# for i in range(20):
#     q = 5 * (i + 1)
#     Q.append(q)
#print(Q)

# ustvari tabelo rangov
R = [2 ** i for i in range(9)]
# for i in range (9):
#     R.append(2**i)


##############################
# 1) poženi model na OG slikah
##############################

print("Poganjam model na originalih...")

out_pred_dir = os.path.join(args.out_pred_dir, 'OG')
main(args.out_viz_dir, out_pred_dir, args.model_type, args.model_path, args.img_dir, args.subimage_size, args.threshold)

###################################################
# 2) stisni na q kvalitete oz. na r ranga in shrani
###################################################

print("Stiskam slike...")

size_percentage_file = open(os.path.join(results_dir, 'size_percentage.csv'), 'w', encoding='UTF8', newline='')
writer = csv.writer(size_percentage_file)
header = ["OG_file", "compressed_file", "OG_size", "compressed_size", "compressed/OG"]
writer.writerow(header)

for q in Q:
    ### PIL_KOMPRESOR stiska in shranjuje:
    dest_dir = os.path.join(args.compressed_dir, 'compressedQ{:03}_pilKompresor'.format(q))
    os.makedirs(dest_dir, exist_ok=True)
    for fName in os.listdir(args.img_dir):
        src_file = os.path.join(args.img_dir, fName)
        dest_file = os.path.join(dest_dir, fName)
        if not os.path.isfile(src_file):
            continue
        compressImagePIL(src_file, dest_file, q)

        # zapiši razmerje velikosti v size_percentage.csv
        if (os.path.isfile(src_file) and os.path.isfile(dest_file)):
            OG_size = os.path.getsize(src_file)
            compr_size = os.path.getsize(dest_file)
            row = [src_file, dest_file, OG_size, compr_size, compr_size/OG_size]
            writer.writerow(row)
        else:
            row = [src_file, dest_file, '/', '/', '/']
            writer.writerow(row)

# pusti vrstico fraj 
writer.writerow(["", "", ""])

for r in R:
    ### SVD_KOMPRESOR stiska in shranjuje:
    dest_dir = os.path.join(args.compressed_dir, 'compressedR{:04}_pcaKompresor'.format(r))
    os.makedirs(dest_dir, exist_ok=True)
    for fName in os.listdir(args.img_dir):
        src_file = os.path.join(args.img_dir, fName)
        dest_file = os.path.join(dest_dir, fName)
        if not os.path.isfile(src_file):
            continue
        compressImagePCA(src_file, dest_file, r)

        # zapiši razmerje velikosti v size_percentage.csv
        if (os.path.isfile(src_file) and os.path.isfile(dest_file)):
            OG_size = os.path.getsize(src_file)
            compr_size = os.path.getsize(dest_file)
            row = [src_file, dest_file, OG_size, compr_size, compr_size/OG_size]
            writer.writerow(row)
        else:
            row = [src_file, dest_file, '/', '/', '/']
            writer.writerow(row)


# pomembna vrstica!
size_percentage_file.close()

################################################
# 3) poženi model nad vsako mapo stisnjenih slik
################################################

print("Poganjam model na stisnjenih slikah...")

for directory in os.listdir(args.compressed_dir):
    if not os.path.isdir(os.path.join(args.compressed_dir, directory)):
        continue
    compressed_dir = os.path.join(args.compressed_dir, directory)
    out_pred_dir = os.path.join(args.out_pred_dir, directory)
    #out_viz_dir = ''    # TODO popravi
    os.makedirs(out_pred_dir, exist_ok=True)
    main(args.out_viz_dir, out_pred_dir, args.model_type, args.model_path, compressed_dir, args.subimage_size, args.threshold)

#################################################
# 4) primerjaj rezultate -> zapiši v csv datoteko
#################################################

# predikcije na osnovi originalov so v mapi 'OG'
# ostale predikcije so v mapah 'compressed*'

print("Primerjam rezultate...")

out_pred_dir = args.out_pred_dir
OG_dir = os.path.join(out_pred_dir, 'OG')


os.makedirs(results_dir, exist_ok=True)
res_file = open(os.path.join(results_dir, 'results.csv'), 'w', encoding='UTF8', newline='')
writer = csv.writer(res_file)

header = ["direktorij", "slika", "presek/unija", "precision", "recall", "mse"]
writer.writerow(header)

# za vsak direktorij izračunaj metrike:
for d in os.listdir(out_pred_dir):
    if not os.path.isdir(os.path.join(out_pred_dir, d)):
        continue

    for i in os.listdir(os.path.join(out_pred_dir, d)):
        # bolje bi bilo, če bi img_OG naredili samo 1x za vsako OG sliko, 
        # vendar mislim, da ne bi smelo biti bistvene časovne razlike
        name_OG = os.path.join(OG_dir, i)
        name_pred = os.path.join(out_pred_dir, d, i)
        
        img_OG = cv2.imread(name_OG) 
        img_pred = cv2.imread(name_pred) 

        # normaliziranje obeh slik
        img_OG = cv2.normalize(img_OG, None, 0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        img_pred = cv2.normalize(img_pred, None, 0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)

        # računanje metrik
        iou = intersection_over_union(img_OG, img_pred)
        prec = precision(img_OG, img_pred, 0.5)
        rec = recall(img_OG, img_pred, 0.5)
        mse1 = mse(img_OG, img_pred)

        # shranjevanje metrik
        row = [d, i, iou, prec, rec, mse1]
        writer.writerow(row)

# pomembna vrstica!
res_file.close()

print("Konec programa.")