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
parser.add_argument('-out_pred_dir', type=str, required=False,  help='prediction output dir', default="./test_results_moje")
parser.add_argument('-threshold', type=float, default=0.2, help='threshold to cut off crack response')
parser.add_argument('-subimage_size', type=int, default=448, help='cut image into smaller parts first to avoid dataloss when resizing')
args = parser.parse_args()

# ustvari tabelo Q - kvalitete za stopnje stiskanja
Q = [1]
for i in range(20):
    q = 5 * (i + 1)
    Q.append(q)
#print(Q)

# ustvari tabelo rangov
R = []
for i in range (12):
    R.append(2**i)

"""
##############################
# 1) poženi model na OG slikah
##############################

out_pred_dir = os.path.join(args.out_pred_dir, 'OG')
main(args.out_viz_dir, out_pred_dir, args.model_type, args.model_path, args.img_dir, args.subimage_size, args.threshold)

###################################################
# 2) stisni na q kvalitete oz. na r ranga in shrani
###################################################

# mapa, kjer so shranjene OG slike
img_dir = args.img_dir #  "D:\\OneDrive\\Dokumenti_ne_sola\\Konferenca_STeKam\\2023\\git\\kompresijaZaModel\\test_images\\A_People\\images_first100\\images\\images"

for q in Q:
    ### PIL_KOMPRESOR stiska in shranjuje:
    # nastavi dest_dir
    # default: TODO: spremeni dest_dir, da mora biti podan kot argument
    dest_dir = os.path.join(img_dir, f'compressedQ{q}_pilKompresor')
    os.makedirs(dest_dir, exist_ok=True)
    for fName in os.listdir(img_dir):
        src_file = os.path.join(img_dir, fName)
        dest_file = os.path.join(dest_dir, fName)
        if not os.path.isfile(src_file):
            continue
        compressImagePIL(src_file, dest_file, q)

for r in R:
    ### SVD_KOMPRESOR stiska in shranjuje:
    # TODO nastavi dest_dir
    # default:
    dest_dir = os.path.join(img_dir, f'compressedR{r}_pcaKompresor')
    os.makedirs(dest_dir, exist_ok=True)
    for fName in os.listdir(img_dir):
        src_file = os.path.join(img_dir, fName)
        dest_file = os.path.join(dest_dir, fName)
        if not os.path.isfile(src_file):
            continue
        compressImagePCA(src_file, dest_file, r)


################################################
# 3) poženi model nad vsako mapo stisnjenih slik
################################################


for directory in os.listdir(args.img_dir):
    if not os.path.isdir(os.path.join(args.img_dir, directory)):
        continue
    img_dir = os.path.join(args.img_dir, directory)
    out_pred_dir = os.path.join(args.out_pred_dir, directory)
    out_viz_dir = ''    # TODO popravi
    os.makedirs(out_pred_dir, exist_ok=True)
    main(out_viz_dir, out_pred_dir, args.model_type, args.model_path, img_dir, args.subimage_size, args.threshold)
"""
#################################################
# 4) primerjaj rezultate -> zapiši v csv datoteko
#################################################

# predikcije na osnovi originalov so v mapi 'OG'
# ostale predikcije so v mapah 'compressed*'


out_pred_dir = args.out_pred_dir
OG_dir = os.path.join(out_pred_dir, 'OG')

# csv naj se shrani v isto mapo, kot so se shranile kompresirane slike
results_dir = out_pred_dir
os.makedirs(results_dir, exist_ok=True)
res_file = open(os.path.join(results_dir, 'results.csv'), 'w', encoding='UTF8', newline='')
writer = csv.writer(res_file)

row = ["direktorij", "slika", "presek/unija", "precision", "recall", "mse"]
writer.writerow(row)

# za vsak direktorij izračunaj metrike:
for d in os.listdir(out_pred_dir):
    if not os.path.isdir(os.path.join(out_pred_dir, d)):
        continue

    for i in os.listdir(os.path.join(out_pred_dir, d)):
        # bolje bi bilo, če bi img_OG naredili samo 1x za vsako OG sliko, 
        # vendar mislim, da ne bi smelo biti bistvene časovne ralzike
        img_OG = cv2.imread(os.path.join(OG_dir, i)) 
        img_pred = cv2.imread(os.path.join(out_pred_dir, d, i)) 

        # računanje metrik
        iou = intersection_over_union(img_OG, img_pred)
        prec = precision(img_OG, img_pred, 0.5)
        rec = recall(img_OG, img_pred, 0.5)
        mse1 = mse(img_OG, img_pred)

        # shranjevanje metrik
        row = [d, i, iou, prec, rec, mse1]
        writer.writerow(row)


res_file.close()