"""
1. Ideja poteka programa:
- poženi model na originalnih slikah in shrani rezultate
- stiskaj slike & sproti poganjaj model 
    - original sliko stisni na Q kvalitete (oz. za X %, odvisno od programa)
    - (shrani stisnjeno sliko?)
    - poženi model nad sliko
    - shrani rezultat v mapo (recimo v resultQ{Q}) 
- primerjaj rezultate:
    - ideja: število pikslov, ki se razlikujejo in za kakšen odstotek (odtenek) se razlikujejo
"""

"""
2. Ideja poteka programa:
1) poženi model na originalnih slikah in shrani rezultate
2) for Q in (1, 5, 10, 15, ..., 90, 95):
    - stisni in shrani slike na Q kvalitete
3) for Q in (1, 5, 10, 15, ..., 90, 95):
    - poženi model nad vsako sliko v mapi, kjer so shranjene slike kvalitete Q
    - shrani rezultate
4) primerjaj rezultate:
    - ideja: število pikslov, ki se razlikujejo in za kakšen odstotek (odtenek) se razlikujejo
"""

#######
# imports
#######
import os
from pil_kompresor import compressImagePIL 
from pcaColor_kompresor import compressImagePCA
import argparse

# prestavimo argumente sem, ker bo treba tako ali tako to pognati preko ukazne vrstice
parser = argparse.ArgumentParser()
parser.add_argument('-img_dir',type=str, help='input dataset directory', default="./test_moje")
parser.add_argument('-model_path', type=str, help='trained model path', default="./models/model_unet_vgg_16_best.pt")
parser.add_argument('-model_type', type=str, choices=['vgg16', 'resnet101', 'resnet34'], default='vgg16')
parser.add_argument('-out_viz_dir', type=str, default='', required=False, help='visualization output dir')
parser.add_argument('-out_pred_dir', type=str, required=False,  help='prediction output dir', default="./test_results_moje")
parser.add_argument('-threshold', type=float, default=0.2, help='threshold to cut off crack response')
parser.add_argument('-subimage_size', type=int, default=-1, help='cut image into smaller parts first to avoid dataloss when resizing')
args = parser.parse_args()


#######
# 1) poženi model na OG slikah
#######

#######
# 2) stisni na q kvalitete oz. na r ranga in shrani
#######

# ustvari tabelo Q - kvalitete za stopnje stiskanja
Q = [1]
for i in range(20):
    q = 5 * (i + 1)
    Q.append(q)
#print(Q)


# smiselno hardcodati tole tabelo rangov R !
R = [5, 50, 100]

# mapa, kjer so shranjene OG slike
# nastavi src_dir
# TODO: spremeni na koncu na args.img_dir oz. nekaj takega ...
src_dir = args.img_dir #  "D:\\OneDrive\\Dokumenti_ne_sola\\Konferenca_STeKam\\2023\\git\\kompresijaZaModel\\test_images\\A_People\\images_first100\\images\\images"

for q in Q:

    ### PIL_KOMPRESOR stiska in shranjuje:
    # nastavi dest_dir
    # default:
    dest_dir = os.path.join(src_dir, f'compressedQ{q}_pilKompresor')
    os.makedirs(dest_dir, exist_ok=True)
    for fName in os.listdir(src_dir):
        src_file = os.path.join(src_dir, fName)
        dest_file = os.path.join(dest_dir, fName)
        if not os.path.isfile(src_file):
            continue
        compressImagePIL(src_file, dest_file, q)

for r in R:
    ### SVD_KOMPRESOR stiska in shranjuje:
    # nastavi dest_dir
    # default:
    dest_dir = os.path.join(src_dir, f'compressedR{r}_pcaKompresor')
    os.makedirs(dest_dir, exist_ok=True)
    for fName in os.listdir(src_dir):
        src_file = os.path.join(src_dir, fName)
        dest_file = os.path.join(dest_dir, fName)
        if not os.path.isfile(src_file):
            continue
        compressImagePCA(src_file, dest_file, r)


##############
# 3) poženi model nad vsako mapo stisnjenih slik
##############
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from inference_unet import main


main(args.out_viz_dir, args.out_pred_dir, args.model_type, args.model_path, args.img_dir, args.subimage_size)


###########
# 4) 
##########