import csv
import os
import cv2
from metrics import *

out_pred_dir = r'D:\OneDrive\Dokumenti_ne_sola\Konferenca_STeKam\2023\git\git\kompresijaZaModel\test_results_moje'
OG_dir = os.path.join(out_pred_dir, 'OG')

results_dir = r'D:\OneDrive\Dokumenti_ne_sola\Konferenca_STeKam\2023\git\git\kompresijaZaModel\rezultati_primerjav'
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