import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import sys
import os

# NASTAVITVE
#src_dir = "D:\\OneDrive\\Dokumenti_ne_sola\\Konferenca_STeKam\\2023\\git\\kompresijaZaModel\\test_images\\A_People\\images\\images"
#dest_dir = "D:\\OneDrive\\Dokumenti_ne_sola\\Konferenca_STeKam\\2023\\git\\kompresijaZaModel\\test_images\\faces\\A_people_first100\\compressedXX_kompresor1"
#pca_components = 20

####################
# PROGRAM
####################



#print("kompresor1_pcaColor.py START")

def compressImagePCA(src_file, dest_file, pca_components):
    #src = os.path.join(src_dir, fName)
    img_raw = cv2.imread(src_file)  
    img = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)
    if (img.shape[0] <= pca_components or img.shape[1] <= pca_components) :
        #print("img.shape <= pca_components")
        return

    #print("i=", i, " - og filesize: ", sys.getsizeof(img.tobytes()))

    # split img into rgb
    r, g, b = cv2.split(img)
    # normalise
    r, g, b = r / 255, g / 255, b / 255

    # naredi PCA na vsako komponento (r,g,b) posebej
    pca_r = PCA(n_components=pca_components)
    reduced_r = pca_r.fit_transform(r)
    pca_g = PCA(n_components=pca_components)
    reduced_g = pca_g.fit_transform(g)
    pca_b = PCA(n_components=pca_components)
    reduced_b = pca_b.fit_transform(b)

    # to bi lahko, če bi htel
    #combined = np.array([reduced_r, reduced_g, reduced_b])

    reconstructed_r = pca_r.inverse_transform(reduced_r) * 255
    reconstructed_g = pca_g.inverse_transform(reduced_g) * 255
    reconstructed_b = pca_b.inverse_transform(reduced_b) * 255

    img_reconstructed = cv2.merge((reconstructed_r, reconstructed_g, reconstructed_b))
    img_reconstructed = cv2.cvtColor(img_reconstructed.astype('float32'), cv2.COLOR_RGB2BGR)
    #print(img_reconstructed.shape)
    #print("i=", i, " - rec filesize: ", sys.getsizeof(img_reconstructed.tobytes()))

    #result_img_name = "image{:05}.jpg".format(i)
    #dest = os.path.join(dest_dir, fName)
    cv2.imwrite(dest_file, img_reconstructed)



#print("kompresor1_pcaColor.py KONEC")