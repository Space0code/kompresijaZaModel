import os
from PIL import Image
 

# NASTAVITVE
quality = 1    # 80, 40, 5, 1
src_dir = "D:\\OneDrive\\Dokumenti - ne šola\\Konferenca_STeKam\\2023\\git\\kompresijaZaModel\\test_images\\A_People\\images\\images"
dest_dir = "D:\\OneDrive\\Dokumenti - ne šola\\Konferenca_STeKam\\2023\\git\\kompresijaZaModel\\test_images\\A_People\\images\\imagesQ{}_pil\\images".format(quality)
#st = 100 #prvih st slik obdelaj


####################
# PROGRAM
####################

# Function to compress the image
def compressImagePIL(src_file, dest_file, quality):
    # accessing the image file
    image = Image.open(src_file)

    #filepath = os.path.join(os.getcwd(), image_file)
    """
    # maximum pixel size
    maxwidth = 1200
    # opening the file
    image = Image.open(filepath)
    # Calculating the width and height of the original photo
    width, height = image.size
    # calculating the aspect ratio of the image
    aspectratio = width / height
 
    # Calculating the new height of the compressed image
    newheight = maxwidth / aspectratio
 
    # Resizing the original image
    image = image.resize((maxwidth, round(newheight)))
 """
    # Saving the image
    image.save(dest_file, optimize=True, quality=quality)
    return
 


if __name__ == "__main__":
    for fName in os.listdir(src_dir):
        src_file =os.path.join(src_dir, fName)
        dest_file = os.path.join(dest_dir, fName)
        compressImagePIL(src_file, dest_file, quality)
        #st -= 1
        #if (st <= 0) : break

    print("KONEC")