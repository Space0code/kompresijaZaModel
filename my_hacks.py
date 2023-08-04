import cv2
import matplotlib.pyplot as plt
import numpy as np


image = cv2.imread("620_VIRB/VIRB0018-3060.JPG")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Step 3: Converting to Grayscale
blur = cv2.GaussianBlur(gray, (5, 5), 0)

# Step 4: Gaussian Blur
edges = cv2.Canny(blur, 50, 150)

# Step 5: Canny Edge Detection
height, width = image.shape[:2]
roi_vertices = [(0, height * 2 / 3), (width/2, height*2/3), (width, height * 2 / 3)]
mask_color = 255
mask = np.zeros_like(edges)
cv2.fillPoly(mask, np.array([roi_vertices], dtype=np.int32), mask_color)
masked_edges = cv2.bitwise_and(edges, mask)

# Step 6: Region of Interest
lines = cv2.HoughLinesP(masked_edges, rho=6, theta=np.pi/60, threshold=160, minLineLength=40, maxLineGap=25)

# Step 7: Hough Transform
line_image = np.zeros_like(image)
for line in lines:
    x1, y1, x2, y2 = line[0]
    cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 5)

# Step 8: Drawing the Lines
final_image = cv2.addWeighted(image, 0.8, line_image, 1, 0)

# Step 9: Overlaying the Lines on the Original Image
plt.imshow(final_image)

# Step 10: Display Image
plt.show()

"""
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# plt.imshow(gray)
# plt.show()

w = 131
blur = cv2.GaussianBlur(gray, (w, w), 0)
# plt.imshow(blur, cmap="gray")
# plt.title("Gaussian Blur")
# plt.show()

for w in [5, 15, 31, 61, 121, 251, 511]:
    blur = cv2.GaussianBlur(gray, (w, w), 0)
    plt.imshow(blur, cmap="gray")
    plt.title(f"Edges for w={w}.")
    plt.show()

lower = 100
edges = cv2.Canny(gray, lower, 200)
plt.imshow(edges, cmap="gray")
plt.title(f"Edges for w={w} and low={lower}")
# plt.imsave(f"img_{w}_{lower}.png", edges, cmap="gray")

# plt.imshow(edges, cmap="gray")
# convert into grey scale image
def grey(image):
    image = np.asarray(image)
    return cv2.cvtColor(
        image, cv2.COLOR_RGB2GRAY
    )  # Gaussian blur to reduce noise and smoothen the image


def gauss(image):
    return cv2.GaussianBlur(image, (5, 5), 0)  # Canny edge detection


def canny(image):
    edges = cv2.Canny(image, 50, 150)
    return edges


"""
