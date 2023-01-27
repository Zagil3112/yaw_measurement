import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import hsv_to_rgb


cohete = cv.imread('Angle_Testing/cohete3.png')
plt.imshow(cohete)
plt.show()

cohete = cv.cvtColor(cohete, cv.COLOR_BGR2RGB)
plt.imshow(cohete)
plt.show()

# Colored 3D Scatter Plot 
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import colors

r, g, b = cv.split(cohete)
fig = plt.figure()
axis = fig.add_subplot(1, 1, 1, projection="3d")

pixel_colors = cohete.reshape((np.shape(cohete)[0]*np.shape(cohete)[1], 3))
norm = colors.Normalize(vmin=-1.,vmax=1.)
norm.autoscale(pixel_colors)
pixel_colors = norm(pixel_colors).tolist()

axis.scatter(r.flatten(), g.flatten(), b.flatten(), facecolors=pixel_colors, marker=".")
axis.set_xlabel("Red")
axis.set_ylabel("Green")
axis.set_zlabel("Blue")
#plt.show()


# HSV 3D plot
hsv_cohete = cv.cvtColor(cohete, cv.COLOR_RGB2HSV)
#plt.imshow(hsv_cohete)
"""
h, s, v = cv.split(hsv_cohete)
fig = plt.figure()
axis = fig.add_subplot(1, 1, 1, projection="3d")
#axis.scatter(h.flatten(), s.flatten(), v.flatten(), facecolors=pixel_colors, marker=".")
axis.set_xlabel("Hue")
axis.set_ylabel("Saturation")
axis.set_zlabel("Value")
plt.show()
"""
# Picking Out a Range

light_orange = (1, 190, 200)
dark_orange = (18, 255, 255)

carton_oscuro=(32.3077, 30.2326, 33.7255) # HSV
carton_claro=(31.2000, 23.4742, 83.5294)  # HSV

verde_claro = (127.9518, 90.2174, 36.0784)
verde_oscuro=(128.0000, 78.9474, 29.8039)

amarillo_oscuro = (51, 90, 79)
amarillo_claro=(42, 89, 85)

#lo_square = np.full((10, 10, 3), light_orange, dtype=np.uint8) / 255.0
#do_square = np.full((10, 10, 3), dark_orange, dtype=np.uint8) / 255.0
square1 = np.full((10, 10, 3), verde_claro, dtype=np.uint8) / 255.0
square2 = np.full((10, 10, 3), verde_oscuro, dtype=np.uint8) / 255.0

plt.subplot(1, 2, 1)
plt.imshow(hsv_to_rgb(square1))
plt.subplot(1, 2, 2)
plt.imshow(hsv_to_rgb(square2))

plt.show()


mask = cv.inRange(hsv_cohete, verde_claro, verde_oscuro )
result = cv.bitwise_and(cohete, cohete, mask=mask)

plt.subplot(1, 2, 1)
plt.imshow(mask, cmap="gray")
plt.subplot(1, 2, 2)
plt.imshow(result)
plt.show()

cv.imwrite('Angle_Testing/cohete_procesado.png',mask)
cv.imwrite('Angle_Testing/cohete_procesado2.png',result)