import numpy as np
from matplotlib import pyplot as plt


img_array = np.load('data\processed\heatmaps\2-in-1_Space_Dye_Athletic_Tank\img_00000001.npy')
plt.imshow(img_array, cmap='gray')
plt.show()