import cv2
import numpy as np

from align_target import align_target
import matplotlib.pyplot as plt
from scipy.sparse.linalg import spsolve as eqsol
import scipy




def poisson_blend(source_image, target_image, target_mask):
    minY, minX = 0, 0
    maxY, maxX = target_image.shape[:-1]
    ranY = maxY - minY
    ranX = maxX - minX
    offset =(0,5)
    target_mask = target_mask[minY:maxY, minX:maxX]
    a = np.float32([[1, 0, offset[0]], [0, 1, offset[1]]])
    source_image = cv2.warpAffine(source_image, a, (ranX, ranY))
    target_mask[target_mask != 0] = 1
    A = scipy.sparse.lil_matrix((ranX,ranX))
    A.setdiag(-1, -1)
    A.setdiag(4)
    A.setdiag(-1, 1)
    A= scipy.sparse.block_diag([A] * ranY).tolil()
    A.setdiag(-1, 1 * ranX)
    A.setdiag(-1, -1 * ranX)
    for y in range(1, ranY - 1):
        for x in range(1, ranX - 1):
            if target_mask[y, x] == 0:
                k = x + y * ranX
                A[k, k] = 1
                A[k, k + ranX] = 0
                A[k, k - ranX] = 0
                A[k, k + 1] = 0
                A[k, k - 1] = 0

    lse=0
    flatMask = target_mask.flatten()
    for ch in range(source_image.shape[2]):
        flatTarget = target_image[minY:maxY, minX:maxX, ch].flatten()
        flatSource = source_image[minY:maxY, minX:maxX, ch].flatten()
        b = A.dot(flatSource) * 1
        b[flatMask == 0] = flatTarget[flatMask == 0]
        x = eqsol(A.tocsc(), b)
        x_c=x.copy()

        x = x.reshape((ranY, ranX))
        x[x > 255] = 255
        x[x < 0] = 0
        x = x.astype('uint8')
        target_image[minY:maxY, minX:maxX, ch] = x
        lse += np.linalg.norm((A * x_c) - b)
    return lse/im_source.shape[2],target_image


if __name__ == '__main__':
    #read source and target images
    source_path = './source1.jpg'
    target_path = './target.jpg'
    source_image = cv2.imread(source_path)
    target_image = cv2.imread(target_path)

    #align target image
    im_source, mask = align_target(source_image, target_image)

    ##poisson blend
    lse,blended_image = poisson_blend(im_source, target_image, mask)
    print("LSE: ",lse)
    plt.imshow(blended_image[:,:,::-1])
    plt.show()