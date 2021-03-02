import numpy as np
import matplotlib.pyplot as plt
import cv2
import scipy
from matplotlib import pyplot as plt

# Add imports if needed:
from scipy.interpolate import interp2d


# end imports

# Add extra functions here:


def plot_images(image_list, title_list, subplot_shape=(1, 1), axis='off', fontsize=15, figsize=(10, 10), cmap=['gray']):
    plt.figure(figsize=figsize)
    for ii, im in enumerate(image_list):
        c_title = title_list[ii]
        if len(cmap) > 1:
            c_cmap = cmap[ii]
        else:
            c_cmap = cmap[0]
        plt.subplot(subplot_shape[0], subplot_shape[1], ii + 1)
        plt.imshow(np.uint8(im), cmap=c_cmap)
        plt.title(c_title, fontsize=fontsize)
        plt.axis(axis)


def perspectiveT(img_dim, M):
    coord = img_dim.reshape(-1, 2).T
    coord = np.vstack((coord, np.ones(coord.shape[1])))
    out_coord = M.dot(coord)
    out_coord /= out_coord[-1]
    out_coord = out_coord[:2]
    out_coord = out_coord.T.reshape(-1, 1, 2)

    return out_coord


def crop_image(final_img):
    # Crop black edge
    final_gray = cv2.cvtColor(final_img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(final_gray, 1, 255, cv2.THRESH_BINARY)
    dino, contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    max_area = 0
    best_rect = (0, 0, 0, 0)

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        deltaHeight = h - y
        deltaWidth = w - x

        area = deltaHeight * deltaWidth

        if area > max_area and deltaHeight > 0 and deltaWidth > 0:
            max_area = area
            best_rect = (x, y, w, h)

    if max_area > 0:
        final_img_crop = final_img[best_rect[1]:best_rect[1] + best_rect[3],
                         best_rect[0]:best_rect[0] + best_rect[2]]
    else:
        final_img_crop = final_img
        print("max_area is zero")

    return final_img_crop


def panorama_maker(img_list, SIFT=True, RANSAC=True, Blending=True):

    images_num = len(img_list)

    for j in range(images_num, 1, -1):
        panoImg_list = []
        for i in range(j - 1):
            img1 = img_list[i]
            img2 = img_list[(i + 1)]

            if SIFT:
                # get points using SIFT
                p1, p2 = getPoints_SIFT(img1, img2)
            else:
                # get points manually
                p1, p2 = getPoints(img1, img2, 5)

            # compute Homography transform
            if RANSAC:
                H2to1 = ransacH(p1[:, :50], p2[:, :50])
            else:
                H2to1 = computeH(p1[:, :6], p2[:, :6])

            # stitch the images
            if Blending:
                panoImg = imageStitching(img1, img2, H2to1, True)
            else:
                panoImg = imageStitching(img1, img2, H2to1, False)

            # crop the unnecessary black background
            panoImg_crop = crop_image(panoImg)

            # add the image to the list
            panoImg_list.append(panoImg_crop)

        # insert the pano list to image list for next iteration
        img_list = panoImg_list

    return panoImg_crop


def AffineH(p1, p2):
    assert (p1.shape[1] == p2.shape[1])
    assert (p1.shape[0] == 2)

    # initialization
    N = p1.shape[1]  # num of correspondences
    A = np.zeros([2 * N, 6])
    b = np.zeros([2 * N, 1])

    # get the x,y coordinates as vectors of p1 and p2
    x = p1[0, :].reshape(-1, 1)  # x of p1
    y = p1[1, :].reshape(-1, 1)  # y of p1
    u = p2[0, :].reshape(-1, 1)  # x of p2
    v = p2[1, :].reshape(-1, 1)  # y of p2

    # define vectors of zeros and ones
    ones_vec = np.ones(N).reshape(-1, 1)
    zeros_vec = np.zeros(N).reshape(-1, 1)

    # define A as we saw in the tutorial
    A[::2] = np.concatenate((u, v, ones_vec, zeros_vec, zeros_vec, zeros_vec), axis=1)
    A[1::2] = np.concatenate((zeros_vec, zeros_vec, zeros_vec, u, v, ones_vec), axis=1)

    # define b as we saw in the tutorial
    b[::2] = x
    b[1::2] = y

    # compute h using Least-squares solution
    h = np.linalg.pinv(A.T @ A) @ np.dot(A.T, b)

    # insert h to H2to1 and make it 3x3 matrix
    H2to1 = h.reshape((2, 3))
    H2to1 = np.vstack((H2to1, np.array([0, 0, 1])))

    return H2to1

# Extra functions end


# HW functions:
def getPoints(im1, im2, N):
    # show the 2 images in 1 figure
    fig = plt.figure(figsize=(20, 10))

    ax1 = fig.add_subplot(1, 2, 1)
    ax1.imshow(np.uint8(im1))
    ax1.set_title("first image")
    ax1.set_axis_off()

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.imshow(np.uint8(im2))
    ax2.set_title("second image")
    ax2.set_axis_off()

    # initialization
    p1 = np.zeros([2, N])
    p2 = np.zeros([2, N])

    point_list = plt.ginput(2 * N, -1)  # get N points from each img

    # run over N correspondences and put in matrix
    for n in range(N):
        p1[0, n] = point_list[2 * n][0]  # even x goes to p1
        p1[1, n] = point_list[2 * n][1]  # even y goes to p1

        p2[0, n] = point_list[2 * n + 1][0]  # odd x goes to p2
        p2[1, n] = point_list[2 * n + 1][1]  # odd y goes to p2

    plt.show()
    return p1, p2


def computeH(p1, p2):
    assert (p1.shape[1] == p2.shape[1])
    assert (p1.shape[0] == 2)

    N = p1.shape[1]  # num of correspondences
    A = np.zeros([2 * N, 9])

    # get the x,y coordinates as vectors of p1 and p2
    x = p1[0, :].reshape(-1, 1)  # x of p1
    y = p1[1, :].reshape(-1, 1)  # y of p1
    u = p2[0, :].reshape(-1, 1)  # x of p2
    v = p2[1, :].reshape(-1, 1)  # y of p2

    # define vectors of zeros and ones
    ones_vec = np.ones(N).reshape(-1, 1)
    zeros_vec = np.zeros(N).reshape(-1, 1)

    # define A as in the theoretical part
    A[::2] = np.concatenate((u, v, ones_vec, zeros_vec, zeros_vec, zeros_vec, -(u * x), -(v * x), -x), axis=1)
    A[1::2] = np.concatenate((zeros_vec, zeros_vec, zeros_vec, u, v, ones_vec, -(u * y), -(v * y), -y), axis=1)

    # compute the SVD decomposition
    (U, D, V) = np.linalg.svd(A)
    h = V.T[:, -1]  # take the eigenvector with the lowest eigenvalue
    h /= h[-1]  # divide by the last value

    # H2to1 is the found eigenvector
    H2to1 = h.reshape(3, 3)

    return H2to1


def warpH(im1, H, out_size):
    # move to LAB space
    im1 = cv2.cvtColor(im1, cv2.COLOR_RGB2LAB)

    # extract im1, out_size shape
    im1_H, im1_W, im1_c = im1.shape
    new_imH, new_imW = out_size

    # initialization
    warp_im1 = np.zeros([out_size[0], out_size[1], im1_c])
    warp_im1[:, :, 1:] -= 128  # [0, -128, -128] in Lab is converting to [0, 0, 0] in RGB
    x_img1, y_img1 = np.arange(im1.shape[0]), np.arange(im1.shape[1])

    f = []
    fill_val = [0, -128, -128]
    for c in range(im1_c):
        z = im1[:, :, c]  # take the intensity of the image channel
        # interpolate the img1 coordinates
        f.insert(c, interp2d(y_img1, x_img1, z, kind='linear', fill_value=fill_val[c]))  # cubic

    for x in range(new_imW):
        for y in range(new_imH):
            source_coord = np.dot(np.linalg.inv(H), [x, y, 1])
            x1, y1, _ = (source_coord / source_coord[2])
            if 0 <= x1 < im1_W:
                if 0 <= y1 < im1_H:
                    for c in range(im1_c):
                        ff = f[c]
                        warp_im1[y, x, c] = ff(x1, y1)

    warp_im1 = cv2.cvtColor(np.uint8(warp_im1), cv2.COLOR_LAB2RGB)

    return np.uint8(warp_im1)


def imageStitching(img1, img2, M, blending=False):
    # Get width and height of input images
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    # Get the canvas dimensions
    img1_dims_temp = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
    img2_dims_temp = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)

    img2_dims = perspectiveT(img2_dims_temp, M)
    img1_dims = perspectiveT(img1_dims_temp, np.linalg.inv(M))

    # Resulting dimensions
    result_dims1 = np.concatenate((img1_dims, img2_dims_temp), axis=0)
    result_dims2 = np.concatenate((img1_dims_temp, img2_dims), axis=0)

    # Getting images together
    # Calculate dimensions of match points
    [x_min1, y_min1] = np.int32(result_dims1.min(axis=0).ravel() - 0.5)
    [x_max1, y_max1] = np.int32(result_dims1.max(axis=0).ravel() + 0.5)

    # Getting images together
    # Calculate dimensions of match points
    [x_min2, y_min2] = np.int32(result_dims2.min(axis=0).ravel() - 0.5)
    [x_max2, y_max2] = np.int32(result_dims2.max(axis=0).ravel() + 0.5)

    Area1 = (x_max1 - x_min1) * (y_max1 - y_min1)
    Area2 = (x_max2 - x_min2) * (y_max2 - y_min2)

    if Area2 <= Area1:
        print('case Area2 <= Area1:')
        # Create output array after affine transformation
        transform_dist = [-x_min2, -y_min2]
        transform_array = np.array([[1, 0, transform_dist[0]],
                                    [0, 1, transform_dist[1]],
                                    [0, 0, 1]])

        # Warp images to get the resulting image
        result_img = warpH(img2, transform_array.dot(M), [y_max2 - y_min2, x_max2 - x_min2])

        # Create masked composite
        (_, mask) = cv2.threshold(cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY), 0, 255, cv2.THRESH_BINARY_INV)
        mask = mask / 255  # set mask to 0,1

        mask_3ch = np.repeat(mask[:, :, np.newaxis], 3, axis=2)

        upper_line = result_img[transform_dist[1], transform_dist[0]:w1 + transform_dist[0]]
        left_line = result_img[transform_dist[1]:h1 + transform_dist[1], transform_dist[0]]

        if blending:
            MASK = np.ones_like(result_img, np.float32)
            MASK[transform_dist[1]:h1 + transform_dist[1], transform_dist[0]:w1 + transform_dist[0]] = mask_3ch
            w = 10#35(sintra)#10(beach)
            offset = 5#150(sintra)#20(beach)
            v_dec = np.linspace(1, 0, 2 * w)
            if np.sum(np.where(upper_line > 0, 1, 0)) > np.sum(np.where(left_line > 0, 1, 0)):
                print('case upper_line')
                MASK[transform_dist[1] + offset - w:transform_dist[1] + offset + w,
                     transform_dist[0]:w1 + transform_dist[0]] += \
                    np.tile(np.reshape(v_dec, [-1, 1, 1]), [1, w1, 3])
                MASK = np.where(MASK > 1, 1, MASK)
                MASK[transform_dist[1] - w:transform_dist[1] + offset - w,
                     transform_dist[0]:w1 + transform_dist[0]] = np.ones([offset, w1, 3])
            else:
                print('case left_line')
                MASK[transform_dist[1]:h1 + transform_dist[1],
                     transform_dist[0] + offset - w:transform_dist[0] + offset + w] += \
                    np.tile(np.reshape(v_dec, [1, -1, 1]), [h1, 1, 3])
                MASK = np.where(MASK > 1, 1, MASK)
                MASK[transform_dist[1]:h1 + transform_dist[1],
                     transform_dist[0] - w:transform_dist[0] + offset - w] = np.ones([h1, offset, 3])

            img1_blend = np.zeros_like(result_img)
            img1_blend[transform_dist[1]:h1 + transform_dist[1], transform_dist[0]:w1 + transform_dist[0]] = img1
            result_img = blender(img1_blend, result_img, MASK)
        else:
            result_img[transform_dist[1]:h1 + transform_dist[1], transform_dist[0]:w1 + transform_dist[0]] *= np.uint8(
                mask_3ch)
            result_img[transform_dist[1]:h1 + transform_dist[1], transform_dist[0]:w1 + transform_dist[0]] = \
                cv2.add(result_img[transform_dist[1]:h1 + transform_dist[1], transform_dist[0]:w1 + transform_dist[0]],
                        img1)

    else:
        print('case Area2 > Area1:')
        # Create output array after affine transformation
        transform_dist = [-x_min1, -y_min1]
        transform_array = np.array([[1, 0, transform_dist[0]],
                                    [0, 1, transform_dist[1]],
                                    [0, 0, 1]])

        # Warp images to get the resulting image
        result_img = warpH(img1, transform_array.dot(np.linalg.inv(M)), [y_max1 - y_min1, x_max1 - x_min1])

        # Create masked composite
        (_, mask) = cv2.threshold(cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY), 0, 255, cv2.THRESH_BINARY_INV)
        mask = mask / 255  # set mask to 0,1

        mask_3ch = np.repeat(mask[:, :, np.newaxis], 3, axis=2)

        bottom_line = result_img[h2 + transform_dist[1] - 1, transform_dist[0]:w2 + transform_dist[0]]
        right_line = result_img[transform_dist[1]:h2 + transform_dist[1], w2 + transform_dist[0] - 1]

        if blending:
            MASK = np.ones_like(result_img, np.float32)
            MASK[transform_dist[1]:h2 + transform_dist[1], transform_dist[0]:w2 + transform_dist[0]] = mask_3ch
            w = 10#35(sintra)#10(beach)
            offset = 5#150(sintra)#20(beach)
            v_dec = np.linspace(0, 1, 2 * w)
            if np.sum(np.where(bottom_line > 0, 1, 0)) > np.sum(np.where(right_line > 0, 1, 0)):
                print('case bottom_line')
                MASK[h2 + transform_dist[1] - offset - w: h2 + transform_dist[1] - offset + w,
                     transform_dist[0]:w2 + transform_dist[0]] += \
                    np.tile(np.reshape(v_dec, [-1, 1, 1]), [1, w2, 3])
                MASK = np.where(MASK > 1, 1, MASK)
                MASK[h2 + transform_dist[1] - offset + w:h2 + transform_dist[1] + w,
                     transform_dist[0]:w2 + transform_dist[0]] = np.ones([offset, w2, 3])
            else:
                print('case right_line')
                MASK[transform_dist[1]:h2 + transform_dist[1],
                     w2 + transform_dist[0] - offset - w:w2 + transform_dist[0] - offset + w] += \
                    np.tile(np.reshape(v_dec, [1, -1, 1]), [h2, 1, 3])
                MASK = np.where(MASK > 1, 1, MASK)
                MASK[transform_dist[1]:h2 + transform_dist[1],
                     w2 + transform_dist[0] - offset + w:w2 + transform_dist[0] + w] = np.ones([h2, offset, 3])

            img2_blend = np.zeros_like(result_img)
            img2_blend[transform_dist[1]:h2 + transform_dist[1], transform_dist[0]:w2 + transform_dist[0]] = img2

            result_img = blender(img2_blend, result_img, MASK)
        else:
            result_img[transform_dist[1]:h2 + transform_dist[1], transform_dist[0]:w2 + transform_dist[0]] *= np.uint8(
                mask_3ch)
            result_img[transform_dist[1]:h2 + transform_dist[1], transform_dist[0]:w2 + transform_dist[0]] = \
                cv2.add(result_img[transform_dist[1]:h2 + transform_dist[1], transform_dist[0]:w2 + transform_dist[0]],
                        img2)

    return result_img


def blender(img1, warp_img2, MASK):
    # make sure the images are float32
    img1 = np.float32(img1)
    warp_img2 = np.float32(warp_img2)

    # alpha blend the images
    blend_img = np.uint8(img1 * (1 - MASK) + warp_img2 * MASK)

    return blend_img


def ransacH(p1, p2, nIter=500, tol=2, Affine=False):
    assert (p1.shape[1] == p2.shape[1])
    assert (p1.shape[0] == 2)

    # initialization
    N = p1.shape[1]
    bestH = np.zeros([3, 3])
    max_inliers = 0

    for iter in range(nIter):
        # pick 4 random correspondences to compute H
        ind = np.random.choice(N, 4, replace=False)
        p1_sample, p2_sample = p1[:, ind], p2[:, ind]

        if Affine:
            H2to1 = AffineH(p1_sample, p2_sample)
        else:
            # compute H based on the 4 picks
            H2to1 = computeH(p1_sample, p2_sample)

        # compute the homograph points using H2to1
        p2_ransac = np.vstack((p2, np.ones(p2.shape[1])))
        p1_ransac = H2to1.dot(p2_ransac)
        p1_ransac /= p1_ransac[-1]

        # calculate the number of inliers
        num_inliers = np.sum(np.where(np.linalg.norm(p1_ransac[:2] - p1, axis=0) > tol, 0, 1))

        if num_inliers > max_inliers:
            # update the max inliers seen
            max_inliers = num_inliers
            # take H2to1 to be best H seen
            bestH = H2to1

        # compute H using only inliers
        p1_inliers = np.delete(p1, np.where(np.linalg.norm(p1_ransac[:2] - p1, axis=0) > tol), axis=1)
        p2_ransac = np.delete(p2_ransac, np.where(np.linalg.norm(p1_ransac[:2] - p1, axis=0) > tol), axis=1)
        p1_ransac = np.delete(p1_ransac, np.where(np.linalg.norm(p1_ransac[:2] - p1, axis=0) > tol), axis=1)

        if Affine:
            H2to1 = AffineH(p1_ransac[:2], p2_ransac[:2])
        else:
            # compute H based on the 4 picks
            H2to1 = computeH(p1_ransac[:2], p2_ransac[:2])

        p1_ransac = H2to1.dot(p2_ransac)
        p1_ransac /= p1_ransac[-1]

        # calculate the number of inliers
        num_inliers = np.sum(np.where(np.linalg.norm(p1_ransac[:2] - p1_inliers, axis=0) > tol, 0, 1))

        if num_inliers > max_inliers:
            # update the max inliers seen
            max_inliers = num_inliers
            # take H2to1 to be best H seen
            bestH = H2to1

    return bestH


def getPoints_SIFT(im1, im2):
    # convert the images to gray scale
    img1_gray = cv2.cvtColor(im1, cv2.COLOR_RGB2GRAY)
    img2_gray = cv2.cvtColor(im2, cv2.COLOR_RGB2GRAY)

    # create sift detector
    sift = cv2.xfeatures2d.SIFT_create()

    # get the key-points and descriptors from the images using sift
    kp1, des1 = sift.detectAndCompute(np.uint8(img1_gray), None)
    kp2, des2 = sift.detectAndCompute(np.uint8(img2_gray), None)

    # create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

    # Match descriptors
    matches = bf.match(des1, des2)

    # Sort the matches in the order of their distance
    matches = sorted(matches, key=lambda x: x.distance)

    # extract the (x,y) points from the matches
    p1 = np.float32([kp1[m.queryIdx].pt for m in matches]).T.reshape(2, -1)
    p2 = np.float32([kp2[m.trainIdx].pt for m in matches]).T.reshape(2, -1)

    return p1, p2


if __name__ == '__main__':
    print('my_homography')
    im1 = cv2.imread('data/incline_L.png')
    im2 = cv2.imread('data/incline_R.png')
    """
    Your code here
    """
