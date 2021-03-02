import numpy as np
import matplotlib.pyplot as plt
import cv2
import scipy
from matplotlib import pyplot as plt
import os

# Add imports if needed:
from my_homography import getPoints, ransacH, getPoints_SIFT
from frame_video_convert import image_seq_to_video
# end imports

# Add functions here:


def getPoints_ORB(img1, img2):
    # convert the images to gray scale
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)

    # create sift detector
    orb = cv2.ORB_create()

    # get the key-points and descriptors from the images using sift
    kp1, des1 = orb.detectAndCompute(np.uint8(img1_gray), None)
    kp2, des2 = orb.detectAndCompute(np.uint8(img2_gray), None)

    # create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Match descriptors
    matches = bf.match(des1, des2)

    # Sort the matches in the order of their distance
    matches = sorted(matches, key=lambda x: x.distance)

    # extract the (x,y) points from the matches
    p1 = np.float32([kp1[m.queryIdx].pt for m in matches]).T.reshape(2, -1)
    p2 = np.float32([kp2[m.trainIdx].pt for m in matches]).T.reshape(2, -1)

    return p1, p2

# Functions end

# HW functions:


def create_ref(im_path):
    # read the image and convert to RGB
    book_img = cv2.imread(im_path)
    book_img = cv2.cvtColor(book_img, cv2.COLOR_BGR2RGB)

    # set out size
    #out_size = [350, 440]
    out_size = [440, 350]
    offset = 30

    # Create a black image
    synthetic_img = np.zeros((out_size[1] + 2 * offset, out_size[0] + 2 * offset, 3), np.uint8)
    # draw a rectangle in the out size to ease the corner detecting
    synthetic_img = cv2.rectangle(synthetic_img, (offset, offset), (out_size[0] + offset, out_size[1] + offset), (255, 255, 255), 1)

    # use the functions in order to warp the image
    N = 4
    p1, p2 = getPoints(book_img, synthetic_img, N)
    best_H1to2 = ransacH(p2, p1)
    #warp_book = warpH(book_img, best_H1to2, out_size)
    warp_book = cv2.warpPerspective(book_img, best_H1to2, (out_size[0] + offset, out_size[1] + offset))

    ref_image = warp_book[offset:, offset:]
    return ref_image


def im2im(ref_image, new_scene_img, new_image, detector='SIFT'):
    # pick detector
    if detector == 'SIFT':
        # get points using SIFT
        p1_sift, p2_sift = getPoints_SIFT(new_scene_img, ref_image)
    elif detector == 'ORB':
        p1_sift, p2_sift = getPoints_ORB(new_scene_img, ref_image)
    else:
        print("No such detector found")
        return

    # compute Homography transform
    #best_H2to1 = ransacH(p1_sift[:, :50], p2_sift[:, :50])
    src_pts = p2_sift[:, :50].T.reshape(-1, 1, 2)
    dst_pts = p1_sift[:, :50].T.reshape(-1, 1, 2)
    best_H2to1, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 2.0)

    # define out size
    out_size = [new_scene_img.shape[0], new_scene_img.shape[1]]

    # compute warp image
    new_image = np.uint8(np.float32(new_image) + np.ones_like(new_image, np.float32))
    warp_img2 = cv2.warpPerspective(new_image, best_H2to1, (out_size[1], out_size[0]))

    # Create inverted mask for warped image
    (_, mask) = cv2.threshold(cv2.cvtColor(warp_img2, cv2.COLOR_BGR2GRAY), 0, 255, cv2.THRESH_BINARY_INV)
    mask = mask / 255  # set mask to 0,1
    # make the mask 3 channels
    mask_3ch = np.repeat(mask[:, :, np.newaxis], 3, axis=2)

    result = new_scene_img * np.uint8(mask_3ch)
    result = cv2.add(result, warp_img2)

    return result


def my_vid2vid(my_vid_list, sub_vid_list, ref_image):
    # initialization
    result_vid_list = []
    images_path = "../output/video"
    output_path = "../output/vid2vid.mp4"

    for frame_num in range(np.min([len(my_vid_list), len(sub_vid_list)])):
        result_frame = im2im(ref_image, my_vid_list[frame_num], sub_vid_list[frame_num], detector='SIFT')
        # write the result frame in images_path folder
        fname = str(frame_num).zfill(4)
        cv2.imwrite(os.path.join(images_path, fname + ".jpg"), cv2.cvtColor(result_frame, cv2.COLOR_RGB2BGR))
        # put the frame in list
        result_vid_list.append(result_frame)

    # convert the frames into a video
    image_seq_to_video(images_path, output_path=output_path, fps=29.0)

    return result_vid_list

if __name__ == '__main__':
    print('my_ar')