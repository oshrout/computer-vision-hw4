#\ imports
import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch
import glob
import os
import shutil
from my_homography import plot_images, getPoints, computeH, imageStitching, ransacH, getPoints_SIFT, \
    crop_image, panorama_maker
from my_ar import create_ref, im2im, my_vid2vid
from frame_video_convert import image_seq_to_video, video_to_image_seq

#\ load and display the images

img1 = cv2.imread('./data/incline_L.png')
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
#img1 = np.float32(img1)

img2 = cv2.imread('./data/incline_R.png')
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
#img2 = np.float32(img2)

# display the images
plot_images([img1, img2], ['first image', 'second image'], subplot_shape=(1, 2), figsize=(20, 10))

#\ manualy pick N corrosponding point from the images and compute the Homography transformation

N = 6
[p1, p2] = getPoints(img1, img2, N)

# compute the Homography transformation
H2to1 = computeH(p1, p2)

#\ show the transformation is correct

# pick arbitrary points from first image
arbitrary_p1 = np.array([[530, 450, 800], [460, 220, 370], [1, 1, 1]])

# compute the points on the 2nd image
p2_points = np.zeros([3, arbitrary_p1.shape[1]])
for i in range(arbitrary_p1.shape[1]):
    p2_points[:, i] = np.linalg.inv(H2to1) @ arbitrary_p1[:, i]
    p2_points[:, i] /= p2_points[-1, i]

# display the results
fig = plt.figure(figsize=(20, 10))
ax1 = fig.add_subplot(1, 2, 1)
ax1.imshow(np.uint8(img1))
plt.plot(arbitrary_p1[0, 0], arbitrary_p1[1, 0], 'or')
plt.plot(arbitrary_p1[0, 1], arbitrary_p1[1, 1], 'om')
plt.plot(arbitrary_p1[0, 2], arbitrary_p1[1, 2], 'og')
ax1.set_title("first image")
ax1.set_axis_off()

ax2 = fig.add_subplot(1, 2, 2)
ax2.imshow(np.uint8(img2))
plt.plot(p2_points[0, 0], p2_points[1, 0], 'or')
plt.plot(p2_points[0, 1], p2_points[1, 1], 'om')
plt.plot(p2_points[0, 2], p2_points[1, 2], 'og')
ax2.set_title("second image")
ax2.set_axis_off()

#\ make panorama image from the 1st and warp images

panoImg = imageStitching(img1, img2, H2to1)

# display the result
plot_images([panoImg], ['Panorama image'])


#\ create panorama using SIFT detector

# get p1 and p2 using sift detector
p1_sift, p2_sift = getPoints_SIFT(img1, img2)

# get the Homography transformation
H2to1_sift = computeH(p1_sift[:, :50], p2_sift[:, :50])

# create panorama image
panoImg_sift = imageStitching(img1, img2, H2to1_sift)

# display the result
plot_images([panoImg_sift], ['Panorama image using SIFT'])


#\ load beach and Sintra images

scale_percent = 20  # percent of original size

# initialization
img_beach_list = []
img_sintra_list = []

for i in range(5):
    # load beach images
    img = cv2.imread('./data/beach' + str(i + 1) + '.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    # resize image
    resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    # add to list
    img_beach_list.append(resized)

    # load Sintra images
    img = cv2.imread('./data/sintra' + str(i + 1) + '.JPG')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    # resize image
    resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    # add to list
    img_sintra_list.append(resized)

#\ Compare SIFT and Manual image selection

# make panorama from beach images using manual and SIFT descriptors
panoImg_manual = panorama_maker(img_beach_list, SIFT=False, RANSAC=False, Blending=False)
panoImg_sift = panorama_maker(img_beach_list, SIFT=True, RANSAC=False, Blending=False)
# display the results
plot_images([panoImg_manual, panoImg_sift],
            ['beach panorama image using manual descriptor', 'beach panorama image using SIFT descriptor'],
            subplot_shape=(1, 2), figsize=(20, 10))
# save the images
cv2.imwrite('./output/beach_panorama1.jpg', cv2.cvtColor(panoImg_manual, cv2.COLOR_RGB2BGR))
cv2.imwrite('./output/beach_panorama2.jpg', cv2.cvtColor(panoImg_sift, cv2.COLOR_RGB2BGR))


# make panorama from sintra images using manual and SIFT descriptors
panoImg_manual = panorama_maker(img_sintra_list, SIFT=False, RANSAC=False, Blending=False)
panoImg_sift = panorama_maker(img_sintra_list, SIFT=True, RANSAC=False, Blending=False)
# display the results
plot_images([panoImg_manual, panoImg_sift],
            ['sintra panorama image using manual descriptor', 'sintra panorama image using SIFT descriptor'],
            subplot_shape=(1, 2), figsize=(20, 10))
# save the images
# cv2.imwrite('./output/sintra_panorama1.jpg', cv2.cvtColor(panoImg_manual, cv2.COLOR_RGB2BGR))
# cv2.imwrite('./output/sintra_panorama2.jpg', cv2.cvtColor(panoImg_sift, cv2.COLOR_RGB2BGR))


#\ implement RANSAC algorithm to improve results

# make panorama from beach images using manual and SIFT descriptors
panoImg_manual = panorama_maker(img_beach_list, SIFT=False, RANSAC=True, Blending=False)
panoImg_sift = panorama_maker(img_beach_list, SIFT=True, RANSAC=True, Blending=False)
# display the results
plot_images([panoImg_manual, panoImg_sift],
            ['beach panorama image using manual and RANSAC', 'beach panorama image using SIFT and RANSAC'],
            subplot_shape=(1, 2), figsize=(20, 10))
# save the images
cv2.imwrite('./output/beach_panorama3.jpg', cv2.cvtColor(panoImg_manual, cv2.COLOR_RGB2BGR))
cv2.imwrite('./output/beach_panorama4.jpg', cv2.cvtColor(panoImg_sift, cv2.COLOR_RGB2BGR))


# make panorama from sintra images using manual and SIFT descriptors
panoImg_manual = panorama_maker(img_sintra_list, SIFT=False, RANSAC=True, Blending=False)
panoImg_sift = panorama_maker(img_sintra_list, SIFT=True, RANSAC=True, Blending=False)
# display the results
plot_images([panoImg_manual, panoImg_sift],
            ['sintra panorama image using manual and RANSAC', 'sintra panorama image using SIFT and RANSAC'],
            subplot_shape=(1, 2), figsize=(20, 10))
# save the images
# cv2.imwrite('./output/sintra_panorama3.jpg', cv2.cvtColor(panoImg_manual, cv2.COLOR_RGB2BGR))
# cv2.imwrite('./output/sintra_panorama4.jpg', cv2.cvtColor(panoImg_sift, cv2.COLOR_RGB2BGR))

#\ use blender function to improve stiching

# make panorama from beach images using manual and SIFT descriptors
panoImg_manual = panorama_maker(img_beach_list, SIFT=False, RANSAC=True, Blending=True)
panoImg_sift = panorama_maker(img_beach_list, SIFT=True, RANSAC=True, Blending=True)
# display the results
plot_images([panoImg_manual, panoImg_sift],
            ['beach panorama image using manual, RANSAC and Blending',
             'beach panorama image using SIFT, RANSAC and Blending'],
            subplot_shape=(1, 2), figsize=(20, 10))
# save the images
cv2.imwrite('./output/beach_panorama5.jpg', cv2.cvtColor(panoImg_manual, cv2.COLOR_RGB2BGR))
cv2.imwrite('./output/beach_panorama6.jpg', cv2.cvtColor(panoImg_sift, cv2.COLOR_RGB2BGR))


# make panorama from sintra images using manual and SIFT descriptors
panoImg_manual = panorama_maker(img_sintra_list, SIFT=False, RANSAC=True, Blending=True)
panoImg_sift = panorama_maker(img_sintra_list, SIFT=True, RANSAC=True, Blending=True)
# display the results
plot_images([panoImg_manual, panoImg_sift],
            ['sintra panorama image using manual, RANSAC and Blending',
             'sintra panorama image using SIFT, RANSAC and Blending'],
            subplot_shape=(1, 2), figsize=(20, 10))
# save the images
# cv2.imwrite('./output/sintra_panorama5.jpg', cv2.cvtColor(panoImg_manual, cv2.COLOR_RGB2BGR))
# cv2.imwrite('./output/sintra_panorama6.jpg', cv2.cvtColor(panoImg_sift, cv2.COLOR_RGB2BGR))

#\ create our panorama

scale_percent = 20#50
backyard_list = []

# read the images and put in a list
for i in range(4):
    # load beach images
    img = cv2.imread('./my_data/backyard' + str(i + 1) + '.jpeg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    # resize image
    resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    # add to list
    backyard_list.append(resized)

# reverse the list
backyard_list.reverse()

# create panorama
panoImg_backyard = panorama_maker(backyard_list)

# display the result
plot_images([panoImg_backyard], ['backyard panorama image'])

# save the image
#cv2.imwrite('./output/backyard_panorama.jpg', cv2.cvtColor(panoImg_backyard, cv2.COLOR_RGB2BGR))


#\ Affine transformation

scale_percent = 20  # percent of original size

# load images
img1 = cv2.imread('./my_data/backyard2.jpeg')
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
width = int(img1.shape[1] * scale_percent / 100)
height = int(img1.shape[0] * scale_percent / 100)
dim = (width, height)
# resize image
img1 = cv2.resize(img1, dim, interpolation=cv2.INTER_AREA)


img2 = cv2.imread('./my_data/backyard3.jpeg')
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
width = int(img2.shape[1] * scale_percent / 100)
height = int(img2.shape[0] * scale_percent / 100)
dim = (width, height)
# resize image
img2 = cv2.resize(img2, dim, interpolation=cv2.INTER_AREA)


Affine_case = [True, False]
panoImg_list = []

for i in range(2):
    # get points using SIFT
    p1, p2 = getPoints_SIFT(img1, img2)

    # compute Affine transform
    #H2to1 = AffineH(p1[:, :4], p2[:, :4])
    H2to1 = ransacH(p1[:, :50], p2[:, :50], tol=5, Affine=Affine_case[i])

    # stitch the images
    panoImg = imageStitching(img1, img2, H2to1, False)

    # crop the unnecessary black background
    panoImg_crop = crop_image(panoImg)
    panoImg_list.append(panoImg_crop)

# plot the results
plot_images(panoImg_list,
            ['panorama using Affine transformation', 'panorama using projective transformation'],
            subplot_shape=(1, 2), figsize=(20, 10))

# load images
img1 = cv2.imread('./data/incline_L.png')
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)

img2 = cv2.imread('./data/incline_R.png')
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)


Affine_case = [True, False]
panoImg_list = []

for i in range(2):
    # get points using SIFT
    p1, p2 = getPoints_SIFT(img1, img2)

    # compute Affine transform
    #H2to1 = AffineH(p1[:, :4], p2[:, :4])
    H2to1 = ransacH(p1[:, :50], p2[:, :50], tol=5, Affine=Affine_case[i])

    # stitch the images
    panoImg = imageStitching(img1, img2, H2to1, False)

    # crop the unnecessary black background
    panoImg_crop = crop_image(panoImg)
    panoImg_list.append(panoImg_crop)

# plot the results
plot_images(panoImg_list,
            ['panorama using Affine transformation', 'panorama using projective transformation'],
            subplot_shape=(1, 2), figsize=(20, 10))


#\ create reference image

# im_path = './my_data/book1.jpeg'
# im_path = './my_data/book_test.jpeg'
im_path = './my_data/book2.jpeg'
ref_image = create_ref(im_path)

plot_images([ref_image], ['ref image'], figsize=(10, 10))

# save the image
# cv2.imwrite('../output/ref_book.jpg', cv2.cvtColor(ref_image, cv2.COLOR_RGB2BGR))


#\ implant image in image

# read the reference image
ref_image = cv2.imread('../output/ref_book_backup.jpg')
ref_image = cv2.cvtColor(ref_image, cv2.COLOR_BGR2RGB)

# read new book image
new_book_img = cv2.imread('./data/pf_scan_scaled_croped.jpg')
new_book_img = cv2.cvtColor(new_book_img, cv2.COLOR_BGR2RGB)
# resize the image to the reference image size
new_book_img = cv2.resize(new_book_img, (ref_image.shape[1], ref_image.shape[0]), interpolation=cv2.INTER_AREA)

for i in range(3):
    # read new scene image
    new_scene_img = cv2.imread('./my_data/book' + str(i + 1) + '.jpeg')
    new_scene_img = cv2.cvtColor(new_scene_img, cv2.COLOR_BGR2RGB)

    # implant image inside other image
    result1 = im2im(ref_image, new_scene_img, new_book_img, detector='SIFT')
    # cv2.imwrite('../output/im2im' + str(i + 1) + '.jpg', cv2.cvtColor(result1, cv2.COLOR_RGB2BGR))
    result2 = im2im(ref_image, new_scene_img, new_book_img, detector='ORB')

    plot_images([new_scene_img, result1, result2], ['new_scene_img', 'result - SIFT', 'result - ORB'],
                subplot_shape=(1, 3), figsize=(30, 10))


#\ convert the videos into frames

# convert main video to images
vid_path = "./my_data/my_video.mp4"
output_path = "../output/my_video_images"
video_to_image_seq(vid_path, output_path)

# convert sub video to images
vid_path = "./my_data/Mirror_Mirror_vid.mp4"
output_path = "../output/sub_video_images"
video_to_image_seq(vid_path, output_path)

#\ video preparation

# read the reference image
ref_image = cv2.imread('../output/ref_book.jpg')
ref_image = cv2.cvtColor(ref_image, cv2.COLOR_BGR2RGB)
# define dimensions of ref frame
width = ref_image.shape[1]
height = ref_image.shape[0]
ref_dim = (width, height)

# gather the images of my video into a list
my_vid_list = []
images_path = "../output/my_video_images"
for filename in glob.glob(os.path.join(images_path, '*.jpg')):
  image = cv2.imread(filename)
  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  my_vid_list.append(image)

# gather the images of sub video into a list
sub_vid_list = []
images_path = "../output/sub_video_images"
for filename in glob.glob(os.path.join(images_path, '*.jpg')):
  image = cv2.imread(filename)
  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  # crop the image
  image = crop_image(image)
  # resize the frames to ref image size
  image = cv2.resize(image, ref_dim, interpolation=cv2.INTER_AREA)
  sub_vid_list.append(image)

#\ create video

result_vid_list = my_vid2vid(my_vid_list, sub_vid_list, ref_image)

#\ show some frames

frame1 = 20
frame2 = 150
frame3 = 230

plot_images([result_vid_list[frame1], result_vid_list[frame2], result_vid_list[frame3]],
            ['frame %d' % frame1, 'frame %d' % frame2, 'frame %d' % frame3],
            subplot_shape=(1, 3), figsize=(30, 10))


#\ apply yolov3 to the video frames

from yolov3.detect import detect_yolov3

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

img_src = '../output/video/'
output = '../output/video_yolov3/'

detect_yolov3(source=img_src, output=output, device=device)


#\ show yolov3 frames

frame1 = 20
frame2 = 150
frame3 = 230

yolov3_output_path = '../output/video_yolov3/'
yolov3_frame1 = cv2.cvtColor(cv2.imread(yolov3_output_path + '00' + str(frame1) + '.jpg'), cv2.COLOR_BGR2RGB)
yolov3_frame2 = cv2.cvtColor(cv2.imread(yolov3_output_path + '0' + str(frame2) + '.jpg'), cv2.COLOR_BGR2RGB)
yolov3_frame3 = cv2.cvtColor(cv2.imread(yolov3_output_path + '0' + str(frame3) + '.jpg'), cv2.COLOR_BGR2RGB)

plot_images([yolov3_frame1, yolov3_frame2, yolov3_frame3],
            ['yolov3 frame %d' % frame1, 'yolov3 frame %d' % frame2, 'yolov3 frame %d' % frame3],
            subplot_shape=(1, 3), figsize=(30, 20))


#\ make frames from van gogh video after applying CycleGAN

vid_path = "./my_data/vid2vid_van.mp4"
output_path = "../output/video_van_gogh"

video_to_image_seq(vid_path, output_path)


#\ apply yolov4 to the video frames

from yolov4.demo import detect_yolov4

cfgfile = './yolov4/cfg/yolov4.cfg'
weightfile = './yolov4/weights/yolov4.weights'
imgs_path = '../output/video_yolov4_pre/'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

detect_yolov4(cfgfile, weightfile, imgs_path, device)


#\ arrange the frames into final video folder

images_path1 = "../output/video_van_gogh"
images_path2 = "../output/video_yolov4"
output_path = "../output/video_final"


for filename in glob.iglob(os.path.join(images_path1, "*.jpg")):
    shutil.copy(filename, output_path)

for filename in glob.iglob(os.path.join(images_path2, "*.jpg")):
    shutil.copy(filename, output_path)


#\ create final video

images_path = "../output/video_final"
output_path = "../output/vid2vid.mp4"

image_seq_to_video(images_path, output_path=output_path, fps=30.0)

#\ show some frames from final video

frame1 = 20
frame2 = 150
frame3 = 230

video_path = '../output/video_final/'
video_frame1 = cv2.cvtColor(cv2.imread(video_path + '00' + str(frame1) + '.jpg'), cv2.COLOR_BGR2RGB)
video_frame2 = cv2.cvtColor(cv2.imread(video_path + '0' + str(frame2) + '.jpg'), cv2.COLOR_BGR2RGB)
video_frame3 = cv2.cvtColor(cv2.imread(video_path + '0' + str(frame3) + '.jpg'), cv2.COLOR_BGR2RGB)

plot_images([video_frame1, video_frame2, video_frame3],
            ['final video frame %d' % frame1, 'final video frame %d' % frame2, 'final video frame %d' % frame3],
            subplot_shape=(1, 3), figsize=(30, 20))

