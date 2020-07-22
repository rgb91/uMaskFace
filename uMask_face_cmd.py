"""
Created by Sanjay at 7/21/2020

Feature: Enter feature name here
Enter feature description here
"""
import os
import time

import dlib
import numpy as np
from facemorpher import blender
from facemorpher import plotter
from facemorpher import warper

from env import PROJECT_HOME
from utils import *


def average_face(imgpaths, dest_filename=None, width=500, height=500, background='black',
                 blur_edges=False, out_filename='result.png'):
    size = (height, width)

    images = []
    point_set = []
    for path in imgpaths:
        img, points = load_image_points(path, size)
        if img is not None:
            images.append(img)
            point_set.append(points)

    if len(images) == 0:
        raise FileNotFoundError('Could not find any valid images. Supported formats are .jpg, .png, .jpeg')

    if dest_filename is not None:
        dest_img, dest_points = load_image_points(dest_filename, size)
        if dest_img is None or dest_points is None:
            raise Exception('No face or detected face points in dest img: ' + dest_filename)
    else:
        dest_img = np.zeros(images[0].shape, np.uint8)
        dest_points = locator.average_points(point_set)

    num_images = len(images)
    result_images = np.zeros(images[0].shape, np.float32)
    for i in range(num_images):
        result_images += warper.warp_image(images[i], point_set[i],
                                           dest_points, size, np.float32)

    result_image = np.uint8(result_images / num_images)
    face_indexes = np.nonzero(result_image)
    dest_img[face_indexes] = result_image[face_indexes]

    mask = blender.mask_from_points(size, dest_points)
    if blur_edges:
        blur_radius = 10
        mask = cv2.blur(mask, (blur_radius, blur_radius))

    if background in ('transparent', 'average'):
        dest_img = np.dstack((dest_img, mask))

        if background == 'average':
            average_background = np.uint8(locator.average_points(images))
            dest_img = blender.overlay_image(dest_img, mask, average_background)

    print('Averaged {} images'.format(num_images))
    plt = plotter.Plotter(False, num_images=1, out_filename=out_filename)
    plt.save(dest_img)


def controller(new_image_path, old_image_path, uuid, grpid):
    input_filename = os.path.basename(new_image_path)
    out_filename = str(input_filename.split('_')[0]) + '.jpg'
    out_img_path = os.path.join(OUTPUT_DIRECTORY, out_filename)
    average_face([new_image_path, old_image_path], background='average', out_filename=out_img_path)
    landmarks, msg = get_landmarks(new_image_path, detector, predictor)
    return prepare_output(out_img_path, landmarks, uuid, grpid, msg)


OUTPUT_DIRECTORY = os.path.join(PROJECT_HOME, 'outputs')
ALLOWED_EXTENSIONS = {'jpg', 'jpeg'}
p = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(p)

if __name__ == '__main__':
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    new_filename = timestamp + '_' + secure_filename(new_file.filename)
    old_filename = timestamp + '_' + secure_filename(old_file.filename)

    new_file.save(os.path.join('tmp', new_filename))
    old_file.save(os.path.join('tmp', old_filename))
    new_im = Image.open(os.path.join('tmp', new_filename))
    old_im = Image.open(os.path.join('tmp', old_filename))

    new_file_width, new_file_height = new_im.size
    old_file_width, old_file_height = old_im.size
    if new_file_width + new_file_height < 1000 or old_file_width + old_file_height < 1000:
        return prepare_output(None, None, uuid, grpid, 'Both images should be at least of size: (500, 500).')

    new_im_resized = _resize(new_im, 500, 500)  # new_im.resize((500, 500))
    old_im_resized = _resize(old_im, 500, 500)  # old_im.resize((500, 500))

    new_dir = os.path.join(app.config['UPLOAD_FOLDER'], grpid)
    if not os.path.exists(new_dir):
        os.mkdir(new_dir)
    new_filepath = os.path.join(new_dir, new_filename)
    old_filepath = os.path.join(new_dir, old_filename)
    new_im_resized.save(new_filepath)
    old_im_resized.save(old_filepath)
    json_data = controller(new_filepath, old_filepath, uuid, grpid)
    return json_data
