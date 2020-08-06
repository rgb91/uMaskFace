"""
::

  uMask Face API Command Line

  Usage:
    uMask_face_cmd.py --new_img=<new_image_path> --old_img=<old_image_path> --out_img=<out_image_path> --out_json=<out_json_path> --uuid=<UUID> --grpid=<GroupID>

  Options:
    -h, --help                  Show this screen
    --new_img=<filename>        Path to new image (.jpg, .jpeg, .png)
    --old_img=<filename>        Path to old image (.jpg, .jpeg, .png)
    --out_img=<filename>        Path to output image (.jpg, .jpeg, .png)
    --out_json=<filename>       Path to JSON (.json)
    --uuid=<uuid>               UUID
    --grpid=<grpid>             Group ID
    --version                   Show version


"""

import os
import sys
import time
import dlib
import numpy as np
from PIL import Image
from docopt import docopt
from facemorpher import blender
from facemorpher import plotter
from facemorpher import warper
from werkzeug.utils import secure_filename
from locator import face_points
from env import PROJECT_HOME
from utils import *

PROJECT_HOME_FACE = os.path.join(PROJECT_HOME, 'unmask-face-api')
UPLOADS_RESIZED_DIR = os.path.join(PROJECT_HOME_FACE, 'uploads_resized')
OUTPUT_DIR = os.path.join(PROJECT_HOME_FACE, 'outputs')
ALLOWED_EXTENSIONS = {'jpg', 'jpeg'}
MIN_HEIGHT = 500
MIN_WIDTH = 500
OUTPUT_HEIGHT = 500
OUTPUT_WIDTH = 500
p = "shape_predictor_68_face_landmarks.dat"
REFERENCE_IMG_PATH = os.path.join(PROJECT_HOME_FACE, 'reference', 'reference.jpg')


def average_face(imgpaths, width=500, height=500, background='black',
                 blur_edges=False, out_filename='result.jpg'):
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

    dest_img, dest_points = load_image_points(REFERENCE_IMG_PATH, size)

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
            # average_background = np.uint8(locator.average_points(images))
            avg_background = perlin_background(images)
            avg_background[np.where((avg_background == [0, 0, 0]).all(axis=2))] = [128, 128, 128]  # black -> gray pixels in background
            dest_img = blender.overlay_image(dest_img, mask, avg_background)

    print('Averaged {} images'.format(num_images))
    plt = plotter.Plotter(False, num_images=1, out_filename=out_filename)
    plt.save(dest_img)


def validate(n_img, o_img):
    # Return value 1: No image found in given path
    if not os.path.exists(n_img) or not os.path.exists(o_img):
        log_error('No image found in given path')
        return False, 1, None, None

    # Return value 2: Image is unreadable (size: 0 bytes)
    if not os.path.getsize(n_img) > 0 or not os.path.getsize(o_img) > 0:
        log_error('Image is unreadable (size: 0 bytes)')
        return False, 2, None, None

    # Read input images
    new_im = Image.open(new_img_path)
    old_im = Image.open(old_img_path)

    # Return value 3: Image is below minimum dimension (500x500)
    new_file_width, new_file_height = new_im.size
    old_file_width, old_file_height = old_im.size

    if new_file_width + new_file_height < MIN_HEIGHT + MIN_WIDTH or \
            old_file_width + old_file_height < MIN_HEIGHT + MIN_WIDTH:
        log_error('Image is below minimum dimension')
        return False, 3, new_im, old_im

    # Return value 4: Face not detected
    # Return value 5: Multiple face detected
    new_im_cv2 = cv2.imread(n_img)
    old_im_cv2 = cv2.imread(o_img)
    new_img_points = face_points(new_im_cv2)
    old_img_points = face_points(old_im_cv2)
    if len(new_img_points) == 0:
        log_error('Face not detected')
        return False, 4, new_im, old_im
    if len(old_img_points) == 0:
        log_error('Face not detected')
        return False, 4, new_im, old_im
    # if len(new_img_points) > 1:
    #     log_error('Multiple face detected')
    #     return False, 5, new_im, old_im

    # Return value 6: Face too small
    face_detector = dlib.get_frontal_face_detector()
    gray_image = cv2.cvtColor(new_im_cv2, cv2.COLOR_BGR2GRAY)
    detected_faces = face_detector(gray_image, 0)
    x = detected_faces[0]
    x1, y1, x2, y2 = x.left(), x.top(), x.right(), x.bottom()
    ratio_of_height = (x2 - x1) / OUTPUT_HEIGHT
    ratio_of_weight = (y2 - y1) / OUTPUT_WIDTH
    if (ratio_of_height + ratio_of_weight) / 2.0 < 0.30:
        log_error('Face too small')
        return False, 6, new_im, old_im

    return True, 0, new_im, old_im


def main(new_im, old_im):
    # Resize input images
    new_im_resized = PIL_Image_resize(new_im, OUTPUT_WIDTH, OUTPUT_HEIGHT)
    old_im_resized = PIL_Image_resize(old_im, OUTPUT_WIDTH, OUTPUT_HEIGHT)

    # Save the resized input images
    new_dir = os.path.join(UPLOADS_RESIZED_DIR, grpid)
    if not os.path.exists(new_dir):
        os.mkdir(new_dir)
    timestamp = time.strftime("%Y%m%d%H%M%S")
    new_img_resized_name = timestamp + '_' + os.path.basename(new_img_path)
    old_img_resized_name = timestamp + '_' + os.path.basename(old_img_path)
    new_img_resized_path = os.path.join(new_dir, new_img_resized_name)
    old_img_resized_path = os.path.join(new_dir, old_img_resized_name)
    new_im_resized.save(new_img_resized_path)
    old_im_resized.save(old_img_resized_path)

    # Process input face images
    try:
        average_face([new_img_resized_path, old_img_resized_path], background='average', out_filename=out_img_path)
    except Exception as e:
        log_error(e)
        return -1
    landmarks, msg = get_landmarks(new_img_resized_path, detector, predictor)
    out_json = get_output_json(landmarks, uuid, grpid, msg)
    with open(out_json_path, 'w') as outfile:
        json.dump(out_json, outfile)
        return 0

if __name__ == '__main__':
    args = docopt(__doc__, version='uMask Face Command line 1.0')
    new_img_path, old_img_path = args['--new_img'], args['--old_img']
    out_img_path, out_json_path = args['--out_img'], args['--out_json']
    uuid, grpid = args['--uuid'], args['--grpid']

    new_img_path = os.path.join(PROJECT_HOME, new_img_path)
    old_img_path = os.path.join(PROJECT_HOME, old_img_path)
    out_img_path = os.path.join(PROJECT_HOME, out_img_path)
    out_json_path = os.path.join(PROJECT_HOME, out_json_path)

    input_is_alright, return_val, new_im, old_im = validate(new_img_path, old_img_path)
    if input_is_alright:
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(p)

        return_val = main(new_im, old_im)
        if return_val == 0:
            print(0, 'SUCCESS')
        else:
            print(return_val, 'ERROR')
    else:
        print(return_val, 'ERROR')
