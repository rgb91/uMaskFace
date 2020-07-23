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
import time

import dlib
import numpy as np
from PIL import Image
from docopt import docopt
from facemorpher import blender
from facemorpher import plotter
from facemorpher import warper
from werkzeug.utils import secure_filename

from env import PROJECT_HOME
from utils import *

UPLOADS_RESIZED_DIR = os.path.join(PROJECT_HOME, 'uploads_resized')
OUTPUT_DIR = os.path.join(PROJECT_HOME, 'outputs')
ALLOWED_EXTENSIONS = {'jpg', 'jpeg'}
MIN_HEIGHT = 500
MIN_WIDTH = 500
OUTPUT_HEIGHT = 500
OUTPUT_WIDTH = 500
p = "shape_predictor_68_face_landmarks.dat"
REFERENCE_IMG_PATH = os.path.join(PROJECT_HOME, 'reference', 'reference.jpg')


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

    # if dest_filename is not None:
    #     dest_img, dest_points = load_image_points(dest_filename, size)
    #     if dest_img is None or dest_points is None:
    #         raise Exception('No face or detected face points in dest img: ' + dest_filename)
    # else:
    #     dest_img = np.zeros(images[0].shape, np.uint8)
    #     dest_points = locator.average_points(point_set)

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
            average_background = perlin_background(images)
            dest_img = blender.overlay_image(dest_img, mask, average_background)

    print('Averaged {} images'.format(num_images))
    plt = plotter.Plotter(False, num_images=1, out_filename=out_filename)
    plt.save(dest_img)


def main():
    # Read input images
    new_im = Image.open(new_img_path)
    old_im = Image.open(old_img_path)

    # Verify size of the input images
    new_file_width, new_file_height = new_im.size
    old_file_width, old_file_height = old_im.size

    if new_file_width + new_file_height < MIN_HEIGHT + MIN_WIDTH or old_file_width + old_file_height < MIN_HEIGHT + MIN_WIDTH:
        print('ERROR: Both images should be at least of size: (500, 500).')

    # Resize input images
    new_im_resized = PIL_Image_resize(new_im, OUTPUT_WIDTH + 100, OUTPUT_HEIGHT + 100)
    old_im_resized = PIL_Image_resize(old_im, OUTPUT_WIDTH + 100, OUTPUT_HEIGHT + 100)

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
        print(e)
        return
    landmarks, msg = get_landmarks(new_img_resized_path, detector, predictor)
    out_json = get_output_json(landmarks, uuid, grpid, msg)
    with open(out_json_path, 'w') as outfile:
        json.dump(out_json, outfile)


if __name__ == '__main__':
    args = docopt(__doc__, version='uMask Face Command line 1.0')
    new_img_path, old_img_path = args['--new_img'], args['--old_img']
    out_img_path, out_json_path = args['--out_img'], args['--out_json']
    uuid, grpid = args['--uuid'], args['--grpid']

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(p)

    main()
    print('SUCCESS: Output face saved at', out_img_path)

# python uMask_face_cmd.py --new_img "./morph_images/group04/new.jpg" --old_img "./morph_images/group04/old.jpg" --out_img "./morph_images/group04/out.jpg" --out_json "./morph_images/group04/out.json" --uuid 12345 --grpid g04perlin
