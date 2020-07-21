"""
Created by Sanjay at 6/24/2020

Feature: Enter feature name here
Enter feature description here
"""
import functools
import os
import time
import traceback

import cv2
import codecs, json
import base64
from matplotlib import gridspec
import matplotlib.pylab as plt
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from imutils import face_utils
import dlib
from PIL import Image, ExifTags
from env import PROJECT_HOME
from tensorflow.python.client import device_lib
from flask import Flask, flash, request, redirect, url_for
from werkzeug.utils import secure_filename

from facemorpher import locator
from facemorpher import aligner
from facemorpher import warper
from facemorpher import blender
from facemorpher import plotter


# print(device_lib.list_local_devices())
# print("TF Version: ", tf.__version__)
# print("TF-Hub version: ", hub.__version__)
# print("Eager mode enabled: ", tf.executing_eagerly())
# print("GPU available: ", tf.test.is_gpu_available())

# class NumpyEncoder(json.JSONEncoder):
#     def default(self, obj):
#         if isinstance(obj, np.ndarray):
#             return obj.tolist()
#         return json.JSONEncoder.default(self, obj)

def crop_center(image):
    """Returns a cropped square image."""
    shape = image.shape
    new_shape = min(shape[1], shape[2])
    offset_y = max(shape[1] - shape[2], 0) // 2
    offset_x = max(shape[2] - shape[1], 0) // 2
    image = tf.image.crop_to_bounding_box(
        image, offset_y, offset_x, new_shape, new_shape)
    return image


@functools.lru_cache(maxsize=None)
def load_image(image_location, image_size=(256, 256), preserve_aspect_ratio=True, is_url=False):
    """Loads and preprocesses images."""
    if is_url:
        # Cache image file locally.
        image_path = tf.keras.utils.get_file(os.path.basename(image_location)[-128:], image_location)
    else:
        image_path = image_location

    # Load and convert to float32 numpy array, add batch dimension, and normalize to range [0, 1].
    img = plt.imread(image_path).astype(np.float32)[np.newaxis, ...]

    if img.max() > 1.0:
        img = img / 255.
    if len(img.shape) == 3:
        img = tf.stack([img, img, img], axis=-1)

    img = crop_center(img)
    img = tf.image.resize(img, image_size, preserve_aspect_ratio=True)
    return img


def show_n(images, titles=('',)):
    n = len(images)
    image_sizes = [image.shape[1] for image in images]
    w = (image_sizes[0] * 6) // 320
    fig = plt.figure(figsize=(w * n, w))
    gs = gridspec.GridSpec(1, n, width_ratios=image_sizes)
    for i in range(n):
        plt.subplot(gs[i])
        plt.imshow(images[i][0], aspect='equal')
        plt.axis('off')
        plt.title(titles[i] if len(titles) > i else '')
    fig.savefig('output.png')


def get_landmarks(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)
    print(len(rects))
    if len(rects) > 1:
        # print('More than one face detected')
        return None, 'More than one face detected'
    elif len(rects) < 1:
        # print('No face detected')
        return None, 'No face detected'
    else:
        # Make the prediction and transform it to numpy array
        shape = predictor(gray, rects[0])
        shape = face_utils.shape_to_np(shape)
        return shape, 'Successfully detected a face'


def list_imgpaths(imgfolder):
    for fname in os.listdir(imgfolder):
        if (fname.lower().endswith('.jpg') or
                fname.lower().endswith('.png') or
                fname.lower().endswith('.jpeg')):
            yield os.path.join(imgfolder, fname)


def sharpen(img):
    blured = cv2.GaussianBlur(img, (0, 0), 2.5)
    return cv2.addWeighted(img, 1.4, blured, -0.4, 0)


def load_image_points(path, size):
    img = cv2.imread(path)
    points = locator.face_points(img)

    if len(points) == 0:
        print('No face in %s' % path)
        return None, None
    else:
        return aligner.resize_align(img, points, size)


def prepare_output(img_path: str, landmarks, uuid: str, grpid: str, msg: str):
    """
    Prepare JSON data to return.
    :param msg:
    :param img: Stylized image (numpy array 3D)
    :param landmarks: List of Landmarks of input face (numpy array)
    :param filename: name of the output file
    :param uuid: String, Unique Identifier
    :param grpid: String, Group Identifier
    :return: JSON String (Base64 image, landmarks)
    """
    if landmarks is None or len(img_path) < 1:
        return json.dumps({
            "uuid": uuid,
            "grpid": grpid,
            "msg": msg,
            "facial_features": '',
            "stylized_image": ''
        })
    # else:
    #     filepath = os.path.join(OUTPUT_DIRECTORY, filename)
    #     img = img.numpy()
    #     tf.keras.preprocessing.image.save_img(filepath, img, data_format=None, file_format=None, scale=True)

    # img = tf.cast(img, tf.uint8)
    # img = tf.image.encode_png(img, compression=-1, name=None)
    # tf.io.write_file(filepath, img, name=None)

    # cv2.imwrite(filepath, img)

    with open(img_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
    out_json_data = {
        "uuid": uuid,
        "grpid": grpid,
        "msg": msg,
        "facial_features": landmarks.tolist(),
        "stylized_image": encoded_image
    }
    return json.dumps(out_json_data)


def stylize(face_image_location, style_image_location):
    # The content image size can be arbitrary.
    # The style prediction model was trained with image size 256 and it's the
    # recommended image size for the style image (though, other sizes work as
    # well but will lead to different results).
    face_img_size = (OUTPUT_IMAGE_SIZE, OUTPUT_IMAGE_SIZE)
    style_img_size = (256, 256)  # Recommended to keep it at 256.

    input_face_image = load_image(face_image_location, face_img_size)
    style_image = load_image(style_image_location, style_img_size, is_url=True)
    style_image = tf.nn.avg_pool(style_image, ksize=[3, 3], strides=[1, 1], padding='SAME')
    # show_n([input_face_image, style_image], ['Content image', 'Style image'])

    # Stylize content image with given style image.
    # This is pretty fast within a few milliseconds on a GPU.
    outputs = hub_module(tf.constant(input_face_image), tf.constant(style_image))
    # print(outputs[0][0].shape)
    return outputs[0][0]


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


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


def main_controller(new_image_path, old_image_path, uuid, grpid):
    # style_image_url = 'https://upload.wikimedia.org/wikipedia/commons/0/0a/The_Great_Wave_off_Kanagawa.jpg'
    # stylized_image = stylize(input_image_path, style_image_url)
    input_filename = os.path.basename(new_image_path)
    out_filename = str(input_filename.split('_')[0]) + '.jpg'
    out_img_path = os.path.join(OUTPUT_DIRECTORY, out_filename)
    average_face([new_image_path, old_image_path], background='average', out_filename=out_img_path)
    landmarks, msg = get_landmarks(new_image_path)
    return prepare_output(out_img_path, landmarks, uuid, grpid, msg)


def _resize(im, width, height):
    try:
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation] == 'Orientation': break
        exif = dict(im._getexif().items())

        if exif[orientation] == 3:
            im = im.rotate(180, expand=True)
        elif exif[orientation] == 6:
            im = im.rotate(270, expand=True)
        elif exif[orientation] == 8:
            im = im.rotate(90, expand=True)

        # image.thumbnail((width, heght), Image.ANTIALIAS)
        im_resized = im.resize((width, height))
        return im_resized
    except:
        traceback.print_exc()


app = Flask(__name__)
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SECRET_KEY'] = 'super secret key'
app.config['UPLOAD_FOLDER'] = os.path.join(PROJECT_HOME, 'uploads')
OUTPUT_DIRECTORY = os.path.join(PROJECT_HOME, 'outputs')
OUTPUT_IMAGE_SIZE = 384  # @param {type:"integer"}
ALLOWED_EXTENSIONS = {'jpg', 'jpeg'}
p = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(p)


@app.route("/")
def hello():
    return "Please send request to " \
           "<a href=\"https://face.d2.comp.nus.edu.sg:5004/umask\">face.d2.comp.nus.edu.sg/umask."


@app.route("/umask", methods=['GET', 'POST'])
def main():
    if request.method == 'POST':
        uuid, grpid = -1, -1
        if 'uuid' not in request.form or 'grpid' not in request.form:
            return prepare_output(None, None, uuid, grpid, 'UUID(uuid) or Group ID(grpid) not found')
        else:
            uuid = request.form['uuid']
            grpid = request.form['grpid']
        if 'new_img' not in request.files or 'old_img' not in request.files:
            return prepare_output(None, None, uuid, grpid, 'Images not posted')
        new_file = request.files['new_img']
        old_file = request.files['old_img']
        if new_file.filename == '':
            return prepare_output(None, None, uuid, grpid, 'New image NOT found')
        if old_file.filename == '':
            return prepare_output(None, None, uuid, grpid, 'Old image NOT found')
        if new_file and allowed_file(new_file.filename):
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
            json_data = main_controller(new_filepath, old_filepath, uuid, grpid)
            return json_data
    return '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form method=post enctype=multipart/form-data>
      <input type="file" name="img">
      <input type="submit" value="Upload">
    </form>
    '''


if __name__ == '__main__':
    # hub_handle = 'https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2'
    # hub_module = hub.load(hub_handle)
    app.run(host='0.0.0.0', port=5004)
