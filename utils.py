"""
Created by Sanjay at 7/21/2020

Feature: Enter feature name here
Enter feature description here
"""
import sys
import cv2
import json
import base64
import locator
import traceback
import numpy as np
from PIL import ExifTags
from imutils import face_utils
# from facemorpher import aligner
import aligner

def allowed_file(filename, allowed_extensions):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions


def get_output_json(landmarks, uuid: str, grpid: str, msg: str):
    """
    Prepare JSON data to return.
    :param msg: String
    :param landmarks: List of Landmarks of input face (numpy array)
    :param uuid: String, Unique Identifier
    :param grpid: String, Group Identifier
    :return: JSON String (Base64 image, landmarks)
    """
    if landmarks is None:
        return json.dumps({
            "uuid": uuid,
            "grpid": grpid,
            "msg": msg,
            "facial_features": ''
        })
    out_json_data = {
        "uuid": uuid,
        "grpid": grpid,
        "msg": msg,
        "facial_features": landmarks.tolist()
    }
    return json.dumps(out_json_data)


def PIL_Image_resize(im, width, height):
    try:
        if im._getexif() is not None:
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


def load_image_points(path, size):
    img = cv2.imread(path)
    points = locator.face_points(img)

    if len(points) == 0:
        # print('No face in %s' % path)
        return None, None
    else:
        return aligner.resize_align(img, points, size)


def get_landmarks(image_path, detector, predictor):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)
    # print(len(rects))
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


def generate_perlin_noise_2d(shape, res, tileable=(False, False)):
    def f(t):
        return 6 * t ** 5 - 15 * t ** 4 + 10 * t ** 3

    delta = (res[0] / shape[0], res[1] / shape[1])
    d = (shape[0] // res[0], shape[1] // res[1])
    grid = np.mgrid[0:res[0]:delta[0], 0:res[1]:delta[1]].transpose(1, 2, 0) % 1
    # Gradients
    angles = 2 * np.pi * np.random.rand(res[0] + 1, res[1] + 1)
    gradients = np.dstack((np.cos(angles), np.sin(angles)))
    if tileable[0]:
        gradients[-1, :] = gradients[0, :]
    if tileable[1]:
        gradients[:, -1] = gradients[:, 0]
    gradients = gradients.repeat(d[0], 0).repeat(d[1], 1)
    g00 = gradients[:-d[0], :-d[1]]
    g10 = gradients[d[0]:, :-d[1]]
    g01 = gradients[:-d[0], d[1]:]
    g11 = gradients[d[0]:, d[1]:]
    # Ramps
    n00 = np.sum(np.dstack((grid[:, :, 0], grid[:, :, 1])) * g00, 2)
    n10 = np.sum(np.dstack((grid[:, :, 0] - 1, grid[:, :, 1])) * g10, 2)
    n01 = np.sum(np.dstack((grid[:, :, 0], grid[:, :, 1] - 1)) * g01, 2)
    n11 = np.sum(np.dstack((grid[:, :, 0] - 1, grid[:, :, 1] - 1)) * g11, 2)
    # Interpolation
    t = f(grid)
    n0 = n00 * (1 - t[:, :, 0]) + t[:, :, 0] * n10
    n1 = n01 * (1 - t[:, :, 0]) + t[:, :, 0] * n11
    return np.sqrt(2) * ((1 - t[:, :, 1]) * n0 + t[:, :, 1] * n1)


def perlin_background(images):
    np.random.seed(0)
    noise2d = generate_perlin_noise_2d((500, 500), (5, 5))
    noise_img1 = np.repeat(noise2d[:, :, np.newaxis], 3, 2).ravel()
    noise_img2 = np.ones(noise_img1.shape) - noise_img1
    noise_img1 = (noise_img1 - np.min(noise_img1)) / np.ptp(noise_img1)
    noise_img2 = (noise_img2 - np.min(noise_img2)) / np.ptp(noise_img2)
    img1_weighted = np.multiply(images[0].ravel(), noise_img1)
    img2_weighted = np.multiply(images[1].ravel(), noise_img2)
    final_img_1d = np.sum(np.vstack((img1_weighted, img2_weighted)), axis=0)
    final_img = np.uint8(final_img_1d.reshape((500, 500, 3)))
    return final_img

def log_error(msg):
    original_stdout = sys.stdout
    with open('log.txt', 'a') as f:
        sys.stdout = f  # Change the standard output to the file we created.
        print(msg)  # write the exception to the log file
        sys.stdout = original_stdout  # Reset the standard output to its original value
