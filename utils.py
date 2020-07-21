"""
Created by Sanjay at 7/21/2020

Feature: Enter feature name here
Enter feature description here
"""
import traceback

import cv2
from PIL import ExifTags
from facemorpher import locator, aligner
from imutils import face_utils


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


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


def PIL_Image_resize(im, width, height):
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


def load_image_points(path, size):
    img = cv2.imread(path)
    points = locator.face_points(img)

    if len(points) == 0:
        print('No face in %s' % path)
        return None, None
    else:
        return aligner.resize_align(img, points, size)


def get_landmarks(image_path, detector, predictor):
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
