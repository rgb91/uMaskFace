## uMask Face API Command Line

### Usage:
1. Update [`env.py`](env.py).
2. Run: 
```
uMask_face_cmd.py --new_img=<new_image_path> --old_img=<old_image_path> --out_img=<out_image_path> --out_json=<out_json_path> --uuid=<UUID> --grpid=<GroupID>
```
**_Please provide square sized images for now._**

### Options:
```
-h, --help                  Show this screen
--new_img=<filename>        Path to new image (.jpg, .jpeg, .png)
--old_img=<filename>        Path to old image (.jpg, .jpeg, .png)
--out_img=<filename>        Path to output image (.jpg, .jpeg, .png)
--out_json=<filename>       Path to JSON (.json)
--uuid=<uuid>               UUID
--grpid=<grpid>             Group ID
--version                   Show version
```

### Return values:
* `0` : Successfully generated Image
* `1` : No image found in given path
* `2` : Image is unreadable (size: 0 bytes)
* `3` : Image is below minimum dimension (500x500)
* `4` : Face not detected
* `5` : Multiple face detected
* `6` : Face too small
* `-1` : Other exceptions

### TODO:
* Add code to crop images to square.