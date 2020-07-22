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

### TODO:
* Add code to crop images to square.
* Add face to image ratio checker.
* Add Perlin noise.