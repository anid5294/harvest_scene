# CAMERA_CONFIG = {
#     "front": {
#         "enabled": True,
#         "width": 640,
#         "height": 480,
#     },

#     "left_wrist": {
#         "enabled": True,
#         "width": 640,
#         "height": 480,
#     },

#     "right_wrist": {
#         "enabled": True,
#         "width": 640,
#         "height": 480,
#     },
# }

CAMERA_CONFIG = {
    "front": {
        "enabled": True,
        "width": 640,
        "height": 480,
        "mode": "model",
    },

    "left_wrist": {
        "enabled": True,
        "width": 640,
        "height": 480,
        "mode": "body",
        "body": "left_wrist_yaw_link",

        # Local position in the wrist body frame.
        # Camera sits above the wrist instead of inside / beside the palm.
        "position": [0.04, 0.0, 0.07],

        # Look along the hand (+X) with a small downward pitch.
        "forward": [1.0, 0.0, -0.30],
        "up": [0.0, 0.0, 1.0],

        # Only used to inherit the camera FOV / projection settings.
        "template_camera": "left_wrist",
    },

    "right_wrist": {
        "enabled": True,
        "width": 640,
        "height": 480,
        "mode": "body",
        "body": "right_wrist_yaw_link",

        # The old right camera was offset strongly to -Y.  Keep it centered
        # over the wrist, like the left camera.
        "position": [0.04, 0.0, 0.07],

        # Look along the hand (+X) with a small downward pitch.
        "forward": [1.0, 0.0, -0.30],
        "up": [0.0, 0.0, 1.0],

        "template_camera": "right_wrist",
    },

    "head": {
        "enabled": True,
        "width": 640,
        "height": 480,
        "mode": "body",

        # In this G1 model the head mesh is attached to torso_link, so mount
        # the camera near head height on torso_link.
        "body": "torso_link",
        "position": [0.10, 0.0, 0.34],

        # Forward (+X) and pitched down toward the tabletop workspace.
        "forward": [1.0, 0.0, -0.65],
        "up": [0.0, 0.0, 1.0],

        # Reuse an existing 70-degree-ish camera projection as the template;
        # recorder.py replaces only its world pose.
        "template_camera": "right_wrist",

        # strawberry_load.py currently does not request "head" explicitly.
        # recorder.py reads this flag and adds it automatically.
        "always_record": True,
    },
}