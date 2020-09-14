import os
import json
import numpy as np

from activevision.defaults import SCENE_ANNOTATIONS_FNAME, ALL_SCENES, ALL_DIRECTIONS, \
    LABEL_MAP_FNAME, IMG_FOLDER


class AVDAnnotations:
    def __init__(self, data_root):
        self.data_root = data_root
        self.annotations = dict()
        self.available_scenes = []
        self.label_id2name_map = self._load_label_map()

    def _load_label_map(self):
        with open(os.path.join(self.data_root, LABEL_MAP_FNAME), 'r') as f:
            label_map_json = json.load(f)

        # Convert string idx to int index
        label_id2name_map = dict()
        for id_, name in label_map_json.items():
            label_id2name_map[int(id_)] = name

        del label_map_json
        return label_id2name_map

    def load_annotations(self, scenes=None):
        # Load all scenes by default
        if scenes is None or (type(scenes) == list and len(scenes) == 0):
            scenes = ALL_SCENES

        # Check all scenes are valid and stored in default available scenes list
        for scene in scenes:
            assert scene in ALL_SCENES, f'Specified scene not available in default scenes: {scene}'

        # Load parameters for each scene and concatenate
        for scene in scenes:
            ann_file_path = os.path.join(self.data_root, scene, SCENE_ANNOTATIONS_FNAME)
            with open(ann_file_path, 'r') as f:
                annotations = json.load(f)
            self.annotations[scene] = annotations
            self.available_scenes.append(scene)

    def get_neighbor_image(self, scene, img_name, direction):
        assert scene in self.available_scenes, f'Specified scene is not loaded: {scene}'
        assert direction in ALL_DIRECTIONS, \
            f'Direction {direction} not in list of available directions\n' \
            f'Possible directions:\n{ALL_DIRECTIONS}'

        return self.annotations[scene][img_name][direction]

    def get_image_boxes_raw(self, scene, img_name):
        assert scene in self.available_scenes, f'Specified scene is not loaded: {scene}'

        return self.annotations[scene][img_name]['bounding_boxes']

    def get_image_boxes(self, scene, img_name):
        boxes_raw = self.get_image_boxes_raw(scene=scene, img_name=img_name)
        if len(boxes_raw) != 0:
            boxes_raw = np.array(boxes_raw)
            boxes = boxes_raw[:, :4]
            instance_ids = boxes_raw[:, 4]
            difficulties = boxes_raw[:, 5]
            instance_names = [self.label_idx_to_name(i) for i in instance_ids]
        else:
            boxes = []
            instance_ids = []
            difficulties = []
            instance_names = []
        return {'boxes': boxes, 'instance_ids': instance_ids, 'difficulties': difficulties,
                'instance_names': instance_names}

    def idx2image(self, scene, idx):
        scene_images = list(self.annotations[scene].keys())
        assert idx < len(scene_images), \
            f'Index is greater than number of images {len(scene_images)} in {scene}.'

        return scene_images[idx]

    def label_idx_to_name(self, idx):
        return self.label_id2name_map[idx]

    def img_name2path(self, scene, name):
        return os.path.join(self.data_root, scene, IMG_FOLDER, name)
