import os
import json
import numpy as np

from activevision.utils.matlab_utils import load_mat
from activevision.defaults import SCENE_ANNOTATIONS_FNAME, ALL_SCENES, ALL_DIRECTIONS, \
    LABEL_MAP_FNAME, IMG_FOLDER


class AVDAnnotations:
    def __init__(self, data_root):
        self.data_root = data_root
        self.annotations = dict()  # Scene: {'img_name1': {...}, ... }
        self.loaded_scenes = []
        self.instance_id2name_map = self._load_label_map()

    def __str__(self):
        return f'AVD Instance Annotations. Scenes: {self.loaded_scenes}'

    def _load_label_map(self):
        with open(os.path.join(self.data_root, LABEL_MAP_FNAME), 'r') as f:
            label_map_json = json.load(f)

        # Convert string idx to int index
        instance_id2name_map = dict()
        for id_, name in label_map_json.items():
            instance_id2name_map[int(id_)] = name

        del label_map_json
        return instance_id2name_map

    def load_annotations(self, scenes=None):
        """ Load annotations for specified scenes.

            Load annotations for specified scenes and add the scene name(s) to
        the list of loaded scenes. If no scene is specified, loads for all
        scenes specified in *activevision* default.

        Parameters
        ----------
        scenes: list or None
            List of scenes to load. All specified scenes must be present in list
        of available scenes by default in *activevision*. Loads all scenes when
        None.

        Returns
        -------
        None
        """
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
            self.loaded_scenes.append(scene)

    def get_neighbor_image(self, scene, img_name, direction):
        assert scene in self.loaded_scenes, f'Specified scene is not loaded: {scene}'
        assert direction in ALL_DIRECTIONS, \
            f'Direction {direction} not in list of available directions\n' \
            f'Possible directions:\n{ALL_DIRECTIONS}'

        return self.annotations[scene][img_name][direction]

    def get_image_boxes_raw(self, scene, img_name):
        assert scene in self.loaded_scenes, f'Specified scene is not loaded: {scene}'

        return self.annotations[scene][img_name]['bounding_boxes']

    def get_image_boxes(self, scene, img_name):
        boxes_raw = self.get_image_boxes_raw(scene=scene, img_name=img_name)
        if len(boxes_raw) != 0:
            boxes_raw = np.array(boxes_raw)
            boxes = boxes_raw[:, :4]
            instance_ids = boxes_raw[:, 4]
            difficulties = boxes_raw[:, 5]
            instance_names = [self.instance_id2name(i) for i in instance_ids]
        else:
            boxes = []
            instance_ids = []
            difficulties = []
            instance_names = []
        return {'instance_boxes': boxes, 'instance_ids': instance_ids,
                'instance_difficulties': difficulties, 'instance_names': instance_names}

    def idx2image(self, scene, idx):
        scene_images = list(self.annotations[scene].keys())
        if idx >= len(scene_images):
            raise IndexError(
                f'Index {idx} is greater than number of images {len(scene_images)} in {scene}.')

        return scene_images[idx]

    def instance_id2name(self, idx):
        return self.instance_id2name_map[idx]

    def img_name2path(self, scene, name):
        return os.path.join(self.data_root, scene, IMG_FOLDER, name)


class AVDCategoryAnns(AVDAnnotations):
    def __init__(self, data_root):
        super().__init__(data_root)
        self.category_ann_path = os.path.join(self.data_root, 'AVD_Category_Bboxes')
        self.available_scenes = self._check_available_scenes()
        self.category_id2name_map = dict()

    def __str__(self):
        return f'AVD Instance Annotations. Scenes: {self.loaded_scenes}'

    def _check_available_scenes(self):
        avail_scenes = []
        for i in os.listdir(self.category_ann_path):
            if os.path.isdir(os.path.join(self.category_ann_path, i)) and i in ALL_SCENES:
                avail_scenes.append(i)
        return avail_scenes

    def load_annotations(self, scenes=None):
        """ Load annotations for both instances and categories.
            First, loads the instance annotations using it's parent class. Then,
            for each image in the loaded annotations, searches for category
            annotation file. *Skips* image annotation if respective annotation
            file is not found and *adds empty list* if no annotation file is
            empty.
        Parameters
        ----------
        scenes: list, None, default=None
            List of scenes for which annotations are to be loaded. If not
            specified, loads for all available instances of default.

        Returns
        -------
        None
        """

        # Load all available scenes by default
        if scenes is None:
            scenes = self.available_scenes.copy()

        # Check specified scenes have category annotations
        for i in scenes:
            assert i in self.available_scenes, f'Category annotations not available for {i}'

        # Load instance annotations
        super().load_annotations(scenes=scenes)

        # Add category annotation for each image
        for scene in self.annotations:
            scene_ann = self.annotations[scene]
            for img_name in scene_ann:
                mat_name = img_name.split('.')[0] + '.mat'
                ann_path = os.path.join(self.category_ann_path, scene, 'bboxes', mat_name)

                if os.path.isfile(ann_path):
                    scene_ann[img_name]['cat_bounding_boxes'] = img_ann = []

                    mat_content = load_mat(ann_path)['bboxes']
                    # Skip if there are no bounding boxes
                    if len(mat_content) == 0:
                        continue

                    mat_content = mat_content[0]
                    # print(mat_content.dtype.names)
                    for obj in mat_content:
                        obj = obj
                        obj_cat = str(np.squeeze(obj[0]))
                        x1 = int(np.squeeze(obj[4]))
                        y1 = int(np.squeeze(obj[3]))
                        x2 = int(np.squeeze(obj[6]))
                        y2 = int(np.squeeze(obj[5]))
                        obj_cat_id = int(np.squeeze(obj[1]))
                        img_ann.append([x1, y1, x2, y2, obj_cat_id])

                        # TODO: Add separate file for category label mapping
                        # Check integrity of object id to name
                        if obj_cat_id not in self.category_id2name_map:
                            self.category_id2name_map[obj_cat_id] = obj_cat
                        else:
                            assert self.category_id2name_map[obj_cat_id] == obj_cat, \
                                f'Category id to name mismatch for {img_name}!'

    def category_id2name(self, idx):
        return self.category_id2name_map[idx]

    def get_image_boxes(self, scene, img_name, box_type='both'):
        """ Get bounding boxes for instances or categories or both.

            Get the list of bounding boxes for instances or categories or both
            depending on `box_type` argument. If the category specific key is
            not found for the specified image, or if there are 0 category
            specific bounding boxes, it returns *empty lists* for all return
            category specific keys. See parent class description for function of
            the same name for behaviour of instance specific output keys.

        Parameters
        ----------
        scene: str
            Scene where the image is located.
        img_name: str
            Name of the image (with .jpg)
        box_type: ('instances, 'categories', 'both'), default='both'
            Specifies what type of bounding boxes to return

        Returns
        -------
        dict
            'instance_boxes', 'instance_ids', 'instance_names', 'instance_difficulties',
            'category_boxes', 'category_ids', 'category_names'
        """

        assert box_type in ('category', 'instance', 'both')
        output_dict = {}

        if box_type == 'instance' or box_type == 'both':
            output_dict.update(super().get_image_boxes(scene=scene, img_name=img_name))

        if box_type == 'category' or box_type == 'both':

            assert scene in self.loaded_scenes, f'Specified scene is not loaded: {scene}'
            # Default empty values
            cat_boxes = []
            cat_ids = []
            category_names = []

            if 'cat_bounding_boxes' in self.annotations[scene][img_name]:
                cat_boxes_raw = self.annotations[scene][img_name]['cat_bounding_boxes']
                if len(cat_boxes_raw) != 0:
                    cat_boxes_raw = np.array(cat_boxes_raw)
                    cat_boxes = cat_boxes_raw[:, :4]
                    cat_ids = cat_boxes_raw[:, 4]
                    category_names = [self.category_id2name(i) for i in cat_ids]

            output_dict.update({'category_boxes': cat_boxes, 'category_ids': cat_ids,
                                'category_names': category_names})

        return output_dict


if __name__ == '__main__':
    from activevision.defaults import AVD_DATASET

    avd_cat = AVDCategoryAnns(data_root=AVD_DATASET)
    print(avd_cat)
    avd_cat.load_annotations()
