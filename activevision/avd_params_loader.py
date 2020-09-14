import os
import numpy as np
import pandas as pd
from activevision.utils.matlab_utils import load_image_struct
from activevision.defaults import IMG_STRUCT_FNAME, ALL_SCENES


class AVDParamsLoader:
    def __init__(self, data_root):
        self.data_root = data_root
        self.params = None
        self.available_scenes = []

    @staticmethod
    def col_convert_function(col_name):
        """Return function for data conversion of column in image_struct df.

        Parameters
        ----------
        col_name: str
            Name of the column in image_struct.mat

        Returns
        -------
        Respective function or identity function if not defined specifically.
        """
        if col_name == 'image_id' or col_name == 'camera_id':
            return lambda el: int(el[0]) if len(el) > 0 else -1
        elif col_name == 'image_name' or col_name.startswith(('rotate', 'translate')):
            # default type is np.str_
            return lambda el: np.NaN if type(el[0]) == np.ndarray else el[0]
        elif col_name in ('t', 'R'):
            return lambda el: el if el.size > 0 else np.NaN
        elif col_name == 'cluster_id':
            return lambda el: el[0][0]
        # Return identity function
        return lambda x: x

    def extract_mat_files(self, scenes=None):
        """ Extract data appropriately for specified scenes in dataframe format.

        Parameters
        ----------
        scenes: list or None, default None
            List of scenes for which data is to be extracted. If None, uses all scenes specified in
            defaults.

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
            path = os.path.join(self.data_root, scene, IMG_STRUCT_FNAME)
            image_structs, scale = load_image_struct(path)
            scale = int(scale)  # default is uint
            columns = image_structs.dtype.names
            dataframe = pd.DataFrame(data=image_structs, columns=columns)

            for col in columns:
                dataframe[col] = dataframe[col].apply(func=self.col_convert_function(col))

            dataframe['scale'] = scale
            dataframe.set_index('image_name', inplace=True)

            if self.params is None:
                self.params = dataframe
            else:
                self.params = self.params.append(dataframe, verify_integrity=True)

            self.available_scenes.append(scene)

