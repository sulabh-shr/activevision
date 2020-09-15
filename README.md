## Active Vision Dataset Utilities
Various utility functions to manipulate and/or visualize the Active Vision Dataset.

### Installation
git clone https://github.com/sulabh-shr/activevision.git  
pip install -e activevision

### Dataset Linking
You can either create an environment variable named **AVD_DATASET** and set the path
to the Active Vision dataset there or you can create a symbolink link to the 
dataset to **activevision/activevision/AVD_Dataset** which is understood as the default
location in defaults.py

### Visualize bounding boxes and Navigate
To visualize bounding boxes use the command.  
`python activevision/viz_nav.py --scene <scene-name> --type <box-type>`  
For example:  `python activevision/viz_nav.py --scene Home_001_1 --type both`  
For help:  
`python activevision/viz_nav.py --help`
