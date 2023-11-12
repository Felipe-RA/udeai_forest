import ee
from geetiles import utils

class DatasetDefinition:

    def __init__(self, dataset_def):
        self.dataset_def = dataset_def

    def get_dataset_name(self):
        return str("treecover2019.py")


    def get_gee_image(self, **kwargs):
        gee_image = ee.ImageCollection('MODIS/006/MOD44B')\
                    .filterDate('2019-01-01', '2019-12-31')\
                    .select('Percent_Tree_Cover', 'Percent_NonTree_Vegetation')\
                    .median()\
                    .visualize(min=0, max=100)        
        
        return gee_image
                     
    def get_dtype(self):
        return 'uint8'
