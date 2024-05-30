import logging

from src.dash import DashRepresentation
from src.star import StarRepresentation
from src.dataloader import DataLoader
from src.utils import dataset_conversion_layers
from logger.custom_logging import configure_logger


configure_logger("logger/logging.yaml")

logger = logging.getLogger("Main")

# =============================================================================

class StarDash:
    def __init__(self, dataset_path: str) -> None:
        # Set the dataset path
        self.dataset_path: str = dataset_path
        
        # Set the dimensions and strides
        self.xyDim: int = 112
        self.strides: int = 2
        
        # Initialize the dataloader
        self.dataloader: DataLoader = DataLoader(dataset_path=self.dataset_path, 
                                                 models_info_path='/models_eval/models_info.json',
                                                 sub_dataset_names = ['train_primesense'],
                                                 xyDim=self.xyDim)
        
        
        # Load the object ids and model info
        self.object_ids: list = self.dataloader.get_object_ids()
        self.model_info: dict = self.dataloader.load_model_info(object_ids=self.object_ids)
        
        self.star = StarRepresentation()
        self.dash = DashRepresentation()
        
    def run(self):
        # for object_id in self.object_ids:
        object_id = self.object_ids[0]
        
        # Load the ground truth data
        found_data: list[dict] = self.dataloader.load_gt_data(object_id)
        logger.info(f'Found data for {len(found_data)} occurencies of object {object_id}')
        
        for element in self.dataloader.generate_data():
            inputs, po_image, isvalid, depth, segmentation = dataset_conversion_layers(element, self.xyDim, self.xyDim, self.model_info[object_id], self.strides)

            break
        
        
    
    



# def main(dataset_path):
#     xyDim = 112
#     strides = 2
    
#     #Load model info, example: tless/models_eval/models_info.json
#     with open(f'{dataset_path}/models_eval/models_info.json') as f:
#         jsondata = json.load(f)
    
    
#     for key, model_info in jsondata.items():
#         try:
#             model_info["symmetries_discrete"] = np.array([
#                 np.array(sym).reshape((4, 4)) for sym in model_info["symmetries_discrete"]
#             ])
#             has_discrete_sym = True
#         except KeyError:
#             #model has no discrete symmetries
#             has_discrete_sym = False
            
#         # isvalid, depth, segmentation are used for loss calculation, but not used for ground truth
#         inputs, po_image, isvalid, depth, segmentation = utils.dataset_conversion_layers(xyDim, xyDim, model_info, strides)
        
#         if has_discrete_sym:
#             offset = model_info["symmetries_discrete"][0][:3,-1] / 2.0
#         else:
#             offset = 0
#         valid_dash = dash.calculate(
#             inputs['rotationmatrix'],
#             po_image,
#             offset = offset
#         )
#         valid_star = star.calculate(model_info[key], po_image)
#         #ToDo: Save star and dash images


if __name__ == "__main__":
    # import argparse
    # parser = argparse.ArgumentParser(description="")
    # parser.add_argument("-dpath", "--dataset-path", type=str, default=".", help="The path to the used dataset")
    # args = parser.parse_args()
    # main(
    #     dataset_path=args.dataset_path
    # )
    stardash = StarDash(dataset_path="/home/domin/Documents/Datasets/tless/")
    stardash.run()
    