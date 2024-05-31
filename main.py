import logging

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

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
        self.number_of_images: int = 0
        
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
        logger.info(f'Working on object {object_id} of {self.object_ids[-1]}')
        
        # Load the ground truth data
        found_data: list[dict] = self.dataloader.load_gt_data(object_id)
        logger.info(f'Found data for {len(found_data)} occurencies of object {object_id}')
        
        self.number_of_images = len(found_data)
        
        all_rgb = []
        all_depth = []
        all_valid_po = []
        all_isvalid = []
        all_segmentation = []
        
        logger.info(f'Generating data for object {object_id}')
        for element in tqdm(self.dataloader.generate_data(), total=self.number_of_images):
            inputs, valid_po, isvalid, depth, segmentation = dataset_conversion_layers(element, self.model_info[object_id], self.strides)



            all_rgb.append(inputs['rgb'])
            all_valid_po.append(valid_po)
            all_isvalid.append(isvalid)
            all_depth.append(depth)
            all_segmentation.append(segmentation)
        
        all_rgb = np.squeeze(all_rgb, axis=1)
        all_valid_po = np.squeeze(all_valid_po, axis=1)
        all_isvalid = np.squeeze(all_isvalid, axis=1)
        all_depth = np.squeeze(all_depth, axis=1)
        all_segmentation = np.squeeze(all_segmentation, axis=1)
        
        logger.warning(f'RGB shape: {all_rgb.shape}, Type: {all_rgb.dtype}')
        logger.warning(f'PO shape: {all_valid_po.shape}, Type: {all_valid_po.dtype}')
        logger.warning(f'Is Valid shape: {all_isvalid.shape}, Type: {all_isvalid.dtype}')
        logger.warning(f'Depth shape: {all_depth.shape}, Type: {all_depth.dtype}')
        logger.warning(f'Segmentation shape: {all_segmentation.shape}, Type: {all_segmentation.dtype}')
        
        pic_numbers = np.random.randint(0, self.number_of_images, 1)
        for picture in pic_numbers:
            f, axarr = plt.subplots(2, 4)
            # Display images and set titles
            axarr[0, 0].imshow(all_rgb[picture])
            axarr[0, 0].set_title('RGB Image')
            axarr[0, 1].imshow(all_rgb[picture])
            axarr[0, 1].set_title('Dash Image')
            axarr[0, 2].imshow(all_rgb[picture])
            axarr[0, 2].set_title('Star Image')
            axarr[0, 3].imshow(all_rgb[picture])
            axarr[0, 3].set_title('Destar Image')
            # axarr[0, 3].axis('off')  # Turn off axis for the empty subplot
            axarr[1, 0].imshow(all_valid_po[picture])
            axarr[1, 0].set_title('Valid PO')
            axarr[1, 1].imshow(all_isvalid[picture])
            axarr[1, 1].set_title('Is Valid')
            axarr[1, 2].imshow(all_depth[picture])
            axarr[1, 2].set_title('Depth Image')
            axarr[1, 3].imshow(all_segmentation[picture])
            axarr[1, 3].set_title('Segmentation')
            # Adjust layout to make space for titles
            plt.tight_layout()
            # Show the plot
            plt.show()
        
        
        
        
    
    



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
    