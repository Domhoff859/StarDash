import logging

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.dash import DashRepresentation
from src.star import StarRepresentation
from src.destar import DestarRepresentation
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
        self.strides: int = 1
        self.number_of_images: int = 0
        
        # Initialize the dataloader
        self.dataloader: DataLoader = DataLoader(dataset_path=self.dataset_path, 
                                                 models_info_path='/models_eval/models_info.json',
                                                 sub_dataset_names = ['train_primesense'],
                                                 xyDim=self.xyDim)
        
        
        # Load the object ids and model info
        self.object_ids: list = self.dataloader.get_object_ids()
        self.model_info: dict = self.dataloader.load_model_info(object_ids=self.object_ids)
        
        self.star = StarRepresentation(self.model_info)
        self.dash = DashRepresentation(self.model_info)
        self.destar = DestarRepresentation(self.model_info)
        
    def run(self):
        # for object_id in self.object_ids:
        object_id = self.object_ids[0]
        logger.info(f'Working on object {object_id} of {self.object_ids[-1]}')
        
        # Load the ground truth data
        found_data: list[dict] = self.dataloader.load_gt_data(object_id)
        self.number_of_images = len(found_data)
        logger.info(f'Found data for {self.number_of_images} occurencies of object {object_id}')
        
        
        
        all_rgb = []
        all_star = []
        all_dash = []
        all_destar = []
        all_depth = []
        all_valid_po = []
        all_isvalid = []
        all_segmentation = []
        
        # Generate the data
        logger.info(f'Generating data for object {object_id}')
        for element in tqdm(self.dataloader.generate_data(), total=self.number_of_images):
            # Get the inputs, valid_po, isvalid, depth, segmentation
            inputs, valid_po, isvalid, depth, segmentation = dataset_conversion_layers(element, self.model_info[object_id], self.strides)

            # Calculate the star and dash representations
            valid_star = self.star.calculate(object_id, valid_po)
            valid_dash = self.dash.calculate(object_id ,inputs['rotationmatrix'], valid_po)
            valid_destar = self.destar.calculate(object_id, valid_star, valid_dash, isvalid, inputs['rotationmatrix'])

            all_rgb.append(inputs['rgb'])
            all_star.append(valid_star)
            all_dash.append(valid_dash)
            all_destar.append(valid_destar)
            all_valid_po.append(valid_po)
            all_isvalid.append(isvalid)
            all_depth.append(depth)
            all_segmentation.append(segmentation)
        
        all_rgb = np.squeeze(all_rgb, axis=1)
        all_star = np.squeeze(all_star, axis=1)
        all_dash = np.squeeze(all_dash, axis=1)
        all_destar = np.squeeze(all_destar, axis=1)
        all_valid_po = np.squeeze(all_valid_po, axis=1)
        all_isvalid = np.squeeze(all_isvalid, axis=1)
        all_depth = np.squeeze(all_depth, axis=1)
        all_segmentation = np.squeeze(all_segmentation, axis=1)
        
        logger.warning(f'RGB shape: {all_rgb.shape}, Type: {all_rgb.dtype}')
        logger.warning(f'Star shape: {all_star.shape}, Type: {all_star.dtype}')
        logger.warning(f'Dash shape: {all_dash.shape}, Type: {all_dash.dtype}')
        logger.warning(f'Destar shape: {all_destar.shape}, Type: {all_destar.dtype}')
        logger.warning(f'PO shape: {all_valid_po.shape}, Type: {all_valid_po.dtype}')
        logger.warning(f'Is Valid shape: {all_isvalid.shape}, Type: {all_isvalid.dtype}')
        logger.warning(f'Depth shape: {all_depth.shape}, Type: {all_depth.dtype}')
        logger.warning(f'Segmentation shape: {all_segmentation.shape}, Type: {all_segmentation.dtype}')
        
        pic_numbers = np.random.randint(0, self.number_of_images, 1)
        for picture in pic_numbers:
            f, axarr = plt.subplots(2, 4)
            # Display images and set titles
            axarr[0, 0].set_title('RGB Image')
            axarr[0, 0].imshow(all_rgb[picture])
            axarr[0, 1].set_title('Dash Image')
            axarr[0, 1].imshow(all_dash[picture])
            axarr[0, 2].set_title('Star Image')
            axarr[0, 2].imshow(all_star[picture])
            axarr[0, 3].set_title('Destar Image')
            axarr[0, 3].imshow(all_destar[picture])
            # axarr[0, 3].axis('off')  # Turn off axis for the empty subplot
            axarr[1, 0].set_title('Valid PO')
            axarr[1, 0].imshow(all_valid_po[picture])
            axarr[1, 1].set_title('Is Valid')
            axarr[1, 1].imshow(all_isvalid[picture])
            axarr[1, 2].set_title('Depth Image')
            axarr[1, 2].imshow(all_depth[picture])
            axarr[1, 3].set_title('Segmentation')
            axarr[1, 3].imshow(all_segmentation[picture])
            # Adjust layout to make space for titles
            plt.tight_layout()
            # Show the plot
            plt.show()


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
    