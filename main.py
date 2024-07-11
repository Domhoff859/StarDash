import logging
import os
from PIL import Image

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
        
        # Initialize the lists
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
            valid_star = (self.star.calculate(object_id=object_id, po_image=valid_po) / np.sqrt(2) / 2 + 127.5).astype(np.uint8)
            valid_dash = (self.dash.calculate(object_id=object_id ,R=inputs['rotationmatrix'], po_image=valid_po) / 2 + 127.5).astype(np.uint8)
            valid_destar = (self.destar.calculate(object_id=object_id, star=valid_star, dash=valid_dash, isvalid=isvalid, train_R=inputs['rotationmatrix']) / np.sqrt(2) / 2 + 127.5).astype(np.uint8)

            valid_po = (valid_po / np.sqrt(2) / 2 + 127.5).astype(np.uint8)
            
            # Append the data to the lists
            all_rgb.append(inputs['rgb'])
            all_star.append(valid_star)
            all_dash.append(valid_dash)
            all_destar.append(valid_destar)
            all_valid_po.append(valid_po)
            all_isvalid.append(isvalid)
            all_depth.append(depth)
            all_segmentation.append(segmentation)
        
        # Convert the lists to numpy arrays
        all_rgb = np.squeeze(all_rgb, axis=1)
        all_star = np.squeeze(all_star, axis=1)
        all_dash = np.squeeze(all_dash, axis=1)
        all_destar = np.squeeze(all_destar, axis=1)
        all_valid_po = np.squeeze(all_valid_po, axis=1)
        all_isvalid = np.squeeze(all_isvalid, axis=1)
        all_depth = np.squeeze(all_depth, axis=1)
        all_segmentation = np.squeeze(all_segmentation, axis=1)
        
        # Print the shapes and types of the arrays
        logger.warning(f'RGB shape: {all_rgb.shape}, Type: {all_rgb.dtype}')
        logger.warning(f'Star shape: {all_star.shape}, Type: {all_star.dtype}')
        logger.warning(f'Dash shape: {all_dash.shape}, Type: {all_dash.dtype}')
        logger.warning(f'Destar shape: {all_destar.shape}, Type: {all_destar.dtype}')
        logger.warning(f'PO shape: {all_valid_po.shape}, Type: {all_valid_po.dtype}')
        logger.warning(f'Is Valid shape: {all_isvalid.shape}, Type: {all_isvalid.dtype}')
        logger.warning(f'Depth shape: {all_depth.shape}, Type: {all_depth.dtype}')
        logger.warning(f'Segmentation shape: {all_segmentation.shape}, Type: {all_segmentation.dtype}')
        
        # Display the images
        pic_numbers = np.random.randint(0, self.number_of_images, 10)
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
            
    def test(self):
        obj_id = '1'
        star_folder_path = os.path.join(self.dataset_path, 'xyz_data', obj_id, 'star')
        dash_folder_path = os.path.join(self.dataset_path, 'xyz_data', obj_id, 'dash')
        nocs_folder_path = os.path.join(self.dataset_path, 'xyz_data', obj_id, 'nocs')
        mask_folder_path = os.path.join(self.dataset_path, 'xyz_data', obj_id, 'mask')
        train_R_folder_path = os.path.join(self.dataset_path, 'xyz_data', obj_id, 'cam_R_m2c')
        
        # Pick one random image from the star_path
        star_files = os.listdir(star_folder_path)
        random_file: str = np.random.choice(star_files)
        
        star_path = os.path.join(star_folder_path, random_file)
        dash_path = os.path.join(dash_folder_path, random_file)
        nocs_path = os.path.join(nocs_folder_path, random_file)
        mask_path = os.path.join(mask_folder_path, random_file)
        train_R_path = os.path.join(train_R_folder_path, random_file.replace('.png', '.npy'))

        star_image = np.array(Image.open(star_path), dtype=np.uint8)[:,:,::-1]
        dash_image = np.array(Image.open(dash_path), dtype=np.uint8)[:,:,::-1]
        nocs_image = np.array(Image.open(nocs_path), dtype=np.uint8)
        mask_image = np.array(Image.open(mask_path), dtype=np.float64)[...,np.newaxis]
        train_R = np.load(train_R_path)
        
        # Calculate the destar image
        destar_image = self.destar.calculate(object_id=obj_id, star=star_image[np.newaxis, ...], dash=dash_image[np.newaxis, ...], isvalid=mask_image, train_R=train_R[np.newaxis,...])
        destar_image = np.squeeze(destar_image, axis=0)
        
        valid_star = self.star.calculate(object_id=obj_id, po_image=(nocs_image.astype(np.float64) - 127.5)[np.newaxis, ...])
        valid_dash = self.dash.calculate(object_id=obj_id ,R=train_R[np.newaxis,...], po_image=(nocs_image.astype(np.float64) - 127.5)[np.newaxis, ...])
        valid_destar = self.destar.calculate(object_id=obj_id, star=valid_star, dash=valid_dash, isvalid=mask_image, train_R=train_R[np.newaxis,...])

        valid_star = np.array(valid_star / np.sqrt(2) / 2 + 127.5, dtype=np.uint8)
        valid_dash = np.array(valid_dash / np.sqrt(2) / 2 + 127.5, dtype=np.uint8)
        
        logger.critical(f'Max Error = {np.max(np.abs(nocs_image - valid_destar))}')
        logger.critical(f'Min Error = {np.min(np.abs(nocs_image - valid_destar))}')
        logger.critical(f'Mean Error = {np.mean(np.abs(nocs_image - valid_destar))}')
        
        valid_star = np.squeeze(valid_star, axis=0)
        valid_dash = np.squeeze(valid_dash, axis=0)
        valid_destar = np.squeeze(valid_destar, axis=0)
        
        logger.critical(f'Max Nocs: {np.max(np.array(nocs_image))}')
        logger.critical(f'Min Nocs: {np.min(np.array(nocs_image))}')
        logger.critical(f'Max Destar: {np.max(destar_image)}')
        logger.critical(f'Min Destar: {np.min(destar_image)}')
        logger.critical(f'Max Valid Destar: {np.max(valid_destar)}')
        logger.critical(f'Min Valid Destar: {np.min(valid_destar)}')
        logger.critical(f'Max Valid Star: {np.max(valid_star)}')
        logger.critical(f'Min Valid Star: {np.min(valid_star)}')
        logger.critical(f'Max Valid Dash: {np.max(valid_dash)}')
        logger.critical(f'Min Valid Dash: {np.min(valid_dash)}')
        
        # Display the images
        f, axarr = plt.subplots(2, 4)
        # Display images and set titles
        axarr[0, 0].set_title('Valid Star Image')
        axarr[0, 0].imshow(valid_star)
        axarr[1, 0].set_title('Star Image')
        axarr[1, 0].imshow(star_image)
        axarr[0, 1].set_title('Valid _Dash Image')
        axarr[0, 1].imshow(valid_dash)
        axarr[1, 1].set_title('Dash Image')
        axarr[1, 1].imshow(dash_image)
        axarr[0, 3].set_title('NOCS Image')
        axarr[0, 3].imshow(nocs_image)
        axarr[1, 3].axis('off')
        axarr[1, 2].set_title('Destar Image')
        axarr[1, 2].imshow(destar_image)
        axarr[0, 2].set_title('Valid Destar Image')
        axarr[0, 2].imshow(valid_destar)
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
    # stardash.run()
    stardash.test()
    