import os
import json
import logging
from typing import Generator
from pathlib import Path
from random import sample

import numpy as np
from PIL import Image
from tqdm import tqdm

logger = logging.getLogger('Dataloader')

class DataLoader():
    def __init__(self, dataset_path: str, models_info_path: str, 
                 sub_dataset_names: list[str], 
                 xyDim: int,
                 dataset_multiplier: int = 1,
                 batch_size: int = 1,
                 random: bool = False, 
                 sigma: float = 0.2, 
                 test_mode: bool = False) -> None:
        # Set the dataset paths
        self.dataset_path: str = dataset_path
        self.models_info_path: str = models_info_path
        
        # Set the parameters
        self.xyDim: int = xyDim
        self.dataset_multiplier: int = dataset_multiplier
        self.batch_size: int = batch_size
        self.random: bool = random
        self.sigma: float = sigma
        self.test_mode: bool = test_mode
        
        # Initialize the found data
        self.found_data: list[dict] = []
        
        
        # Get all folders in the dataset path
        folders: list[str] = [f.path for f in os.scandir(self.dataset_path) if f.is_dir()]
        
        # Filter the folders based on the datasets parameter
        folders: list[Path] = [Path(f) for f in folders if any([d in f for d in sub_dataset_names])]
        
        # Check if all the folders exist
        if len(folders) != len(sub_dataset_names):
            logger.error(f'Expected {len(sub_dataset_names)} folders but found {len(folders)}')
            logger.error(f'Expected folders: {sub_dataset_names}')
            logger.error(f'Found folders: {folders}')
        
        # Set the dataset names
        self.sub_dataset_paths: list[Path] = folders
        logger.debug(f'Found dataset paths: {self.sub_dataset_paths}')
    
    def open_annotator(self, name: str) -> dict:
        """
        Check if the file exists open it and return the json data.

        Args:
            name (str): Path to the file.

        Returns:
            dict: Json data from the file.
        """
        # Check if the file exists
        assert(Path(name).is_file())
        
        # Open the file and return the json data
        with open(name) as f:
            return json.load(f)
        
    def get_object_ids(self) -> list:
        """
        Get the object ids from the dataset.

        Returns:
            list: List of object ids.
        """
        # Construct the path to load the models_info.json file
        path = Path(f'{self.dataset_path}{self.models_info_path}')
        logger.debug(f"Path to load : {path}")

        # Load the object ids from the models_info.json file
        with open(path, 'r') as f:
            jsondata: dict = json.load(f)
            return list(jsondata.keys())
        
    def load_model_info(self, object_ids: list) -> dict:
            """
            Load all model infos from the dataset.

            Args:
                models_info_path (str, optional): Path to the models_info.json file. Defaults to '/models_eval/models_info.json'.

            Returns:
                dict: Dictionary with all model infos.
            """
            # Construct the path to load the models_info.json file
            path = Path(f'{self.dataset_path}{self.models_info_path}')
            logger.debug(f"Path to load : {path}")
            
            # Load the json data from the models_info.json file
            with open(path, 'r') as f:
                jsondata = json.load(f)
                model_info = {}
            
                # Iterate over each object id
                for object_id in object_ids:
                    object_info = {}
                    
                    key = str(object_id)
                    assert(key in jsondata)

                    # Extract the relevant information for the object
                    object_info["diameter"] = jsondata[key]["diameter"]
                    object_info["mins"] = np.array([jsondata[key]["min_x"],jsondata[key]["min_y"],jsondata[key]["min_z"]])
                    object_info["maxs"] = np.array([jsondata[key]["size_x"],jsondata[key]["size_y"],jsondata[key]["size_z"]]) +  object_info["mins"]

                    # Check if discrete symmetries are present
                    if "symmetries_discrete" in jsondata[key]:
                        object_info["symmetries_discrete"] = [np.array(_).reshape((4,4)) for _ in jsondata[key]["symmetries_discrete"]]
                    else:
                        object_info["symmetries_discrete"] = []

                    # Check if continuous symmetries are present
                    object_info["symmetries_continuous"] = "symmetries_continuous" in jsondata[key]
                        
                    model_info[object_id] = object_info
                
            return model_info
    
    def load_gt_data(self, object_id: str) -> list:
        """
        Load the ground truth data for the object id.

        Args:
            object_id (str): Object id to load the data for.

        Returns:
            list: List of ground truth data.
        """
        found_data: list[dict] = []
        for rd in self.sub_dataset_paths:
            for root, sub_dirs, files in os.walk(rd):
                logger.info(f'Loading data from {root}')
                for sd in tqdm(sub_dirs):
                    dir: str = f'{root}/{sd}'
                    
                    # Open the required annotation files
                    scene_gt: dict = self.open_annotator(f'{dir}/scene_gt.json')
                    scene_gt_info: dict = self.open_annotator(f'{dir}/scene_gt_info.json')
                    scene_camera: dict = self.open_annotator(f'{dir}/scene_camera.json')
                    
                    # Check if the lengths of the annotation files are the same
                    assert(len(scene_gt) == len(scene_gt_info))
                    assert(len(scene_gt) == len(scene_camera))
                    
                    # Iterate over each key-value pair in scene_gt
                    for key, gt_values in scene_gt.items():
                        for vi, v in enumerate(gt_values):
                            # Check if the object_id matches and visibility fraction is greater than 0.1
                            if str(v["obj_id"]) == object_id and scene_gt_info[key][vi]["visib_fract"] > 0.1: 
                            
                                new_data: dict = {}
                                new_data['root'] = dir
                                new_data['file_name'] = "{:06d}".format(int(key))
                                new_data['oi_name'] = "{:06d}".format(vi)
                                new_data['cam_R_m2c'] = np.array(v["cam_R_m2c"]).reshape((3,3))
                                new_data['cam_t_m2c'] = np.array(v["cam_t_m2c"])
                                
                                bbox_obj: list = scene_gt_info[key][vi]["bbox_obj"]
                                new_data['bbox_start'] =np.array(bbox_obj[:2])
                                new_data['bbox_dims'] = np.array(bbox_obj[2:])
                                
                                new_data['cam_K'] = np.array(scene_camera[key]["cam_K"]).reshape((3,3))
                                new_data['depth_scale'] = scene_camera[key]["depth_scale"]
                                
                                new_data['visib_fract'] = scene_gt_info[key][vi]["visib_fract"]
                                
                                found_data.append(new_data)
                break
            
        self.found_data = found_data
        return found_data
    
    def load_training_data_item(self, data_element: dict) -> tuple[np.array]:
        """
        Load the data item from the data element.

        Args:
            data_element (dict): Data element to load the data from.

        Returns:
            tuple[8x np.array]: Tuple of loaded data consisting of image, depth image, segmentation, camera matrix, camera rotation, camera translation, bounding box start and bounding box dimensions.
        """
        img: np.array = np.array(Image.open(f'{data_element["root"]}/rgb/{data_element["file_name"]}{".png" if "primesense" in data_element["root"] else ".jpg"}'))
        depthimg: np.array = np.array(Image.open(f'{data_element["root"]}/depth/{data_element["file_name"]}.png'), np.float32)
        depthimg *= data_element["depth_scale"]
            
        seg = np.array(Image.open(f'{data_element["root"]}/mask_visib/{data_element["file_name"]}_{data_element["oi_name"]}.png'))
        return img, depthimg, seg, np.array(data_element["cam_K"]), np.array(data_element["cam_R_m2c"]), np.array(data_element["cam_t_m2c"]), np.array(data_element['bbox_start']), np.array(data_element['bbox_dims'])
    
    def load_test_data_item(self, data_element: dict) -> tuple[np.array]:
        """
        Load the data item from the data element.

        Args:
            data_element (dict): Data element to load the data from.

        Returns:
            tuple[5x np.array]: Tuple of loaded data consisting of image, depth image, camera matrix, bounding box start and bounding box dimensions.
        """
        img: np.array = np.array(Image.open(f'{data_element["root"]}/rgb/{data_element["file_name"]}{".png" if "primesense" in data_element["root"] else ".jpg"}'))
        depthimg: np.array = np.array(Image.open(f'{data_element["root"]}/depth/{data_element["file_name"]}.png'), np.float32)
        depthimg *= data_element["depth_scale"]
            
        return img, depthimg, data_element["cam_K"], data_element['bbox_start'], data_element['bbox_dims']
    
    def extract_training_item(self, loaded_data_element: tuple[np.array]) -> tuple[np.array]:
        """
        Extract the training item from the loaded data element.

        Args:
            loaded_data_element (tuple[np.array]): Loaded data element.

        Returns:
            tuple[np.array]: Tuple of extracted data consisting of image, depth image, segmentation, camera matrix, camera rotation, camera translation, bounding box start and bounding box dimensions.
        """
        # Extract the variables from the extracted_data_element
        img, depth, seg, cam_K, R, t, bbs, bbd = loaded_data_element
        
        # Calculate the scale and new bounding box coordinates
        scale_diff = np.maximum(np.random.normal(1, self.sigma), 0.6)
        scale = np.max(bbd) / self.xyDim * scale_diff
        new_bbs = bbs + (bbd - np.max(bbd)) / 2 - (scale_diff- 1) * np.max(bbd) / 2  + np.random.normal(0, self.sigma * np.max(bbd) / 2., 2)
        
        # Calculate the transformation matrix and coordinate matrix
        transformation = [scale, 0, new_bbs[0], 0.0, scale,  new_bbs[1], 0.0, 0.0]
        coord_K = np.stack([np.array([scale,scale]), new_bbs])
        
        # Convert numpy arrays to PIL images
        img_pil = Image.fromarray(img)
        depth_pil = Image.fromarray(depth)
        seg_pil = Image.fromarray(seg)

        # Apply transformation to images
        transformed_img_pil = img_pil.transform((self.xyDim, self.xyDim), Image.AFFINE, transformation, resample=Image.BILINEAR)
        transformed_depth_pil = depth_pil.transform((self.xyDim, self.xyDim), Image.AFFINE, transformation, resample=Image.BILINEAR)
        transformed_seg_pil = seg_pil.transform((self.xyDim, self.xyDim), Image.AFFINE, transformation, resample=Image.BILINEAR)

        # Convert PIL images back to numpy arrays
        transformed_img = np.array(transformed_img_pil)
        transformed_depth = np.array(transformed_depth_pil)
        transformed_seg = np.array(transformed_seg_pil)
        
        return transformed_img, transformed_depth, transformed_seg, cam_K, R, t, coord_K
    
    def extract_test_item(self, loaded_data_element: tuple[np.array]) -> tuple[np.array]:
        """
        Extract the test item from the loaded data element.

        Args:
            loaded_data_element (tuple[np.array]): Loaded data element.

        Returns:
            tuple[np.array]: Tuple of extracted data consisting of image, depth image, camera matrix, bounding box start and bounding box dimensions.
        """
        # Extract the variables from the extracted_data_element
        img, depth, cam_K, bbs, bbd = loaded_data_element
        
        # Calculate the scale and new bounding box coordinates
        scale = np.max(bbd) / self.xyDim
        new_bbs = bbs + (bbd - np.max(bbd)) / 2
        
        # Calculate the transformation matrix and coordinate matrix
        transformation = [scale, 0, new_bbs[0], 0.0, scale,  new_bbs[1], 0.0, 0.0]
        coord_K = np.stack([np.array([scale,scale]), new_bbs])
        
        # Convert numpy arrays to PIL images
        img_pil = Image.fromarray(img)
        depth_pil = Image.fromarray(depth)

        # Apply transformation to images
        transformed_img_pil = img_pil.transform((self.xyDim, self.xyDim), Image.AFFINE, transformation, resample=Image.BILINEAR)
        transformed_depth_pil = depth_pil.transform((self.xyDim, self.xyDim), Image.AFFINE, transformation, resample=Image.BILINEAR)

        # Convert PIL images back to numpy arrays
        transformed_img = np.array(transformed_img_pil)
        transformed_depth = np.array(transformed_depth_pil)
        
        return transformed_img, transformed_depth, cam_K, coord_K
    
    def batch_data(self, data_element: dict) -> list[dict]:
        """
        Batch the data element.

        Args:
            data_element (dict): Data element to batch.

        Returns:
            list[dict]: List of batched data elements.
        """
        # Initialize an empty list to store the batched loaded data
        batched_loaded_data = []
        
        # Check if the test mode is enabled
        if self.test_mode:
            # Load the test data item
            loaded_data = self.load_test_data_item(data_element)
            
            # Batch the loaded data for the specified batch size
            for _ in range(self.batch_size):
                batched_loaded_data.append(self.extract_test_item(loaded_data))
        else:
            # Load the training data item
            loaded_data = self.load_training_data_item(data_element)
        
            # Batch the loaded data for the specified batch size
            for _ in range(self.batch_size):
                batched_loaded_data.append(self.extract_training_item(loaded_data))
            
        # Return the batched loaded data
        return batched_loaded_data
    
    def generate_data(self) -> Generator[dict, None, None]:
        """
        Generate the data for training or testing.

        Yields:
            dict: Data element.
        """
        # Enlarge the found data by the dataset multiplier
        data: list[dict] = self.found_data * self.dataset_multiplier
        
        # Shuffle the data if random is enabled
        if self.random:
            data = sample(data, k=len(data))
            
        # Iterate over each data element
        for d in data:
            batched_d = self.batch_data(d)
            
            # Yield the batched data
            for elem in batched_d:
                yield elem
            