import os
import json
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm


class DataLoader():
    def __init__(self) -> None:
        pass
    
    def load_model_info(self):
        pass
    
    def load_gt_data(self, root_dirs, object_id) -> list:
        found_data = []
        
        for rd in root_dirs:
            pass