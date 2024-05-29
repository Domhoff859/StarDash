import logging

from src.dash import DashRepresentation
from src.star import StarRepresentation
from src.dataloader import DataLoader
from logger.custom_logging import configure_logger

# =============================================================================
# Configure logger
configure_logger("logger/logging.yaml")
logger = logging.getLogger(__name__)

# =============================================================================
class Main:
    def __init__(self, dataset_path:str) -> None:
        
        self.dataset_path = dataset_path
        
        self.dataloader = DataLoader()
        self.starrepresentation = StarRepresentation()
        self.dashrepresentation = DashRepresentation()

    def run(self):
        logger.info("Running main")

# =============================================================================

dataset_path = "/home/domin/Documents/Datasets/tless"


if __name__ == "__main__":
    main = Main(dataset_path=dataset_path)
    main.run()