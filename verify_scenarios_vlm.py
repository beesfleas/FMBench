import logging
from components.scenarios.common_vlm_scenarios import VQAScenario

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    verify_vqa()

