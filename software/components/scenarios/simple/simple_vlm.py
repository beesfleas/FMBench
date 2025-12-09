# components/scenarios/simple/simple_vlm.py
from ..base import BaseScenario
from typing import List, Dict, Any
from PIL import Image

class SimpleVLMScenario(BaseScenario):
    """Ultra-simple VLM test - just verify model runs"""
    
    def get_tasks(self) -> List[Dict[str, Any]]:
        # Create a simple test image
        self._create_test_image()
        
        return [
            {
                "prompt": "What color you see?",
                "image": "test_image.jpg",
                "expected_output": "any response",
                "task_type": "basic_test"
            }
        ]
    
    def _create_test_image(self):
        """Create a simple test image"""
        img = Image.new('RGB', (100, 100), color='red')
        img.save('test_image.jpg')
        print("Created test_image.jpg")
    
    def evaluate(self, task: Dict[str, Any], model_output: str) -> Dict[str, Any]:
        return {
            "task_type": "basic_test",
            "model_output": model_output,
            "has_response": len(model_output.strip()) > 0,
            "success": len(model_output.strip()) > 0
        }