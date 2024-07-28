# models/model_interface.py
from abc import ABC, abstractmethod

class ModelInterface(ABC):
    @abstractmethod
    async def generate_content_async(self, content):
        """Generate content based on the provided input."""
        pass
