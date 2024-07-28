# models/gemini_model.py
from models.model_interface import ModelInterface
from vertexai.generative_models import GenerativeModel
from config.gemini_config import gemini_generation_config, gemini_safety_settings
from config.system_prompt import system_prompt
from vertexai.generative_models import GenerativeModel, Image, Part


class GeminiModel(ModelInterface):
    def __init__(self, model_name):
        self.model = GenerativeModel(
            model_name,
            generation_config=gemini_generation_config,
            safety_settings=gemini_safety_settings,
            system_instruction=system_prompt
        )

    async def generate_content_async(self, content):
        image = Part.from_data(
            mime_type="image/jpeg",
            data=content[1]["bytes"]
        )
        payload = [image, content[0]]      
        # Gemini-specific logic for generating content
        response = await self.model.generate_content_async(payload)
        input_tokens_count = response.usage_metadata.prompt_token_count
        output_tokens_count = response.usage_metadata.candidates_token_count

        return response.text, input_tokens_count, output_tokens_count

    async def review_content_async(self, review_text):
        # Gemini-specific logic for reviewing content
        review_response = await self.model.generate_content_async([review_text])
        return review_response.text.strip().lower() == 'true'