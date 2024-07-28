import base64
from models.model_interface import ModelInterface
from config.gpt4o_mini_config import gpt4o_mini_generation_config
import openai
from config.system_prompt import system_prompt
from openai import AsyncOpenAI


class GPT4OMiniModel(ModelInterface):
    def __init__(self, api_key):
        openai.api_key = api_key
        self.system_prompt = system_prompt
        self.client = AsyncOpenAI(api_key=api_key)


    async def generate_content_async(self, content):
        question = content[0]  # Assuming content[1] is the question
        image_data = content[1]  # Assuming content[0] is the image data

        # Convert to Base64
        base64_encoded = base64.b64encode(image_data['bytes'])
        base64_string = base64_encoded.decode('utf-8')

        response = await self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": [
                    {"type": "text", "text": question},
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/png;base64,{base64_string}"}
                    },
                ]}
            ],
            temperature=gpt4o_mini_generation_config['temperature'],
        )
        
        input_tokens_count = response.usage.prompt_tokens
        output_tokens_count = response.usage.completion_tokens
        
        return response.choices[0].message.content, input_tokens_count, output_tokens_count

    async def review_content_async(self, review_text):
        # Gemini-specific logic for reviewing content
        review_response = await self.model.generate_content_async([review_text])
        return review_response.text.strip().lower() == 'true'