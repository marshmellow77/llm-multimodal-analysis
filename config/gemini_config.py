from config.base_config import generation_config
from vertexai.generative_models import HarmBlockThreshold, HarmCategory

gemini_generation_config = {
    **generation_config,  # Inherit base settings
    "max_output_tokens": 1000,  # Model-specific setting
}

gemini_safety_settings = {
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
}