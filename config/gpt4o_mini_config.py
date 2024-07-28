from config.base_config import generation_config

gpt4o_mini_generation_config = {
    **generation_config,  # Inherit base settings
    "max_tokens": 1000,  # Model-specific setting
}