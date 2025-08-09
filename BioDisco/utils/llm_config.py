# utils/llm_config.py
import autogen

DEFAULT_MODEL_NAME = "mistral-large-2411"

def get_config_list(model_name: str = DEFAULT_MODEL_NAME):
    return autogen.config_list_from_models(model_list=[model_name])

config_list = get_config_list(DEFAULT_MODEL_NAME)

KEYWORD_CONFIG = {
    "chat_model": DEFAULT_MODEL_NAME,
    "cache_seed": 42,
    "temperature": 0.3,
    "config_list": config_list,
    "timeout": 540000,
    "max_output_tokens": 256
}

DOMAIN_CONFIG = {
    "chat_model": DEFAULT_MODEL_NAME,
    "cache_seed": 42,
    "temperature": 0.2,
    "config_list": config_list,
    "timeout": 540000,
    "max_output_tokens": 128
}

KG_AGENT_CONFIG = {
    "chat_model": DEFAULT_MODEL_NAME,
    "cache_seed": 42,
    "temperature": 0.3,
    "config_list": config_list,
    "timeout": 540000,
    "max_output_tokens": 800
}

PLANNER_CONFIG = {
    "chat_model": DEFAULT_MODEL_NAME,
    "cache_seed": 42,
    "temperature": 0.4,
    "config_list": config_list,
    "timeout": 540000,
    "max_output_tokens": 350
}

BACKGROUND_SUMMARISER_CONFIG = {
    "chat_model": DEFAULT_MODEL_NAME,
    "cache_seed": 42,
    "temperature": 0.3,
    "config_list": config_list,
    "timeout": 540000,
    "max_output_tokens": 320
}

SCIENTIST_AGENT_CONFIG = {
    "chat_model": DEFAULT_MODEL_NAME,
    "cache_seed": 42,
    "temperature": 0.5,
    "config_list": config_list,
    "timeout": 540000,
    "max_output_tokens": 96
}

PUBMED_AGENT_CONFIG = {
    "chat_model": DEFAULT_MODEL_NAME,
    "cache_seed": 42,
    "temperature": 0.2,
    "config_list": config_list,
    "timeout": 540000,
    "max_output_tokens": 900
}

CRITIC_AGENT_CONFIG = {
    "chat_model": DEFAULT_MODEL_NAME,
    "cache_seed": 42,
    "temperature": 0.3,
    "config_list": config_list,
    "timeout": 540000,
    "max_output_tokens": 512
}

REVISION_AGENT_CONFIG = {
    "chat_model": DEFAULT_MODEL_NAME,
    "cache_seed": 42,
    "temperature": 0.2,
    "config_list": config_list,
    "timeout": 540000,
    "max_output_tokens": 128
}

REFINE_AGENT_CONFIG = {
    "chat_model": DEFAULT_MODEL_NAME,
    "cache_seed": 42,
    "temperature": 0.5,
    "config_list": config_list,
    "timeout": 540000,
    "max_output_tokens": 128
}

DECISION_AGENT_CONFIG = {
    "chat_model": DEFAULT_MODEL_NAME,
    "cache_seed": 42,
    "temperature": 0.2,
    "config_list": config_list,
    "timeout": 540000,
    "max_output_tokens": 128
}