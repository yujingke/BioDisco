# gpt-40-mini

import autogen

# Load configuration
config_list_gpt4o_mini = autogen.config_list_from_models(model_list=["gpt-4o-mini"])

config_list_turbo_alias_gpt_4o_mini = autogen.config_list_from_models(model_list=["gpt-4o-mini"])

gpt4o_mini_config = {
    "chat_model": "gpt-4o-mini",                
    "cache_seed": 42,
    "temperature": 0.8,
    "config_list": config_list_gpt4o_mini,
    "timeout": 540000,
    "max_output_tokens": 1000      
}

gpt4o_mini_config_graph = {
    "chat_model": "gpt-4o-mini",
    "cache_seed": 42,
    "temperature": 0.8,
    "config_list": config_list_gpt4o_mini,
    "timeout": 540000,
    "max_output_tokens": 1500
}

gpt4turbo_mini_config = {
    "chat_model": "gpt-4o-mini",
    "cache_seed": 42,
    "temperature": 0.8,
    "config_list": config_list_turbo_alias_gpt_4o_mini,
    "timeout": 540000,
    "max_output_tokens": 1024
}

gpt4turbo_mini_config_graph = {
    "chat_model": "gpt-4o-mini",
    "cache_seed": 42,
    "temperature": 0.8,
    "config_list": config_list_turbo_alias_gpt_4o_mini,
    "timeout": 540000,
    "max_output_tokens": 2000
}