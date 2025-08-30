import os

def set_offline_mode(config_offline: dict):
    if config_offline.get('hf_hub_offline', False):
        os.environ['HF_HUB_OFFLINE'] = '1'
    if config_offline.get('transformers_offline', False):
        os.environ['TRANSFORMERS_OFFLINE'] = '1'

def unset_offline_mode():
    if 'HF_HUB_OFFLINE' in os.environ:
        del os.environ['HF_HUB_OFFLINE']
    if 'TRANSFORMERS_OFFLINE' in os.environ:
        del os.environ['TRANSFORMERS_OFFLINE']