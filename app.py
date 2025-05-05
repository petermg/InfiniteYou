# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates. All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import gc
import os
import logging
import glob
import re
import json

import gradio as gr
import pillow_avif
import torch
from huggingface_hub import snapshot_download
from pillow_heif import register_heif_opener
from safetensors.torch import load_file

from pipelines.pipeline_infu_flux import InfUFluxPipeline

# Set up logging to both file and console
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log', mode='a'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Parse command-line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="InfiniteYou-FLUX Gradio Demo")
    parser.add_argument('--cuda_device', default=0, type=int, help="CUDA device index")
    return parser.parse_args()

args = parse_args()

# Set CUDA device
torch.cuda.set_device(args.cuda_device)

# Register HEIF support for Pillow
register_heif_opener()

# Flag to track if models have been downloaded
models_downloaded = False

class ModelVersion:
    STAGE_1 = "sim_stage1"
    STAGE_2 = "aes_stage2"
    DEFAULT_VERSION = STAGE_2
    
ENABLE_ANTI_BLUR_DEFAULT = False
ENABLE_REALISM_DEFAULT = False
QUANTIZE_8BIT_DEFAULT = True
CPU_OFFLOAD_DEFAULT = True
OUTPUT_DIR = "./results"
MAX_LORA_FIELDS = 5  # Maximum number of LoRA fields to display

# Available built-in LoRAs
AVAILABLE_LORAS = {
    "realism": "./models/InfiniteYou/supports/optional_loras/flux_realism_lora.safetensors",
    "anti-blur": "./models/InfiniteYou/supports/optional_loras/flux_anti_blur_lora.safetensors",
}

loaded_pipeline_config = {
    "model_version": "aes_stage2",
    "loras": [],
    "quantize_8bit": False,
    "cpu_offload": False,
    'pipeline': None
}

def download_models():
    global models_downloaded
    if not models_downloaded:
        logger.info("Downloading models...")
        snapshot_download(repo_id='ByteDance/InfiniteYou', local_dir='./models/InfiniteYou', local_dir_use_symlinks=False)
        try:
            snapshot_download(repo_id='black-forest-labs/FLUX.1-dev', local_dir='./models/FLUX.1-dev', local_dir_use_symlinks=False)
        except Exception as e:
            logger.error(f"Failed to download FLUX.1-dev: {e}")
            print('\nYou are downloading `black-forest-labs/FLUX.1-dev` to `./models/FLUX.1-dev` but failed. '
                  'Please accept the agreement and obtain access at https://huggingface.co/black-forest-labs/FLUX.1-dev. '
                  'Then, use `huggingface-cli login` and your access tokens at https://huggingface.co/settings/tokens to authenticate. '
                  'After that, run the code again.')
            print('\nYou can also download it manually from HuggingFace and put it in `./models/InfiniteYou`, '
                  'or you can modify `base_model_path` in `app.py` to specify the correct path.')
            raise Exception("Model download failed")
        # Verify built-in LoRA files exist
        for lora_name, lora_path in AVAILABLE_LORAS.items():
            if not os.path.exists(lora_path):
                logger.error(f"Built-in LoRA file missing: {lora_path}")
                raise FileNotFoundError(f"Built-in LoRA file missing: {lora_path}")
        models_downloaded = True
        logger.info("Models and LoRAs downloaded successfully.")

def prepare_pipeline(model_version, loras, quantize_8bit, cpu_offload):
    logger.info(f"Preparing pipeline with model_version={model_version}, loras={loras}, quantize_8bit={quantize_8bit}, cpu_offload={cpu_offload}")
    
    # Force reload if LoRAs differ to ensure they are applied
    if (
        loaded_pipeline_config['pipeline'] is not None
        and loaded_pipeline_config["loras"] == loras
        and loaded_pipeline_config["quantize_8bit"] == quantize_8bit
        and loaded_pipeline_config["cpu_offload"] == cpu_offload
        and model_version == loaded_pipeline_config["model_version"]
    ):
        logger.info("Reusing existing pipeline.")
        return loaded_pipeline_config['pipeline']
    
    loaded_pipeline_config["loras"] = loras
    loaded_pipeline_config["quantize_8bit"] = quantize_8bit
    loaded_pipeline_config["cpu_offload"] = cpu_offload
    loaded_pipeline_config["model_version"] = model_version

    pipeline = loaded_pipeline_config['pipeline']
    if pipeline is None or pipeline.model_version != model_version:
        logger.info(f'Switching model to {model_version}')
        if pipeline is not None:
            logger.info("Deleting existing pipeline.")
            del pipeline
            del loaded_pipeline_config['pipeline']
            gc.collect()
            torch.cuda.empty_cache()

        model_path = f'./models/InfiniteYou/infu_flux_v1.0/{model_version}'
        logger.info(f'Loading model from {model_path}')

        pipeline = InfUFluxPipeline(
            base_model_path='./models/FLUX.1-dev',
            infu_model_path=model_path,
            insightface_root_path='./models/InfiniteYou/supports/insightface',
            image_proj_num_tokens=8,
            infu_flux_version='v1.0',
            model_version=model_version,
            quantize_8bit=quantize_8bit,
            cpu_offload=cpu_offload,
        )

        loaded_pipeline_config['pipeline'] = pipeline

    # Unload previous LoRA weights
    try:
        pipeline.pipe.unload_lora_weights()
        logger.info("Unloaded previous LoRA weights.")
    except Exception as e:
        logger.error(f"Failed to unload LoRA weights: {e}")

    # Load new LoRAs
    if loras:
        logger.info(f"Loading LoRAs: {loras}")
        for lora in loras:
            lora_path = lora[0]
            if not os.path.exists(lora_path):
                logger.error(f"LoRA path does not exist: {lora_path}")
                raise FileNotFoundError(f"LoRA path does not exist: {lora_path}")
            if not lora_path.endswith('.safetensors'):
                logger.error(f"LoRA file must be a .safetensors file: {lora_path}")
                raise ValueError(f"LoRA file must be a .safetensors file: {lora_path}")
        try:
            pipeline.load_loras(loras)
            logger.info("LoRAs loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load LoRAs: {e}")
            raise RuntimeError(f"Failed to load LoRAs: {e}")

    return pipeline

def generate_image(
    input_image, 
    control_image, 
    prompt, 
    seed, 
    width,
    height,
    guidance_scale, 
    num_steps, 
    infusenet_conditioning_scale, 
    infusenet_guidance_start,
    infusenet_guidance_end,
    lora_state,
    quantize_8bit,
    cpu_offload,
    model_version
):
    # Convert lora_state to loras format
    loras = convert_lora_state_to_loras(lora_state)
    logger.info(f"Converted lora_state to loras: {loras}")

    # Download models if not already done
    download_models()

    # Prepare pipeline with user-selected options
    pipeline = prepare_pipeline(
        model_version=model_version,
        loras=loras,
        quantize_8bit=quantize_8bit,
        cpu_offload=cpu_offload
    )

    if seed == 0:
        seed = torch.seed() & 0xFFFFFFFF
        logger.info(f"Generated random seed: {seed}")

    try:
        image = pipeline(
            id_image=input_image,
            prompt=prompt,
            control_image=control_image,
            seed=seed,
            width=width,
            height=height,
            guidance_scale=guidance_scale,
            num_steps=num_steps,
            infusenet_conditioning_scale=infusenet_conditioning_scale,
            infusenet_guidance_start=infusenet_guidance_start,
            infusenet_guidance_end=infusenet_guidance_end,
            cpu_offload=cpu_offload,
        )
        # Save the generated image
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        index = len(os.listdir(OUTPUT_DIR))
        prompt_name = ''.join(c if c.isalnum() or c in '_-' else '_' for c in prompt[:50].replace(' ', '_')).strip('_')
        out_name = f"{index:05d}_{prompt_name}_seed{seed}.png"
        out_path = os.path.join(OUTPUT_DIR, out_name)
        image.save(out_path)
        logger.info(f"Image saved to {out_path}")
        return gr.update(value=image, label=f"Generated Image, seed={seed}, saved to {out_path}"), str(seed)
    except Exception as e:
        logger.error(f"Error during image generation: {e}")
        gr.Error(f"An error occurred: {e}")
        return gr.update(), str(seed)

def generate_examples(id_image, control_image, prompt_text, seed, lora_state, model_version):
    # Convert lora_state to loras format
    loras = convert_lora_state_to_loras(lora_state)
    logger.info(f"Generating example with loras: {loras}")
    # Use default values for quantize_8bit and cpu_offload for examples
    return generate_image(
        id_image, control_image, prompt_text, seed, 864, 1152, 3.5, 30, 1.0, 0.0, 1.0,
        lora_state, QUANTIZE_8BIT_DEFAULT, CPU_OFFLOAD_DEFAULT, model_version
    )

def convert_lora_state_to_loras(lora_state):
    loras = []
    for lora in lora_state:
        if lora["name"] != "None" and lora["path"]:
            # Ensure weight is a float
            try:
                weight = float(lora["weight"])
            except (ValueError, TypeError):
                logger.warning(f"Invalid weight {lora['weight']} for LoRA {lora['name']}, defaulting to 1.5")
                weight = 1.5
            loras.append([lora["path"], lora["name"], weight])
    logger.debug(f"Converted lora_state: {lora_state} to loras: {loras}")
    return loras

def read_safetensors_header(file_path):
    """Read the header of a safetensors file and attempt to parse JSON metadata."""
    try:
        with open(file_path, 'rb') as f:
            # Read the first 8 bytes to get the header length
            header_len_bytes = f.read(8)
            if len(header_len_bytes) != 8:
                logger.warning(f"Failed to read header length from {file_path}")
                return {}
            header_len = int.from_bytes(header_len_bytes, byteorder='little')
            # Read the header data
            header_data = f.read(header_len)
            if len(header_data) != header_len:
                logger.warning(f"Incomplete header read from {file_path}: expected {header_len}, got {len(header_data)}")
                return {}
            # Try decoding as UTF-8, replacing invalid characters
            try:
                header_str = header_data.decode('utf-8')
            except UnicodeDecodeError:
                header_str = header_data.decode('utf-8', errors='replace')
            logger.debug(f"Raw header from {file_path}: {header_str[:1000]}...")
            # Parse JSON
            try:
                header_json = json.loads(header_str)
                metadata = header_json.get("__metadata__", {})
                if not isinstance(metadata, dict):
                    logger.warning(f"Metadata in {file_path} is not a dictionary: {metadata}")
                    metadata = {"raw_metadata": str(metadata)}
                metadata = {k: str(v) for k, v in metadata.items()}
                logger.info(f"Parsed metadata from header in {file_path}: {metadata}")
                return metadata
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse JSON header in {file_path}: {e}")
                return {}
    except Exception as e:
        logger.error(f"Error reading safetensors header from {file_path}: {e}")
        return {}

def list_lora_files(directory):
    """Scan the specified directory for .safetensors files and return their filenames, full paths, and metadata."""
    logger.info(f"Listing LoRA files in directory: {directory}")
    if not directory:
        logger.warning("No LoRA directory specified, returning empty list")
        return []
    try:
        safetensors_files = glob.glob(os.path.join(directory, "*.safetensors"))
        lora_list = []
        for f in safetensors_files:
            try:
                # First attempt to load metadata using safetensors.torch
                safetensors_data = load_file(f, device="cpu")
                file_keys = list(safetensors_data.keys())
                logger.debug(f"Keys in {f}: {file_keys}")
                
                # Try standard __metadata__ key
                metadata = safetensors_data.get("__metadata__", {})
                if not metadata:
                    # Check for alternative metadata keys
                    for key in file_keys:
                        if "metadata" in key.lower() or "info" in key.lower():
                            try:
                                metadata = safetensors_data[key]
                                if isinstance(metadata, str):
                                    metadata = json.loads(metadata)
                                logger.debug(f"Found metadata in key {key}: {metadata}")
                                break
                            except Exception as e:
                                logger.warning(f"Failed to parse metadata from key {key} in {f}: {e}")
                
                # If no metadata found, try reading the header directly
                if not metadata:
                    logger.info(f"No metadata found via load_file for {f}, attempting header read")
                    metadata = read_safetensors_header(f)
                
                # Ensure metadata is a dictionary and serializable
                if not isinstance(metadata, dict):
                    logger.warning(f"Metadata in {f} is not a dictionary: {metadata}")
                    metadata = {"raw_metadata": str(metadata)}
                metadata = {k: str(v) for k, v in metadata.items()}
                
                logger.info(f"Metadata for {f}: {metadata}")
                lora_list.append([os.path.basename(f), f, metadata])
            except Exception as e:
                logger.warning(f"Failed to load metadata for {f}: {e}")
                lora_list.append([os.path.basename(f), f, {}])
        logger.info(f"Found LoRA files with metadata: {lora_list}")
        return lora_list
    except Exception as e:
        logger.error(f"Error listing LoRA files in {directory}: {e}")
        return []

def update_lora_fields(lora_list, last_valid_lora_state):
    """Update visibility and values of LoRA fields and their rows based on lora_list."""
    logger.debug(f"update_lora_fields called with lora_list: {lora_list}, last_valid_lora_state: {last_valid_lora_state}")
    updates = []
    valid_choices = ["None"] + list(AVAILABLE_LORAS.keys())
    # Ensure at least one LoRA field is visible
    if not lora_list:
        logger.warning("lora_list is empty, restoring last_valid_lora_state")
        lora_list = last_valid_lora_state
        if not lora_list:
            lora_list = [{"name": "None", "path": "", "weight": 1.5, "metadata": {}}]
    for i in range(MAX_LORA_FIELDS):
        if i < len(lora_list):
            lora = lora_list[i]
            display_name = lora["name"].split('_')[0] if lora["name"] != "None" and '_' in lora["name"] else lora["name"]
            if display_name not in valid_choices and not lora["path"]:
                logger.warning(f"Invalid LoRA name {display_name} in lora_list at index {i}, defaulting to 'None'")
                display_name = "None"
            path = lora["path"]
            try:
                weight = float(lora["weight"])
            except (ValueError, TypeError):
                logger.warning(f"Invalid weight {lora['weight']} for LoRA {lora['name']}, defaulting to 1.5")
                weight = 1.5
            metadata = lora.get("metadata", {})
            metadata_display = "\n".join([f"{k}: {v}" for k, v in metadata.items()]) if metadata else "No metadata available"
            visible = True
        else:
            display_name = "None"
            path = ""
            weight = 1.5
            metadata_display = "No metadata available"
            visible = False
        updates.extend([
            gr.update(value=display_name, visible=visible),  # lora_name (dropdown)
            gr.update(value=path, visible=visible and display_name not in valid_choices),  # lora_path (textbox)
            gr.update(value=weight, visible=visible),  # lora_weight
            gr.update(value=metadata_display, visible=visible),  # lora_metadata
            gr.update(visible=visible),  # remove_btn
            gr.update(visible=visible),  # row
        ])
    logger.debug(f"Returning updates for lora_components: {updates}")
    return updates

def add_lora(lora_list, last_valid_lora_state):
    if len(lora_list) < MAX_LORA_FIELDS:
        new_list = lora_list + [{"name": "None", "path": "", "weight": 1.5, "metadata": {}}]
        logger.info(f"Added new LoRA. New lora_list: {new_list}")
        return new_list, len(new_list) - 1, new_list  # Update last_valid_lora_state
    logger.info("Maximum LoRA fields reached.")
    return lora_list, None, last_valid_lora_state

def remove_lora(index, lora_list, last_valid_lora_state):
    if index < len(lora_list) and len(lora_list) > 1:
        new_list = lora_list.copy()
        new_list.pop(index)
        logger.info(f"Removed LoRA at index {index}. New lora_list: {new_list}")
        return new_list, min(index, len(new_list) - 1), new_list  # Update last_valid_lora_state
    logger.warning(f"Cannot remove LoRA at index {index}. lora_list length: {len(lora_list)}")
    return lora_list, index, last_valid_lora_state

def sanitize_lora_name(name):
    """Remove invalid characters from LoRA name to make it a valid PyTorch module name."""
    if not name or name == "None":
        return name
    clean_name = re.sub(r'\.safetensors$', '', name)
    clean_name = re.sub(r'[^a-zA-Z0-9_]', '_', clean_name)
    if clean_name and clean_name[0].isdigit():
        clean_name = f"lora_{clean_name}"
    logger.debug(f"Sanitized LoRA name: {name} -> {clean_name}")
    return clean_name

def update_lora(index, name, path, weight, lora_list, metadata=None):
    """Update a single LoRA entry in lora_list."""
    logger.debug(f"update_lora called with index={index}, name={name}, path={path}, weight={weight}, lora_list={lora_list}, metadata={metadata}")
    valid_choices = ["None"] + list(AVAILABLE_LORAS.keys())
    if name not in valid_choices and not path:
        logger.warning(f"Invalid LoRA name '{name}' at index {index}, defaulting to 'None'")
        name = "None"
        path = ""
        metadata = {}
    try:
        weight = float(weight)
    except (ValueError, TypeError):
        logger.warning(f"Invalid weight '{weight}' at index {index}, defaulting to 1.5")
        weight = 1.5
    if path and not path.endswith('.safetensors'):
        logger.warning(f"Invalid LoRA path '{path}' at index {index}, defaulting to 'None'")
        name = "None"
        path = ""
        metadata = {}
    elif name in AVAILABLE_LORAS:
        path = AVAILABLE_LORAS[name]
        try:
            safetensors_data = load_file(path, device="cpu")
            file_keys = list(safetensors_data.keys())
            logger.debug(f"Keys in built-in LoRA {path}: {file_keys}")
            metadata = safetensors_data.get("__metadata__", {})
            if not metadata:
                for key in file_keys:
                    if "metadata" in key.lower() or "info" in key.lower():
                        try:
                            metadata = safetensors_data[key]
                            if isinstance(metadata, str):
                                metadata = json.loads(metadata)
                            logger.debug(f"Found metadata in key {key} for {path}: {metadata}")
                            break
                        except Exception as e:
                            logger.warning(f"Failed to parse metadata from key {key} in {path}: {e}")
                if not metadata:
                    logger.info(f"No metadata found via load_file for {path}, attempting header read")
                    metadata = read_safetensors_header(path)
            metadata = {k: str(v) for k, v in metadata.items()}
            logger.info(f"Metadata for built-in LoRA {path}: {metadata}")
        except Exception as e:
            logger.warning(f"Failed to load metadata for built-in LoRA {path}: {e}")
            metadata = {}

    new_list = lora_list.copy()
    if index >= len(new_list):
        new_list.append({"name": "None", "path": "", "weight": 1.5, "metadata": {}})

    sanitized_name = sanitize_lora_name(name)
    unique_name = f"{sanitized_name}_{index}" if sanitized_name != "None" else "None"
    new_list[index] = {"name": unique_name, "path": path, "weight": weight, "metadata": metadata or {}}
    
    logger.info(f"Updated lora_list: {new_list}")
    return new_list

def update_lora_name(index, name, lora_list, last_valid_lora_state):
    """Update LoRA name and path based on selection."""
    logger.debug(f"update_lora_name called with index={index}, name={name}, lora_list={lora_list}")
    if index >= len(lora_list):
        return lora_list, last_valid_lora_state
    current_lora = lora_list[index]
    if current_lora["name"].split('_')[0] == name:
        return lora_list, last_valid_lora_state
    weight = float(current_lora["weight"]) if current_lora["weight"] else 1.5
    path = AVAILABLE_LORAS.get(name, "")
    metadata = {}
    if path:
        try:
            safetensors_data = load_file(path, device="cpu")
            file_keys = list(safetensors_data.keys())
            logger.debug(f"Keys in {path}: {file_keys}")
            metadata = safetensors_data.get("__metadata__", {})
            if not metadata:
                for key in file_keys:
                    if "metadata" in key.lower() or "info" in key.lower():
                        try:
                            metadata = safetensors_data[key]
                            if isinstance(metadata, str):
                                metadata = json.loads(metadata)
                            logger.debug(f"Found metadata in key {key}: {metadata}")
                            break
                        except Exception as e:
                            logger.warning(f"Failed to parse metadata from key {key} in {path}: {e}")
                if not metadata:
                    logger.info(f"No metadata found via load_file for {path}, attempting header read")
                    metadata = read_safetensors_header(path)
            metadata = {k: str(v) for k, v in metadata.items()}
            logger.info(f"Metadata for {path}: {metadata}")
        except Exception as e:
            logger.warning(f"Failed to load metadata for {path}: {e}")
            metadata = {}
    new_list = update_lora(index, name, path, weight, lora_list, metadata)
    return new_list, new_list  # Update last_valid_lora_state

def select_custom_lora(active_lora_index, lora_name, lora_list, custom_loras, last_valid_lora_state):
    """Update lora_state with the selected custom LoRA for the specified index and reset dropdown."""
    logger.debug(f"select_custom_lora called with active_lora_index={active_lora_index}, lora_name={lora_name}, lora_list={lora_list}, custom_loras={custom_loras}, last_valid_lora_state={last_valid_lora_state}")
    if active_lora_index is None or active_lora_index >= len(lora_list):
        logger.warning(f"Invalid active_lora_index {active_lora_index}, no update performed")
        return lora_list, None, last_valid_lora_state
    if lora_name is None:
        logger.debug("lora_name is None, skipping update to avoid resetting LoRA slot")
        return lora_list, None, last_valid_lora_state
    current_lora = lora_list[active_lora_index]
    weight = float(current_lora["weight"]) if current_lora["weight"] else 1.5
    path = ""
    metadata = {}
    if lora_name:
        for name, full_path, meta in custom_loras:
            if name == lora_name:
                path = full_path
                metadata = meta
                break
    name = lora_name if path else "None"
    updated_list = update_lora(active_lora_index, name, path, weight, lora_list, metadata)
    logger.debug(f"select_custom_lora returning updated_list: {updated_list}, custom_lora_select: None, last_valid_lora_state: {updated_list}")
    return updated_list, None, updated_list  # Update last_valid_lora_state

def update_lora_path(index, path, lora_list, last_valid_lora_state):
    """Update LoRA path manually entered by user."""
    logger.debug(f"update_lora_path called with index={index}, path={path}, lora_list={lora_list}")
    if index >= len(lora_list):
        return lora_list, last_valid_lora_state
    current_lora = lora_list[index]
    name = os.path.basename(path) if path and isinstance(path, str) and path.endswith('.safetensors') else "None"
    weight = float(current_lora["weight"]) if current_lora["weight"] else 1.5
    metadata = {}
    if path and isinstance(path, str) and path.endswith('.safetensors'):
        try:
            safetensors_data = load_file(path, device="cpu")
            file_keys = list(safetensors_data.keys())
            logger.debug(f"Keys in {path}: {file_keys}")
            metadata = safetensors_data.get("__metadata__", {})
            if not metadata:
                for key in file_keys:
                    if "metadata" in key.lower() or "info" in key.lower():
                        try:
                            metadata = safetensors_data[key]
                            if isinstance(metadata, str):
                                metadata = json.loads(metadata)
                            logger.debug(f"Found metadata in key {key}: {metadata}")
                            break
                        except Exception as e:
                            logger.warning(f"Failed to parse metadata from key {key} in {path}: {e}")
                if not metadata:
                    logger.info(f"No metadata found via load_file for {path}, attempting header read")
                    metadata = read_safetensors_header(path)
            metadata = {k: str(v) for k, v in metadata.items()}
            logger.info(f"Metadata for {path}: {metadata}")
        except Exception as e:
            logger.warning(f"Failed to load metadata for {path}: {e}")
            metadata = {}
    else:
        path = ""
    if current_lora["path"] == path:
        return lora_list, last_valid_lora_state
    new_list = update_lora(index, name, path, weight, lora_list, metadata)
    return new_list, new_list  # Update last_valid_lora_state

def update_lora_weight(index, weight, lora_list, last_valid_lora_state):
    """Update LoRA weight, preserving name, path, and metadata."""
    logger.debug(f"update_lora_weight called with index={index}, weight={weight}, lora_list={lora_list}")
    if index >= len(lora_list):
        return lora_list, last_valid_lora_state
    current_lora = lora_list[index]
    name = current_lora["name"].split('_')[0] if '_' in current_lora["name"] else current_lora["name"]
    path = current_lora["path"]
    metadata = current_lora.get("metadata", {})
    try:
        weight = float(weight)
    except (ValueError, TypeError):
        weight = 1.5
    if current_lora["weight"] == weight:
        return lora_list, last_valid_lora_state
    new_list = update_lora(index, name, path, weight, lora_list, metadata)
    return new_list, new_list  # Update last_valid_lora_state

sample_list = [
    ['./assets/examples/man.jpg', None, 'A sophisticated gentleman exuding confidence. He is dressed in a 1990s brown plaid jacket with a high collar, paired with a dark grey turtleneck. His trousers are tailored and charcoal in color, complemented by a sleek leather belt. The background showcases an elegant library with bookshelves, a marble fireplace, and warm lighting, creating a refined and cozy atmosphere. His relaxed posture and casual hand-in-pocket stance add to his composed and stylish demeanor', 666, [], 'aes_stage2'],
    ['./assets/examples/man.jpg', './assets/examples/man_pose.jpg', 'A man, portrait, cinematic', 42, [{"name": "realism_1", "path": AVAILABLE_LORAS["realism"], "weight": 1.5, "metadata": {}}], 'aes_stage2'],
    ['./assets/examples/man.jpg', None, 'A man, portrait, cinematic', 12345, [], 'sim_stage1'],
    ['./assets/examples/woman.jpg', './assets/examples/woman.jpg', 'A woman, portrait, cinematic', 1621695706, [], 'sim_stage1'],
    ['./assets/examples/woman.jpg', None, 'A young woman holding a sign with the text "InfiniteYou", "Infinite" in black and "You" in red, pure background', 3724009365, [], 'aes_stage2'],
    ['./assets/examples/woman.jpg', None, 'A photo of an elegant Javanese bride in traditional attire, with long hair styled into intricate a braid made of many fresh flowers, wearing a delicate headdress made from sequins and beads. She\'s holding flowers, light smiling at the camera, against a backdrop adorned with orchid blooms. The scene captures her grace as she stands amidst soft pastel colors, adding to its dreamy atmosphere', 42, [{"name": "realism_1", "path": AVAILABLE_LORAS["realism"], "weight": 1.5, "metadata": {}}], 'aes_stage2'],
    ['./assets/examples/woman.jpg', None, 'A photo of an elegant Javanese bride in traditional attire, with long hair styled into intricate a braid made of many fresh flowers, wearing a delicate headdress made from sequins and beads. She\'s holding flowers, light smiling at the camera, against a backdrop adorned with orchid blooms. The scene captures her grace as she stands amidst soft pastel colors, adding to its dreamy atmosphere', 42, [], 'sim_stage1'],
]

with gr.Blocks() as demo:
    session_state = gr.State({})
    default_model_version = "v1.0"
    lora_state = gr.State([{"name": "None", "path": "", "weight": 1.5, "metadata": {}}])
    custom_loras = gr.State([])  # Store [filename, full_path, metadata] triples
    active_lora_index = gr.State(0)  # Track which LoRA slot the custom_lora_select targets
    last_valid_lora_state = gr.State([{"name": "None", "path": "", "weight": 1.5, "metadata": {}}])  # Cache last valid lora_state

    gr.HTML("""
    <div style="text-align: center; max-width: 900px; margin: 0 auto;">
        <h1 style="font-size: 1.5rem; font-weight: 700; display: block;">InfiniteYou-FLUX</h1>
        <h2 style="font-size: 1.2rem; font-weight: 300; margin-bottom: 1rem; display: block;">Official Gradio Demo for <a href="https://arxiv.org/abs/2503.16418">InfiniteYou: Flexible Photo Recrafting While Preserving Your Identity</a></h2>
        <a href="https://bytedance.github.io/InfiniteYou">[Project Page]</a> 
        <a href="https://arxiv.org/abs/2503.16418">[Paper]</a> 
        <a href="https://github.com/bytedance/InfiniteYou">[Code]</a> 
        <a href="https://huggingface.co/ByteDance/InfiniteYou">[Model]</a>
        <style>
            .full-width input[type="text"] {
                width: 100% !important;
                box-sizing: border-box;
            }
        </style>
    </div>
    """)

    gr.Markdown("""
    ### 💡 How to Use This Demo:
    1. **Upload an identity (ID) image containing a human face.** For multiple faces, only the largest face will be detected. The face should ideally be clear and large enough, without significant occlusions or blur.
    2. **Enter the text prompt to describe the generated image and select the model version.** Please refer to **important usage tips** under the Generated Image field.
    3. *[Optional] Upload a control image containing a human face.* Only five facial keypoints will be extracted to control the generation. If not provided, we use a black control image, indicating no control.
    4. *[Optional] Adjust advanced hyperparameters or apply optional LoRAs to meet personal needs.* Please refer to **important usage tips** under the Generated Image field.
    5. **Click the "Generate" button to generate an image.** Enjoy!
    """)
    
    with gr.Row():
        with gr.Column():
            with gr.Row():
                ui_id_image = gr.Image(label="Identity Image", type="pil", height=370)

                with gr.Column():
                    ui_control_image = gr.Image(label="Control Image [Optional]", type="pil", height=370)
            
            ui_prompt_text = gr.Textbox(label="Prompt", value="Portrait, 4K, high quality, cinematic")
            ui_model_version = gr.Dropdown(
                label="Model Version",
                choices=[ModelVersion.STAGE_1, ModelVersion.STAGE_2],
                value=ModelVersion.DEFAULT_VERSION,
            )

            ui_btn_generate = gr.Button("Generate")
            with gr.Accordion("Advanced", open=False):
                with gr.Row():
                    ui_num_steps = gr.Number(label="num steps", value=30)
                    ui_seed = gr.Number(label="seed (0 for random)", value=0)
                with gr.Row():
                    ui_last_seed = gr.Textbox(label="Last Seed Used", value="", interactive=False)
                with gr.Row():
                    ui_width = gr.Number(label="width", value=864)
                    ui_height = gr.Number(label="height", value=1152)
                ui_guidance_scale = gr.Number(label="guidance scale", value=3.5)
                ui_infusenet_conditioning_scale = gr.Slider(minimum=0.0, maximum=1.0, value=1.0, step=0.05, label="infusenet conditioning scale")
                with gr.Row():
                    ui_infusenet_guidance_start = gr.Slider(minimum=0.0, maximum=1.0, value=0.0, step=0.05, label="infusenet guidance start")
                    ui_infusenet_guidance_end = gr.Slider(minimum=0.0, maximum=1.0, value=1.0, step=0.05, label="infusenet guidance end")
                with gr.Row():
                    ui_quantize_8bit = gr.Checkbox(label="Enable 8-bit quantization", value=True)
                    ui_cpu_offload = gr.Checkbox(label="Enable CPU offloading", value=True)

            with gr.Accordion("LoRAs [Optional]", open=True) as lora_accordion:
                lora_components = []
                metadata_components = []
                with gr.Column():
                    lora_dir = gr.Textbox(label="LoRA Directory", value="./loras")
                    list_lora_btn = gr.Button("Refresh LoRAs")
                    custom_lora_select = gr.Dropdown(
                        label="Select Custom LoRA",
                        choices=[],
                        value=None,
                        allow_custom_value=True,
                        interactive=True
                    )
                    for i in range(MAX_LORA_FIELDS):
                        with gr.Row(visible=i == 0) as row:
                            lora_name = gr.Dropdown(
                                label=f"LoRA {i+1}",
                                choices=["None"] + list(AVAILABLE_LORAS.keys()),
                                value="None",
                                allow_custom_value=True
                            )
                            lora_path = gr.Textbox(
                                label=f"LoRA Path {i+1} (Selected LoRA)",
                                value="",
                                visible=False,
                                elem_id=f"lora-path-{i}"
                            )
                            lora_weight = gr.Slider(
                                label=f"LoRA Weight {i+1}",
                                minimum=0.0,
                                maximum=2.0,
                                value=1.5,
                                step=0.1
                            )
                            remove_btn = gr.Button(f"Remove LoRA {i+1}")
                            lora_components.extend([lora_name, lora_path, lora_weight, None, remove_btn, row])
                        lora_metadata = gr.Textbox(
                            label=f"Metadata for LoRA {i+1}",
                            value="No metadata available",
                            interactive=False,
                            visible=i == 0,
                            lines=5,
                            elem_classes="full-width",
                            elem_id=f"lora-metadata-{i}"
                        )
                        metadata_components.append(lora_metadata)
                        lora_components[3 + i*6] = lora_metadata  # Replace None with lora_metadata in lora_components

                        lora_name.change(
                            fn=update_lora_name,
                            inputs=[gr.State(value=i), lora_name, lora_state, last_valid_lora_state],
                            outputs=[lora_state, last_valid_lora_state],
                            queue=False
                        ).then(
                            fn=update_lora_fields,
                            inputs=[lora_state, last_valid_lora_state],
                            outputs=lora_components,
                            queue=False
                        )
                        lora_path.change(
                            fn=update_lora_path,
                            inputs=[gr.State(value=i), lora_path, lora_state, last_valid_lora_state],
                            outputs=[lora_state, last_valid_lora_state],
                            queue=False
                        ).then(
                            fn=update_lora_fields,
                            inputs=[lora_state, last_valid_lora_state],
                            outputs=lora_components,
                            queue=False
                        )
                        lora_weight.change(
                            fn=update_lora_weight,
                            inputs=[gr.State(value=i), lora_weight, lora_state, last_valid_lora_state],
                            outputs=[lora_state, last_valid_lora_state],
                            queue=False
                        ).then(
                            fn=update_lora_fields,
                            inputs=[lora_state, last_valid_lora_state],
                            outputs=lora_components,
                            queue=False
                        )
                        remove_btn.click(
                            fn=remove_lora,
                            inputs=[gr.State(value=i), lora_state, last_valid_lora_state],
                            outputs=[lora_state, active_lora_index, last_valid_lora_state],
                            queue=False
                        ).then(
                            fn=update_lora_fields,
                            inputs=[lora_state, last_valid_lora_state],
                            outputs=lora_components,
                            queue=False
                        )
                        custom_lora_select.change(
                            fn=select_custom_lora,
                            inputs=[active_lora_index, custom_lora_select, lora_state, custom_loras, last_valid_lora_state],
                            outputs=[lora_state, custom_lora_select, last_valid_lora_state],
                            queue=False
                        ).then(
                            fn=update_lora_fields,
                            inputs=[lora_state, last_valid_lora_state],
                            outputs=lora_components,
                            queue=False
                        )

                    add_lora_btn = gr.Button("Add Another LoRA")
                    add_lora_btn.click(
                        fn=add_lora,
                        inputs=[lora_state, last_valid_lora_state],
                        outputs=[lora_state, active_lora_index, last_valid_lora_state],
                        queue=False
                    ).then(
                        fn=update_lora_fields,
                        inputs=[lora_state, last_valid_lora_state],
                        outputs=lora_components,
                        queue=False
                    )

                    list_lora_btn.click(
                        fn=list_lora_files,
                        inputs=[lora_dir],
                        outputs=[custom_loras],
                        queue=False
                    ).then(
                        fn=lambda cl: gr.update(choices=[name for name, _, _ in cl], value=None),
                        inputs=[custom_loras],
                        outputs=[custom_lora_select],
                        queue=False
                    )

        with gr.Column():
            image_output = gr.Image(label="Generated Image", interactive=False, height=550, format='png')
            gr.Markdown(
                """
                ### ❗️ Important Usage Tips:
                - **Model Version**: `aes_stage2` is used by default for better text-image alignment and aesthetics. For higher ID similarity, try `sim_stage1`.
                - **Useful Hyperparameters**: Usually, there is NO need to adjust too much. If necessary, try a slightly larger `--infusenet_guidance_start` (*e.g.*, `0.1`) only (especially helpful for `sim_stage1`). If still not satisfactory, then try a slightly smaller `--infusenet_conditioning_scale` (*e.g.*, `0.9`).
                - **Optional LoRAs**: Select built-in LoRAs (e.g., `realism`, `anti-blur`) from the "LoRA" dropdowns. For custom LoRAs, specify a directory containing .safetensors files (defaults to `./loras`) and click "Refresh LoRAs". LoRAs are automatically loaded from `./loras` on startup. Select a custom LoRA from the "Select Custom LoRA" dropdown to apply it to the active LoRA field (the most recently added or first available), and its full path will appear in "LoRA Path". Adjust weights (0.0 to 2.0) to control influence. Add multiple LoRAs with the "Add Another LoRA" button (up to 5), and remove unwanted ones with "Remove". LoRA metadata (e.g., trigger words, recommended weight) is displayed in the "Metadata for LoRA" field below each LoRA. LoRAs are optional and were NOT used in our paper unless specified.
                - **Gender Prompt**: If the generated gender is not preferred, add specific words in the prompt, such as 'a man', 'a woman', *etc*. We encourage using inclusive and respectful language.
                - **Performance Options**: Enable `8-bit quantization` to reduce memory usage and `CPU offloading` to use CPU memory for parts of the model, which can help on systems with limited GPU memory.
                - **Automatic Saving**: Generated images are automatically saved to the `./results` folder with filenames like `index_prompt_seed.png`.
                - **Reusing Seeds**: The "Last Seed Used" field shows the seed from the most recent generation. Copy it to the "seed" input to reuse it.
                """
            )

    gr.Examples(
        sample_list,
        inputs=[ui_id_image, ui_control_image, ui_prompt_text, ui_seed, lora_state, ui_model_version],
        outputs=[image_output, ui_last_seed],
        fn=generate_examples,
        cache_examples=False,
    )

    ui_btn_generate.click(
        fn=generate_image, 
        inputs=[
            ui_id_image, 
            ui_control_image, 
            ui_prompt_text, 
            ui_seed, 
            ui_width,
            ui_height,
            ui_guidance_scale, 
            ui_num_steps, 
            ui_infusenet_conditioning_scale, 
            ui_infusenet_guidance_start,
            ui_infusenet_guidance_end,
            lora_state,
            ui_quantize_8bit,
            ui_cpu_offload,
            ui_model_version
        ], 
        outputs=[image_output, ui_last_seed], 
        concurrency_id="gpu"
    )

    demo.load(
        fn=list_lora_files,
        inputs=[lora_dir],
        outputs=[custom_loras],
        queue=False
    ).then(
        fn=lambda cl: gr.update(choices=[name for name, _, _ in cl], value=None),
        inputs=[custom_loras],
        outputs=[custom_lora_select],
        queue=False
    )

    with gr.Accordion("Local Gradio Demo for Developers", open=False):
        gr.Markdown(
            'Please refer to our GitHub repository to [run the InfiniteYou-FLUX gradio demo locally](https://github.com/bytedance/InfiniteYou#local-gradio-demo).'
        )
    
    gr.Markdown(
        """
        --- 
        ### 📜 Disclaimer and Licenses 
        The images used in this demo are sourced from consented subjects or generated by the models. These pictures are intended solely to show the capabilities of our research. If you have any concerns, please contact us, and we will promptly remove any appropriate content.
        
        The use of the released code, model, and demo must strictly adhere to the respective licenses. 
        Our code is released under the [Apache 2.0 License](https://github.com/bytedance/InfiniteYou/blob/main/LICENSE), 
        and our model is released under the [Creative Commons Attribution-NonCommercial 4.0 International Public License](https://huggingface.co/ByteDance/InfiniteYou/blob/main/LICENSE) 
        for academic research purposes only. Any manual or automatic downloading of the face models from [InsightFace](https://github.com/deepinsight/insightface), 
        the [FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev) base model, LoRAs, *etc.*, must follow their original licenses and be used only for academic research purposes.

        This research aims to positively impact the field of Generative AI. Any usage of this method must be responsible and comply with local laws. The developers do not assume any responsibility for any potential misuse.
        """
    )    

    gr.Markdown(
        """
        ### 📖 Citation

        If you find InfiniteYou useful for your research or applications, please cite our paper:

        ```bibtex
        @article{jiang2025infiniteyou,
          title={{InfiniteYou}: Flexible Photo Recrafting While Preserving Your Identity},
          author={Jiang, Liming and Yan, Qing and Jia, Yumin and Liu, Zichuan and Kang, Hao and Lu, Xin},
          journal={arXiv preprint},
          volume={arXiv:2503.16418},
          year={2025}
        }
        ```

        We also appreciate it if you could give a star ⭐ to our [Github repository](https://github.com/bytedance/InfiniteYou). Thanks a lot!
        """
    )

demo.queue()
demo.launch(server_name='127.0.0.1',share=True)
