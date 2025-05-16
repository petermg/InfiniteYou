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
from datetime import datetime
import argparse
import gc
import os
import logging
import glob
import re
import json
import tempfile
import requests
from PIL import Image, PngImagePlugin
import numpy as np
import gradio as gr
import pillow_avif
from huggingface_hub import snapshot_download
from pillow_heif import register_heif_opener
from safetensors.torch import load_file
import insightface
from insightface.app import FaceAnalysis
import cv2
from scipy import ndimage
import torch
from pipelines.pipeline_infu_flux import InfUFluxPipeline

# Force reconfiguration of logging
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

def configure_logging(debug_to_log):
    handlers = [logging.StreamHandler()]
    if debug_to_log:
        handlers.append(logging.FileHandler('app.log', mode='a', encoding='utf-8'))
    
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers,
        force=True
    )

logger = logging.getLogger(__name__)
logger.info("Starting InfiniteYou-FLUX Gradio Demo")

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
INTERMEDIATE_DIR = "./results/intermediates"  # For debugging id_pil
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

time = (datetime.now().strftime("%Y%m%d%H%M%S"))

def download_models():
    global models_downloaded
    if not models_downloaded:
        logger.info("Downloading models...")
        try:
            snapshot_download(repo_id='ByteDance/InfiniteYou', local_dir='./models/InfiniteYou', local_dir_use_symlinks=False)
            logger.info("Downloaded InfiniteYou model")
        except Exception as e:
            logger.error(f"Failed to download InfiniteYou: {e}")
            raise
        try:
            snapshot_download(repo_id='ChuckMcSneed/FLUX.1-dev', local_dir='./models/FLUX.1-dev', local_dir_use_symlinks=False)
            logger.info("Downloaded FLUX.1-dev model")
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

def download_arcface_models(model_dir="C:\\Users\\pgomb\\.insightface\\models\\arcface"):
    """Download ArcFace models if not present."""
    os.makedirs(model_dir, exist_ok=True)
    models = [
        {"url": "https://huggingface.co/maze/faceX/resolve/main/w600k_r50.onnx", "name": "w600k_r50.onnx"},
        {"url": "https://huggingface.co/maze/faceX/resolve/main/det_10g.onnx", "name": "det_10g.onnx"}
    ]
    for model in models:
        model_path = os.path.join(model_dir, model["name"])
        if not os.path.exists(model_path):
            logger.info(f"Downloading {model['name']} to {model_path}")
            try:
                response = requests.get(model['url'], stream=True)
                response.raise_for_status()
                with open(model_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                logger.info(f"Successfully downloaded {model['name']}")
            except Exception as e:
                logger.error(f"Failed to download {model['name']}: {str(e)}")
                raise
        else:
            logger.debug(f"Model {model['name']} already exists at {model_path}")

def preprocess_image(img: Image.Image, target_size: tuple = (640, 640)) -> np.ndarray:
    """Preprocess image for ArcFace input with normalization."""
    try:
        # Convert to RGB and resize while maintaining aspect ratio
        img = img.convert('RGB')
        img_np = np.array(img)
        h, w = img_np.shape[:2]
        scale = min(target_size[0] / h, target_size[1] / w)
        new_h, new_w = int(h * scale), int(w * scale)
        img_resized = cv2.resize(img_np, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Pad to target size
        padded_img = np.zeros((target_size[0], target_size[1], 3), dtype=np.uint8)
        pad_h = (target_size[0] - new_h) // 2
        pad_w = (target_size[1] - new_w) // 2
        padded_img[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = img_resized
        
        # Normalize pixel values to [0, 1] and standardize
        padded_img = padded_img.astype(np.float32) / 255.0
        padded_img = (padded_img - 0.5) / 0.5  # Standardize to [-1, 1]
        
        # Convert to BGR for insightface
        img_bgr = cv2.cvtColor(padded_img, cv2.COLOR_RGB2BGR)
        logger.debug(f"Preprocessed image shape: {img_bgr.shape}, mean: {np.mean(img_bgr):.3f}, std: {np.std(img_bgr):.3f}")
        return img_bgr
    except Exception as e:
        logger.error(f"Failed to preprocess image: {str(e)}")
        return None

def align_face(img_array: np.ndarray, face) -> np.ndarray:
    """Align face based on landmarks."""
    try:
        landmarks = face.landmark_2d_106
        # Use eye landmarks (e.g., indices 35 and 104 for left and right eyes)
        left_eye = landmarks[35]
        right_eye = landmarks[104]
        dy = right_eye[1] - left_eye[1]
        dx = right_eye[0] - left_eye[0]
        angle = np.degrees(np.arctan2(dy, dx))
        
        # Compute center of eyes
        center = ((left_eye[0] + right_eye[0]) / 2, (left_eye[1] + right_eye[1]) / 2)
        
        # Rotate image
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        aligned_img = cv2.warpAffine(img_array, M, (img_array.shape[1], img_array.shape[0]))
        
        logger.debug(f"Aligned face with angle: {angle:.2f} degrees")
        return aligned_img
    except Exception as e:
        logger.error(f"Failed to align face: {str(e)}")
        return img_array

def extract_arcface_embedding(image_path, det_size=(640, 640)):
    """Extract ArcFace embedding for a single image using insightface."""
    logger.debug(f"Extracting ArcFace embedding for {image_path}")
    try:
        # Verify model files
        download_arcface_models()
        
        # Initialize FaceAnalysis
        app = FaceAnalysis(name='arcface', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        app.prepare(ctx_id=0, det_size=det_size, det_thresh=0.03)
        
        # Load and preprocess image
        img = Image.open(image_path)
        img_array = preprocess_image(img, det_size)
        if img_array is None:
            logger.warning(f"Failed to preprocess image {image_path}")
            return None
        
        # Detect faces
        faces = app.get(img_array)
        if not faces:
            logger.warning(f"No face detected for embedding in {image_path}")
            return None
        
        # Align face
        img_array = align_face(img_array, faces[0])
        
        # Re-detect face on aligned image
        faces = app.get(img_array)
        if not faces:
            logger.warning(f"No face detected after alignment in {image_path}")
            return None
        
        # Extract embedding
        embedding = faces[0].embedding
        if embedding is None:
            logger.error(f"Embedding is None for {image_path}, det_score: {faces[0].det_score}")
            return None
        
        # Verify embedding
        if not isinstance(embedding, np.ndarray) or embedding.size == 0:
            logger.error(f"Invalid embedding for {image_path}: {embedding}")
            return None
        
        logger.debug(f"Embedding shape: {embedding.shape}, norm: {np.linalg.norm(embedding)}")
        logger.info(f"Extracted embedding for {image_path}")
        return embedding
    except Exception as e:
        logger.error(f"Failed to extract embedding for {image_path}: {str(e)}")
        return None

def align_and_average_faces(id_images, det_size=(640, 640)):
    """Align faces based on landmarks and average them pixel-wise."""
    logger.info("Starting face alignment and averaging for provided images")
    app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=det_size)
    
    aligned_images = []
    valid_image_paths = []
    
    # Detect landmarks for all images
    landmarks_list = []
    for id_image in id_images:
        try:
            if isinstance(id_image, dict) and 'path' in id_image:
                image_path = id_image['path']
            else:
                image_path = id_image
            logger.debug(f"Processing image for landmarks: {image_path}")
            img = Image.open(image_path).convert('RGB')
            img_array = np.array(img)
            faces = app.get(img_array)
            
            if not faces:
                logger.warning(f"No face detected in {image_path}")
                continue
                
            landmarks = faces[0].landmark_2d_106
            landmarks_list.append(landmarks)
            aligned_images.append(img_array)
            valid_image_paths.append(image_path)
            logger.info(f"Landmarks detected in {image_path}")
        except Exception as e:
            logger.error(f"Failed to process image {image_path}: {e}")
            continue
    
    if not landmarks_list:
        logger.error("No valid faces detected for alignment")
        raise ValueError("No valid faces detected for alignment")
    
    # Choose reference landmarks (first image)
    ref_landmarks = landmarks_list[0]
    ref_image_shape = aligned_images[0].shape[:2]
    
    # Align images to reference
    aligned_arrays = []
    for i, (landmarks, img_array) in enumerate(zip(landmarks_list, aligned_images)):
        try:
            src_points = np.array([
                landmarks[33],  # Left eye
                landmarks[88],  # Right eye
                landmarks[55],  # Nose tip
            ], dtype=np.float32)
            dst_points = np.array([
                ref_landmarks[33],
                ref_landmarks[88],
                ref_landmarks[55],
            ], dtype=np.float32)
            
            M, _ = cv2.estimateAffinePartial2D(src_points, dst_points)
            if M is None:
                logger.warning(f"Failed to compute affine transform for {valid_image_paths[i]}")
                continue
                
            aligned_array = cv2.warpAffine(
                img_array,
                M,
                (ref_image_shape[1], ref_image_shape[0]),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REPLICATE
            )
            aligned_arrays.append(aligned_array)
            logger.info(f"Aligned image {valid_image_paths[i]}")
        except Exception as e:
            logger.error(f"Failed to align image {valid_image_paths[i]}: {e}")
            continue
    
    if not aligned_arrays:
        logger.error("No images aligned successfully")
        raise ValueError("No images aligned successfully")
    
    # Average aligned images
    try:
        aligned_arrays = [arr.astype(np.float32) for arr in aligned_arrays]
        averaged_array = np.mean(aligned_arrays, axis=0)
        averaged_array = np.clip(averaged_array, 0, 255).astype(np.uint8)
        averaged_image = Image.fromarray(averaged_array)
        logger.info(f"Averaged {len(aligned_arrays)} images into composite")
        return averaged_image
    except Exception as e:
        logger.error(f"Failed to average images: {e}")
        raise ValueError(f"Failed to average images: {str(e)}")

def select_best_face_image(id_images, mode="best_face", det_size=(640, 640)):
    """Select the best face, average aligned faces, or average embeddings based on mode."""
    logger.info(f"Selecting face image with raw mode: {mode}")
    
    # Validate mode
    valid_modes = ["best_face", "averaged_face", "averaged_embedding"]
    if mode not in valid_modes:
        logger.warning(f"Invalid mode '{mode}', defaulting to 'best_face'")
        mode = "best_face"
    logger.debug(f"Normalized mode: {mode}")
    
    # Normalize id_images to list of paths
    image_paths = []
    for id_image in id_images:
        path = id_image['path'] if isinstance(id_image, dict) and 'path' in id_image else id_image
        image_paths.append(path)
    logger.debug(f"Input image paths: {image_paths}, count: {len(image_paths)}")
    
    if mode == "averaged_face":
        try:
            result = align_and_average_faces(id_images, det_size)
            # Save intermediate for debugging
            os.makedirs(INTERMEDIATE_DIR, exist_ok=True)
            out_path = os.path.join(INTERMEDIATE_DIR, f"averaged_face_{time}.png")
            result.save(out_path)
            logger.info(f"Saved averaged face to {out_path}")
            return result
        except Exception as e:
            logger.error(f"Averaged face processing failed: {str(e)}, falling back to best_face")
            mode = "best_face"
    
    if mode == "averaged_embedding":
        logger.info("Starting embedding averaging for face selection")
        embeddings = []
        valid_images = []
        valid_image_paths = []
        
        try:
            for id_image in id_images:
                if isinstance(id_image, dict) and 'path' in id_image:
                    image_path = id_image['path']
                else:
                    image_path = id_image
                logger.debug(f"Processing embedding for {image_path}")
                embedding = extract_arcface_embedding(image_path, det_size)
                if embedding is not None:
                    embeddings.append(embedding)
                    valid_images.append(Image.open(image_path).convert('RGB'))
                    valid_image_paths.append(image_path)
                    logger.info(f"Embedding extracted for {image_path}, shape: {embedding.shape}")
                else:
                    logger.warning(f"No embedding extracted for {image_path}")
            
            logger.debug(f"Extracted {len(embeddings)} valid embeddings")
            if not embeddings:
                logger.error("No valid embeddings extracted")
                raise ValueError("No valid embeddings extracted")
            
            # Average embeddings
            avg_embedding = np.mean(embeddings, axis=0)
            logger.debug(f"Average embedding shape: {avg_embedding.shape}, norm: {np.linalg.norm(avg_embedding)}")
            avg_embedding = avg_embedding / np.linalg.norm(avg_embedding)  # Normalize
            logger.debug(f"Normalized average embedding norm: {np.linalg.norm(avg_embedding)}")
            
            # Find closest image to averaged embedding
            best_image = None
            best_score = -1
            best_image_path = None
            
            for img, emb, path in zip(valid_images, embeddings, valid_image_paths):
                score = np.dot(emb, avg_embedding)
                logger.debug(f"Similarity score for {path}: {score}")
                if score > best_score:
                    best_score = score
                    best_image = img
                    best_image_path = path
            
            if best_image is None:
                logger.error("No image selected after embedding comparison")
                raise ValueError("No image selected after embedding comparison")
                
            logger.info(f"Selected image with closest embedding: {best_image_path} (score: {best_score})")
            # Save intermediate for debugging
            os.makedirs(INTERMEDIATE_DIR, exist_ok=True)
            out_path = os.path.join(INTERMEDIATE_DIR, f"averaged_embedding_{time}.png")
            best_image.save(out_path)
            logger.info(f"Saved averaged embedding image to {out_path}")
            return best_image
        except Exception as e:
            logger.error(f"Averaged embedding Foundation Seriesprocessing failed: {str(e)}, falling back to best_face")
            mode = "best_face"
    
    # Best face selection
    logger.info("Starting face detection for best face selection")
    app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=det_size)
    
    best_image = None
    best_confidence = -1
    best_image_path = None
    
    for id_image in id_images:
        try:
            if isinstance(id_image, dict) and 'path' in id_image:
                image_path = id_image['path']
            else:
                image_path = id_image
            logger.debug(f"Processing image: {image_path}")
            img = Image.open(image_path).convert('RGB')
            img_array = np.array(img)
            faces = app.get(img_array)
            
            if faces:
                confidence = faces[0].det_score
                logger.info(f"Image {image_path}: Face detected with confidence {confidence}")
                if confidence > best_confidence:
                    best_confidence = confidence
                    best_image = img
                    best_image_path = image_path
            else:
                logger.warning(f"No face detected in {image_path}")
        except Exception as e:
            logger.error(f"Failed to process image {image_path}: {str(e)}")
            continue
    
    if best_image is None:
        logger.error("No valid face detected in any provided images")
        raise ValueError("No valid face detected in any provided images")
    
    logger.info(f"Selected best image: {best_image_path} with confidence {best_confidence}")
    # Save intermediate for debugging
    os.makedirs(INTERMEDIATE_DIR, exist_ok=True)
    out_path = os.path.join(INTERMEDIATE_DIR, f"best_face_{time}.png")
    best_image.save(out_path)
    logger.info(f"Saved best face image to {out_path}")
    return best_image

def prepare_pipeline(model_version, loras, quantize_8bit, cpu_offload, debug_to_log):
    logger.info(f"Preparing pipeline with model_version={model_version}, loras={loras}, debug_to_log={debug_to_log}")
    
    if (
        loaded_pipeline_config['pipeline'] is not None
        and loaded_pipeline_config["loras"] == loras
        and loaded_pipeline_config["quantize_8bit"] == quantize_8bit
        and loaded_pipeline_config["cpu_offload"] == cpu_offload
        and model_version == loaded_pipeline_config["model_version"]
    ):
        logger.info("Reusing existing pipeline")
        return loaded_pipeline_config['pipeline']
    
    loaded_pipeline_config["loras"] = loras
    loaded_pipeline_config["quantize_8bit"] = quantize_8bit
    loaded_pipeline_config["cpu_offload"] = cpu_offload
    loaded_pipeline_config["model_version"] = model_version

    pipeline = loaded_pipeline_config['pipeline']
    if pipeline is None or pipeline.model_version != model_version:
        logger.info(f"Switching to model: {model_version}")
        if pipeline is not None:
            logger.debug("Deleting existing pipeline")
            del pipeline
            del loaded_pipeline_config['pipeline']
            gc.collect()
            torch.cuda.empty_cache()

        model_path = f'./models/InfiniteYou/infu_flux_v1.0/{model_version}'
        logger.debug(f'Loading model from {model_path}')

        try:
            pipeline = InfUFluxPipeline(
                base_model_path='./models/FLUX.1-dev',
                infu_model_path=model_path,
                insightface_root_path='./models/InfiniteYou/supports/insightface',
                image_proj_num_tokens=8,
                infu_flux_version='v1.0',
                model_version=model_version,
                quantize_8bit=quantize_8bit,
                cpu_offload=cpu_offload,
                debug_to_log=debug_to_log
            )
            logger.info("Pipeline initialized")
        except Exception as e:
            logger.error(f"Failed to initialize pipeline: {e}")
            raise

        loaded_pipeline_config['pipeline'] = pipeline

    try:
        pipeline.pipe.unload_lora_weights()
        logger.debug("Unloaded previous LoRA weights")
    except Exception as e:
        logger.error(f"Failed to unload LoRA weights: {e}")

    if loras:
        logger.debug(f"Loading LoRAs: {loras}")
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
            logger.debug("LoRAs loaded")
        except Exception as e:
            logger.error(f"Failed to load LoRAs: {e}")
            raise RuntimeError(f"Failed to load LoRAs: {e}")

    logger.info("Pipeline preparation complete")
    return pipeline

def generate_image(
    id_images, 
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
    model_version,
    num_images,
    face_selection_mode,
    debug_to_log
):
    logger.info("Generate button clicked: Entering generate_image")
    logger.debug(f"Raw inputs: id_images={type(id_images)}, prompt={prompt}, seed={seed}, num_images={num_images}, face_selection_mode={face_selection_mode}, debug_to_log={debug_to_log}")
    
    # Reconfigure logging based on debug_to_log
    configure_logging(debug_to_log)
    
    # Log raw face_selection_mode
    logger.info(f"Raw face selection mode from UI: {face_selection_mode}")
    
    # Normalize id_images to a list
    logger.debug("Normalizing id_images")
    if id_images is None:
        id_images = []
    elif isinstance(id_images, dict):
        id_images = [id_images]
    elif not isinstance(id_images, list):
        id_images = [id_images]
    
    logger.debug(f"Normalized id_images: {[img['path'] if isinstance(img, dict) else img for img in id_images]}")

    # Validate inputs
    if not id_images:
        logger.warning("No identity images provided")
        gr.Error("Please upload at least one identity image")
        return gr.update(), ""
    
    loras = convert_lora_state_to_loras(lora_state)
    logger.debug(f"LoRAs: {loras}")

    logger.debug("Checking model download")
    try:
        download_models()
        logger.debug("Models downloaded")
    except Exception as e:
        logger.error(f"Model download failed: {e}")
        gr.Error(f"Model download failed: {str(e)}")
        return gr.update(), ""

    logger.debug("Preparing pipeline")
    try:
        pipeline = prepare_pipeline(
            model_version=model_version,
            loras=loras,
            quantize_8bit=quantize_8bit,
            cpu_offload=cpu_offload,
            debug_to_log=debug_to_log
        )
        logger.debug("Pipeline ready")
    except Exception as e:
        logger.error(f"Pipeline preparation failed: {e}")
        gr.Error(f"Pipeline preparation failed: {str(e)}")
        return gr.update(), ""

    logger.debug("Processing num_images")
    try:
        num_images = int(num_images)
        if num_images < 1:
            num_images = 1
    except (ValueError, TypeError):
        num_images = 1
    logger.debug(f"Number of images to generate: {num_images}")

    logger.debug("Selecting face image")
    try:
        id_pil = select_best_face_image(id_images, mode=face_selection_mode.lower())
        logger.debug(f"Selected face with mode: {face_selection_mode}")
    except ValueError as e:
        logger.error(f"Face selection error: {e}")
        gr.Error(str(e))
        return gr.update(), ""
    except Exception as e:
        logger.error(f"Unexpected face selection error: {e}")
        gr.Error(f"Unexpected face selection error: {str(e)}")
        return gr.update(), ""

    images = []
    seeds = []
    base_seed = seed

    logger.debug("Starting image generation loop")
    for i in range(num_images):
        current_seed = base_seed
        if current_seed == 0:
            current_seed = torch.seed() & 0xFFFFFFFF
            logger.debug(f"Random seed for image {i+1}: {current_seed}")
        else:
            current_seed = base_seed + i
            logger.debug(f"Seed for image {i+1}: {current_seed}")

        logger.debug(f"Generating image {i+1}")
        try:
            image = pipeline(
                id_image=id_pil,
                prompt=prompt,
                control_image=control_image,
                seed=current_seed,
                width=width,
                height=height,
                guidance_scale=guidance_scale,
                num_steps=num_steps,
                infusenet_conditioning_scale=infusenet_conditioning_scale,
                infusenet_guidance_start=infusenet_guidance_start,
                infusenet_guidance_end=infusenet_guidance_end,
                cpu_offload=cpu_offload,
            )
            logger.debug(f"Image {i+1} generated")
            if image is None:
                logger.error(f"Pipeline returned None for image {i+1}")
                gr.Error(f"Image generation failed: Pipeline returned None")
                continue

            os.makedirs(OUTPUT_DIR, exist_ok=True)
            index = len(os.listdir(OUTPUT_DIR))
            prompt_name = ''.join(c if c.isalnum() or c in '_-' else '_' for c in prompt[:50].replace(' ', '_')).strip('_')
            out_name = f"{index:05d}_{prompt_name}_seed{current_seed}_time{time}_img{i+1}.png"
            out_path = os.path.join(OUTPUT_DIR, out_name)

            metadata = {
                "prompt": prompt,
                "loras": [
                    {
                        "name": lora["name"],
                        "path": lora["path"],
                        "weight": lora["weight"]
                    } for lora in lora_state if lora["name"] != "None" and lora.get("path", "")
                ],
                "model_version": model_version,
                "guidance_scale": guidance_scale,
                "num_steps": num_steps,
                "seed": current_seed,
                "width": width,
                "height": height,
                "infusenet_conditioning_scale": infusenet_conditioning_scale,
                "infusenet_guidance_start": infusenet_guidance_start,
                "infusenet_guidance_end": infusenet_guidance_end,
                "quantize_8bit": quantize_8bit,
                "cpu_offload": cpu_offload,
                "num_input_images": len(id_images),
                "face_selection_mode": face_selection_mode,
                "debug_to_log": debug_to_log
            }

            metadata_json = json.dumps(metadata, indent=2)
            png_info = PngImagePlugin.PngInfo()
            png_info.add_text("parameters", metadata_json)

            image.save(out_path, pnginfo=png_info)
            logger.info(f"Image {i+1} saved to {out_path}")
            images.append(image)
            seeds.append(str(current_seed))
        except Exception as e:
            logger.error(f"Error generating image {i+1}: {e}")
            gr.Error(f"Error generating image {i+1}: {str(e)}")
            continue

    if not images:
        logger.error("No images generated")
        gr.Error("No images were generated")
        return gr.update(), ",".join(seeds)

    logger.info(f"Generated {len(images)} images with seeds: {seeds}")
    return gr.update(value=images, label=f"Generated Images, seeds={','.join(seeds)}, saved to {OUTPUT_DIR}"), ",".join(seeds)

def generate_examples(id_images, control_image, prompt_text, seed, lora_state, model_version, face_selection_mode="best_face", debug_to_log=False):
    loras = convert_lora_state_to_loras(lora_state)
    logger.debug(f"Generating example with loras={loras}, face_selection_mode={face_selection_mode}, debug_to_log={debug_to_log}")
    configure_logging(debug_to_log)
    result, last_seed = generate_image(
        id_images, control_image, prompt_text, seed, 864, 1152, 3.5, 30, 1.0, 0.0, 1.0,
        lora_state, QUANTIZE_8BIT_DEFAULT, CPU_OFFLOAD_DEFAULT, model_version, num_images=1,
        face_selection_mode=face_selection_mode,
        debug_to_log=debug_to_log
    )
    images = result.value if isinstance(result.value, list) else [result.value] if result.value else []
    return gr.update(value=images, label=f"Generated Example, seed={last_seed}"), last_seed

def convert_lora_state_to_loras(lora_state):
    loras = []
    for lora in lora_state:
        if lora["name"] != "None" and lora.get("path", ""):
            try:
                weight = float(lora["weight"])
            except (ValueError, TypeError):
                logger.warning(f"Invalid weight {lora['weight']} for LoRA {lora['name']}, defaulting to 1.5")
                weight = 1.5
            loras.append([lora["path"], lora["name"], weight])
    logger.debug(f"Converted lora_state to loras: {loras}")
    return loras

def read_safetensors_header(file_path):
    try:
        with open(file_path, 'rb') as f:
            header_len_bytes = f.read(8)
            if len(header_len_bytes) != 8:
                logger.warning(f"Failed to read header length from {file_path}")
                return {}
            header_len = int.from_bytes(header_len_bytes, byteorder='little')
            header_data = f.read(header_len)
            if len(header_data) != header_len:
                logger.warning(f"Incomplete header read from {file_path}")
                return {}
            header_str = header_data.decode('utf-8')
            header_json = json.loads(header_str)
            metadata = header_json.get("__metadata__", {})
            metadata = {k: str(v) for k, v in metadata.items()}
            logger.debug(f"Read metadata from {file_path}: {metadata}")
            return metadata
    except Exception as e:
        logger.error(f"Error reading safetensors header from {file_path}: {e}")
        return {}

def list_lora_files(directory):
    logger.debug(f"Listing LoRA files in {directory}")
    if not directory:
        return []
    safetensors_files = glob.glob(os.path.join(directory, "*.safetensors"))
    lora_list = []
    for f in safetensors_files:
        try:
            metadata = read_safetensors_header(f)
            lora_list.append([os.path.basename(f), f, metadata])
        except Exception as e:
            logger.warning(f"Failed to read metadata for {f}: {e}")
            lora_list.append([os.path.basename(f), f, {}])
    logger.debug(f"Found LoRA files: {lora_list}")
    return lora_list

def update_lora_fields(lora_list, last_valid_lora_state):
    logger.debug(f"Updating LoRA fields with lora_list={lora_list}")
    updates = []
    valid_choices = ["None"] + list(AVAILABLE_LORAS.keys())
    if not lora_list:
        lora_list = last_valid_lora_state or [{"name": "None", "path": "", "weight": 1.5, "metadata": {}}]
    for i in range(MAX_LORA_FIELDS):
        if i < len(lora_list):
            lora = lora_list[i]
            display_name = lora["name"].split('_')[0] if lora["name"] != "None" and '_' in lora["name"] else lora["name"]
            if display_name not in valid_choices and not lora.get("path", ""):
                display_name = "None"
            path = lora.get("path", "")
            weight = float(lora["weight"]) if lora.get("weight") else 1.5
            metadata = lora.get("metadata", {})
            metadata_display = "\n".join([f"{k}: {v}" for k, v in metadata.items()]) if metadata else "No metadata"
            visible = True
        else:
            display_name = "None"
            path = ""
            weight = 1.5
            metadata_display = "No metadata"
            visible = False
        updates.extend([
            gr.update(value=display_name, visible=visible),
            gr.update(value=path, visible=visible and display_name not in valid_choices),
            gr.update(value=weight, visible=visible),
            gr.update(value=metadata_display, visible=visible),
            gr.update(visible=visible),
            gr.update(visible=visible),
        ])
    return updates

def add_lora(lora_list, last_valid_lora_state):
    if len(lora_list) < MAX_LORA_FIELDS:
        new_list = lora_list + [{"name": "None", "path": "", "weight": 1.5, "metadata": {}}]
        logger.debug(f"Added LoRA: {new_list}")
        return new_list, len(new_list) - 1, new_list
    logger.info("Max LoRA fields reached")
    return lora_list, None, last_valid_lora_state

def remove_lora(index, lora_list, last_valid_lora_state):
    if index < len(lora_list) and len(lora_list) > 1:
        new_list = lora_list[:index] + lora_list[index+1:]
        logger.debug(f"Removed LoRA at index {index}: {new_list}")
        return new_list, min(index, len(new_list) - 1), new_list
    logger.warning("Cannot remove LoRA")
    return lora_list, index, last_valid_lora_state

def sanitize_lora_name(name):
    if not name or name == "None":
        return name
    clean_name = re.sub(r'\.safetensors$', '', name)
    clean_name = re.sub(r'[^a-zA-Z0-9_]', '_', clean_name)
    if clean_name[0].isdigit():
        clean_name = f"lora_{clean_name}"
    logger.debug(f"Sanitized LoRA name: {name} -> {clean_name}")
    return clean_name

def update_lora(index, name, path, weight, lora_list, metadata=None):
    logger.debug(f"Updating LoRA {index}: name={name}, path={path}, weight={weight}")
    valid_choices = ["None"] + list(AVAILABLE_LORAS.keys())
    if name not in valid_choices and not path:
        name = "None"
        path = ""
        metadata = {}
    weight = float(weight) if weight else 1.5
    if path and not path.endswith('.safetensors'):
        name = "None"
        path = ""
        metadata = {}
    elif name in AVAILABLE_LORAS:
        path = AVAILABLE_LORAS[name]
        metadata = read_safetensors_header(path)

    new_list = lora_list.copy()
    if index >= len(new_list):
        new_list.append({"name": "None", "path": "", "weight": 1.5, "metadata": {}})
    sanitized_name = sanitize_lora_name(name)
    unique_name = f"{sanitized_name}_{index}" if sanitized_name != "None" else "None"
    new_list[index] = {"name": unique_name, "path": path, "weight": weight, "metadata": metadata or {}}
    logger.debug(f"Updated lora_list: {new_list}")
    return new_list

def update_lora_name(index, name, lora_list, last_valid_lora_state):
    logger.debug(f"Updating LoRA name at index {index} to {name}")
    if index >= len(lora_list):
        return lora_list, last_valid_lora_state
    current_lora = lora_list[index]
    if current_lora["name"].split('_')[0] == name:
        return lora_list, last_valid_lora_state
    weight = float(current_lora["weight"]) if current_lora.get("weight") else 1.5
    path = AVAILABLE_LORAS.get(name, "")
    metadata = read_safetensors_header(path) if path else {}
    new_list = update_lora(index, name, path, weight, lora_list, metadata)
    return new_list, new_list

def select_custom_lora(active_lora_index, lora_name, lora_list, custom_loras, last_valid_lora_state):
    logger.debug(f"Selecting custom LoRA: index={active_lora_index}, name={lora_name}")
    if active_lora_index is None or active_lora_index >= len(lora_list):
        return lora_list, None, last_valid_lora_state
    if lora_name is None:
        return lora_list, None, last_valid_lora_state
    current_lora = lora_list[active_lora_index]
    weight = float(current_lora["weight"]) if current_lora.get("weight") else 1.5
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
    return updated_list, None, updated_list

def update_lora_path(index, path, lora_list, last_valid_lora_state):
    logger.debug(f"Updating LoRA path at index {index} to {path}")
    if index >= len(lora_list):
        return lora_list, last_valid_lora_state
    current_lora = lora_list[index]
    name = os.path.basename(path) if path and path.endswith('.safetensors') else "None"
    weight = float(current_lora["weight"]) if current_lora.get("weight") else 1.5
    metadata = read_safetensors_header(path) if path and path.endswith('.safetensors') else {}
    new_list = update_lora(index, name, path, weight, lora_list, metadata)
    return new_list, new_list

def update_lora_weight(index, weight, lora_list, last_valid_lora_state):
    logger.debug(f"Updating LoRA weight at index {index} to {weight}")
    if index >= len(lora_list):
        return lora_list, last_valid_lora_state
    current_lora = lora_list[index]
    name = current_lora["name"].split('_')[0] if '_' in current_lora["name"] else current_lora["name"]
    path = current_lora.get("path", "")
    metadata = current_lora.get("metadata", {})
    weight = float(weight) if weight else 1.5
    new_list = update_lora(index, name, path, weight, lora_list, metadata)
    return new_list, new_list

with gr.Blocks() as demo:
    session_state = gr.State({})
    default_model_version = "v1.0"
    lora_state = gr.State([{"name": "None", "path": "", "weight": 1.5, "metadata": {}}])
    custom_loras = gr.State([])
    active_lora_index = gr.State(0)
    last_valid_lora_state = gr.State([{"name": "None", "path": "", "weight": 1.5, "metadata": {}}])

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
    1. **Upload multiple identity (ID) images of the same person's face.** These images will be processed to select the best face, create an averaged composite, or average embeddings based on the "Face Selection Mode". Each image should contain a clear, large face without significant occlusions or blur.
    2. **Enter the text prompt to describe the generated image and select the model version.** Please refer to **important usage tips** under the Generated Image field.
    3. *[Optional] Upload a control image containing a human face.* Only five facial keypoints will be extracted to control the generation. If not provided, we use a black control image, indicating no control.
    4. *[Optional] Adjust advanced hyperparameters or apply optional LoRAs to meet personal needs.* Please refer to **important usage tips** under the Generated Image field.
    5. **Specify the number of images to generate for the identity (default is 1).**
    6. **Click the "Generate" button to generate images.** Enjoy!
    """)
    
    with gr.Row():
        with gr.Column():
            with gr.Row():
                ui_id_image = gr.File(label="Identity Images (Multiple of Same Face)", file_types=[".png", ".jpg", ".jpeg", ".heif"], file_count="multiple", height=370)

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
                    ui_face_selection_mode = gr.Dropdown(
                        label="Face Selection Mode",
                        choices=["best_face", "averaged_face", "averaged_embedding"],
                        value="best_face",
                    )
                with gr.Row():
                    ui_num_steps = gr.Number(label="num steps", value=30)
                    ui_seed = gr.Number(label="seed (0 for random)", value=0)
                with gr.Row():
                    ui_last_seed = gr.Textbox(label="Last Seeds Used", value="", interactive=False)
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
                    ui_debug_to_log = gr.Checkbox(label="Debug to Log File", value=False)
                ui_num_images = gr.Number(label="Number of Images to Generate", value=1, minimum=1, precision=0)

            with gr.Accordion("LoRAs [Optional]", open=True):
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
                                visible=False
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
                            lines=5
                        )
                        metadata_components.append(lora_metadata)
                        lora_components[3 + i*6] = lora_metadata

                        lora_name.change(
                            fn=update_lora_name,
                            inputs=[gr.State(value=i), lora_name, lora_state, last_valid_lora_state],
                            outputs=[lora_state, last_valid_lora_state]
                        ).then(
                            fn=update_lora_fields,
                            inputs=[lora_state, last_valid_lora_state],
                            outputs=lora_components
                        )
                        lora_path.change(
                            fn=update_lora_path,
                            inputs=[gr.State(value=i), lora_path, lora_state, last_valid_lora_state],
                            outputs=[lora_state, last_valid_lora_state]
                        ).then(
                            fn=update_lora_fields,
                            inputs=[lora_state, last_valid_lora_state],
                            outputs=lora_components
                        )
                        lora_weight.change(
                            fn=update_lora_weight,
                            inputs=[gr.State(value=i), lora_weight, lora_state, last_valid_lora_state],
                            outputs=[lora_state, last_valid_lora_state]
                        ).then(
                            fn=update_lora_fields,
                            inputs=[lora_state, last_valid_lora_state],
                            outputs=lora_components
                        )
                        remove_btn.click(
                            fn=remove_lora,
                            inputs=[gr.State(value=i), lora_state, last_valid_lora_state],
                            outputs=[lora_state, active_lora_index, last_valid_lora_state]
                        ).then(
                            fn=update_lora_fields,
                            inputs=[lora_state, last_valid_lora_state],
                            outputs=lora_components
                        )
                        custom_lora_select.change(
                            fn=select_custom_lora,
                            inputs=[active_lora_index, custom_lora_select, lora_state, custom_loras, last_valid_lora_state],
                            outputs=[lora_state, custom_lora_select, last_valid_lora_state]
                        ).then(
                            fn=update_lora_fields,
                            inputs=[lora_state, last_valid_lora_state],
                            outputs=lora_components
                        )

                    add_lora_btn = gr.Button("Add Another LoRA")
                    add_lora_btn.click(
                        fn=add_lora,
                        inputs=[lora_state, last_valid_lora_state],
                        outputs=[lora_state, active_lora_index, last_valid_lora_state]
                    ).then(
                        fn=update_lora_fields,
                        inputs=[lora_state, last_valid_lora_state],
                        outputs=lora_components
                    )

                    list_lora_btn.click(
                        fn=list_lora_files,
                        inputs=[lora_dir],
                        outputs=[custom_loras]
                    ).then(
                        fn=lambda cl: gr.update(choices=[name for name, _, _ in cl], value=None),
                        inputs=[custom_loras],
                        outputs=[custom_lora_select]
                    )

        with gr.Column():
            image_output = gr.Gallery(label="Generated Images", interactive=False, height=550)
            gr.Markdown(
                """
                ### ❗️ Important Usage Tips:
                - **Model Version**: `aes_stage2` is used by default for better text-image alignment and aesthetics. For higher ID similarity, try `sim_stage1`.
                - **Face Selection Mode**: Determines how the identity image is selected from multiple uploaded images:
                  - **`best_face`**: Analyzes each uploaded image using a face detection model and selects the image with the highest confidence score for a detected face. This mode is ideal when you want the clearest and most reliable single image to represent the identity, prioritizing quality and clarity.
                  - **`averaged_face`**: Aligns all uploaded images based on facial landmarks (e.g., eyes and nose) to a reference image, then averages the pixel values to create a composite image. This mode is useful for creating a smoothed, representative identity that blends features from multiple images, reducing noise or inconsistencies.
                  - **`averaged_embedding`**: Extracts ArcFace embeddings (numerical representations of facial features) from each image, averages these embeddings to create a composite embedding, and selects the original image whose embedding is closest to this average. This mode is effective for capturing a consistent identity across varied images by focusing on deep facial features.
                - **Useful Hyperparameters**: Usually, there is NO need to adjust too much. If necessary, try a slightly larger `--infusenet_guidance_start` (*e.g.*, `0.1`) only (especially helpful for `sim_stage1`). If still not satisfactory, then try a slightly smaller `--infusenet_conditioning_scale` (*e.g.*, `0.9`).
                - **Optional LoRAs**: Select built-in LoRAs (e.g., `realism`, `anti-blur`) from the "LoRA" dropdowns. For custom LoRAs, specify a directory containing .safetensors files (defaults to `./lora`) and click "Refresh LoRAs". LoRAs are automatically loaded from `./loras` on startup. Select a custom LoRA from the "Select Custom LoRA" dropdown to apply it to the active LoRA field (the most recently added or first available), and its full path will appear in "LoRA Path". Adjust weights (0.0 to 2.0) to control influence. Add multiple LoRAs with the "Add Another LoRA" button (up to 5), and remove unwanted ones with "Remove". LoRA metadata (e.g., trigger words, recommended weight) is displayed in the "Metadata for LoRA" field below each LoRA. LoRAs are optional and were NOT used in our paper unless specified.
                - **Gender Prompt**: If the generated gender is not preferred, add specific words in the prompt, such as 'a man', 'a woman', *etc*. We encourage using inclusive and respectful language.
                - **Performance Options**: Enable `8-bit quantization` to reduce memory usage and `CPU offloading` to use CPU memory for parts of the model, which can help on systems with limited GPU memory. Enable `Debug to Log File` to write logs to `app.log` and `pipeline.log` for debugging; disable when not needed to avoid unnecessary file writes.
                - **Automatic Saving**: Generated images are automatically saved to the `./results` folder with filenames like `index_prompt_seed_imgN.png`.
                - **Reusing Seeds**: The "Last Seeds Used" field shows the seeds from the most recent generation. Copy them to the "seed" input to reuse them.
                - **Multiple Images**: Upload multiple images of the same face to improve identity accuracy. The system selects the best image, averages them, or uses embeddings based on the "Face Selection Mode". Specify the number of output images to generate (default is 1). Each image uses a unique seed (incremented from the base seed or random if set to 0).
                - **Debugging**: Intermediate images for each face selection mode are saved to `./results/intermediates` (e.g., `best_face_<time>.png`, `averaged_face_<time>.png`, `averaged_embedding_<time>.png`) to verify differences.
                """
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
            ui_model_version,
            ui_num_images,
            ui_face_selection_mode,
            ui_debug_to_log
        ],
        outputs=[image_output, ui_last_seed]
    )

    demo.load(
        fn=list_lora_files,
        inputs=[lora_dir],
        outputs=[custom_loras]
    ).then(
        fn=lambda cl: gr.update(choices=[name for name, _, _ in cl], value=None),
        inputs=[custom_loras],
        outputs=[custom_lora_select]
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

if __name__ == "__main__":
    demo.queue()
    demo.launch(server_name='127.0.0.1')
