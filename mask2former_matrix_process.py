# generate_segmentation_masks.py

import os
import json
import argparse
import sys
from pathlib import Path
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation

# --- Configuration ---
# Mapping dataset identifiers to model names
MODEL_MAPPING = {
    "ade20k": "facebook/mask2former-swin-large-ade-semantic",
    "cityscapes": "facebook/mask2former-swin-large-cityscapes-semantic",
}

# Define default input processing sizes (can be overridden)
# These might differ from the final *output* mask resolution
DEFAULT_INPUT_RES = {
    "ade20k": (512, 512),     # ADE20k models often work well with square inputs
    "cityscapes": (512, 512) # Cityscapes aspect ratio is often 2:1
}

def derive_output_path(image_path_str: str, base_output_dir: str = None) -> Path:
    """
    Derives the output path for the mask based on the input image path.
    Example: /path/to/dataset_dir/img.png -> /path/to/dataset_dir_mask/img.png
    If base_output_dir is provided, it's used as the root.
    Example: /data/images/set1/001.png with base_output_dir /output/masks
             -> /output/masks/set1_mask/001.png
    """
    image_path = Path(image_path_str)
    original_parent_name = image_path.parent.name
    mask_parent_name = original_parent_name + "_mask"

    if base_output_dir:
        output_parent_dir = Path(base_output_dir) / mask_parent_name
    else:
        # Place the mask directory alongside the original image directory
        output_parent_dir = image_path.parent.parent / mask_parent_name

    output_path = output_parent_dir / image_path.name
    return output_path

def process_and_save_mask(
    image_path_str: str,
    model,
    image_processor,
    device,
    output_resolution: tuple[int, int] | None,
    input_processing_resolution: tuple[int, int],
    output_path: Path,
    overwrite: bool = False
):
    """
    Processes a single image, generates its segmentation mask, resizes it,
    and saves it as a grayscale PNG.
    """
    if not overwrite and output_path.exists():
        # print(f"Skipping existing mask: {output_path}")
        return True # Indicate success (or skip)

    try:
        image = Image.open(image_path_str).convert('RGB')
    except FileNotFoundError:
        print(f"Error: Image file not found: {image_path_str}. Skipping.")
        return False
    except Exception as e:
        print(f"Error opening image {image_path_str}: {e}. Skipping.")
        return False

    try:
        # 1. Prepare input for the model (resize to input_processing_resolution)
        # Mask2Former processor handles resizing internally based on its settings,
        # but we specify the image here. If specific input size is crucial:
        # image_for_input = image.resize(input_processing_resolution, Image.BILINEAR) # Or LANCZOS
        # inputs = image_processor(images=image_for_input, return_tensors="pt")

        # Simpler: Let processor handle sizing based on its config or common practice
        inputs = image_processor(images=image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # 2. Perform inference
        with torch.no_grad():
            outputs = model(**inputs)

        # 3. Post-process to get the semantic map (at original image size or specified target)
        # We get it at the model's processing size first for accuracy.
        # Note: The processor might have internal size logic. Check its behavior.
        # If we want the mask initially at input_processing_resolution:
        predicted_semantic_map = image_processor.post_process_semantic_segmentation(
            outputs, target_sizes=[input_processing_resolution[::-1]] # Needs (height, width)
        )[0]
        # If we want it closer to original size initially (might be memory intensive for large images):
        # predicted_semantic_map = image_processor.post_process_semantic_segmentation(
        #     outputs, target_sizes=[image.size[::-1]] # Needs (height, width)
        # )[0]


        # 4. Convert to NumPy array (Grayscale mask with class IDs)
        mask_array = predicted_semantic_map.cpu().numpy().astype(np.uint8) # Use uint8 if < 256 classes

        # 5. Create PIL Image from the array ('L' mode for grayscale)
        mask_image = Image.fromarray(mask_array, mode='L')

        # 6. Resize to final output resolution (if specified) using NEAREST neighbor
        if output_resolution:
            if mask_image.size != output_resolution:
                mask_image = mask_image.resize(output_resolution, Image.NEAREST)

        # 7. Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # 8. Save the grayscale mask
        mask_image.save(output_path, format='PNG')
        return True

    except Exception as e:
        print(f"Error processing image {image_path_str}: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    parser = argparse.ArgumentParser(description="Generate grayscale semantic segmentation masks for images listed in an aggregated JSON file.")
    parser.add_argument('-j', '--json-path', type=str, required=True,
                        help="Path to the aggregated JSON file containing image paths (output of run_multi_dataset_processing.py).")
    parser.add_argument('-d', '--dataset-type', type=str, required=True, choices=MODEL_MAPPING.keys(),
                        help="Type of dataset the model was trained on (determines model and labels).")
    parser.add_argument('-o', '--output-resolution', type=str, default=None,
                        help="Desired output resolution for the masks (e.g., '1024x512'). If None, uses the model's processing size.")
    parser.add_argument('--input-res', type=str, default=None,
                        help="Override model's input processing resolution (e.g., '512x512'). Uses dataset defaults if None.")
    parser.add_argument('--base-output-dir', type=str, default=None,
                        help="Optional base directory to store all '*_mask' folders. If None, places them alongside original dataset folders.")
    parser.add_argument('--metadata-dir', type=str, default=None,
                        help="Directory to save the label mapping metadata JSON file. Defaults to the directory of the input JSON.")
    parser.add_argument('--overwrite', action='store_true',
                        help="Overwrite existing mask files.")
    parser.add_argument('--limit', type=int, default=None,
                        help="Process only the first N images found in the JSON.")
    parser.add_argument('--skip-sequences', type=int, default=0,
                        help="Skip the first N sequences in the JSON file.")
    parser.add_argument('--images-per-sequence', type=int, default=None,
                         help="Process only the first N images within each sequence.")


    args = parser.parse_args()

    # --- Parse Resolutions ---
    final_output_resolution = None
    if args.output_resolution:
        try:
            w, h = map(int, args.output_resolution.lower().split('x'))
            final_output_resolution = (w, h)
            print(f"Target mask output resolution: {final_output_resolution}")
        except ValueError:
            print(f"Error: Invalid output resolution format '{args.output_resolution}'. Use WxH (e.g., 1024x512). Exiting.")
            sys.exit(1)

    input_processing_resolution = None
    if args.input_res:
         try:
            w, h = map(int, args.input_res.lower().split('x'))
            input_processing_resolution = (w, h)
         except ValueError:
            print(f"Error: Invalid input processing resolution format '{args.input_res}'. Use WxH. Exiting.")
            sys.exit(1)
    else:
        input_processing_resolution = DEFAULT_INPUT_RES.get(args.dataset_type)
        if not input_processing_resolution:
             print(f"Warning: No default input resolution for {args.dataset_type}. Model/processor defaults will be used.")
    if input_processing_resolution:
        print(f"Model input processing resolution: {input_processing_resolution}")


    # --- Device Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Load Model and Processor ---
    model_name = MODEL_MAPPING[args.dataset_type]
    print(f"Loading model: {model_name}...")
    try:
        image_processor = AutoImageProcessor.from_pretrained(model_name)
        # Note: Some processors might allow setting size directly, e.g.,
        # if input_processing_resolution:
        #     image_processor.size = {"height": input_processing_resolution[1], "width": input_processing_resolution[0]}

        model = Mask2FormerForUniversalSegmentation.from_pretrained(model_name)
        model.to(device)
        model.eval()
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model or processor '{model_name}': {e}")
        sys.exit(1)

    # --- Get Label Mapping ---
    id2label = model.config.id2label
    label2id = model.config.label2id
    print(f"Loaded {len(id2label)} labels for {args.dataset_type}.")

    # --- Save Metadata ---
    metadata_dir = Path(args.metadata_dir) if args.metadata_dir else Path(args.json_path).parent
    metadata_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = metadata_dir / f"{args.dataset_type}_label_mapping.json"
    metadata = {
        "model_name": model_name,
        "dataset_type": args.dataset_type,
        "id2label": {str(k): v for k, v in id2label.items()}, # Convert keys to strings for JSON
        "label2id": label2id,
        "note": "Grayscale values in the mask PNGs correspond to the IDs in id2label."
    }
    try:
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"Label mapping saved to: {metadata_path}")
    except IOError as e:
        print(f"Warning: Could not save metadata file: {e}")

    # --- Load Image Paths from JSON ---
    print(f"Loading image paths from: {args.json_path}")
    try:
        with open(args.json_path, 'r') as f:
            sequences_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Input JSON file not found at {args.json_path}")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {args.json_path}")
        sys.exit(1)

    # Flatten the list of sequences into a list of unique image paths
    all_image_paths = []
    seen_paths = set()
    image_count_in_json = 0

    if not isinstance(sequences_data, list):
        print(f"Error: Expected a list of sequences in {args.json_path}, found {type(sequences_data)}. Exiting.")
        sys.exit(1)

    # Apply sequence skipping
    sequences_to_process = sequences_data[args.skip_sequences:]
    print(f"Processing {len(sequences_to_process)} sequences (skipped {args.skip_sequences}).")

    for sequence in sequences_to_process:
        if not isinstance(sequence, list):
            print(f"Warning: Expected a list for a sequence, found {type(sequence)}. Skipping this item.")
            continue

        images_in_seq_count = 0
        for item in sequence:
             # Stop processing images in this sequence if the limit is reached
            if args.images_per_sequence is not None and images_in_seq_count >= args.images_per_sequence:
                break

            if isinstance(item, dict) and 'image_path' in item:
                path_str = item['image_path']
                image_count_in_json += 1
                if path_str not in seen_paths:
                    all_image_paths.append(path_str)
                    seen_paths.add(path_str)
                images_in_seq_count += 1
            else:
                print(f"Warning: Expected a dictionary with 'image_path' in sequence, found {type(item)}. Skipping.")


    print(f"Found {image_count_in_json} image entries across {len(sequences_to_process)} sequences.")
    print(f"Found {len(all_image_paths)} unique image paths to process.")

    # Apply overall limit if specified
    if args.limit is not None:
        all_image_paths = all_image_paths[:args.limit]
        print(f"Processing limit applied: {len(all_image_paths)} images.")

    # --- Process Images ---
    success_count = 0
    fail_count = 0
    pbar = tqdm(all_image_paths, desc="Generating masks")
    for image_path_str in pbar:
        output_mask_path = derive_output_path(image_path_str, args.base_output_dir)
        pbar.set_postfix({"Processing": os.path.basename(image_path_str)})

        # Determine the effective input resolution for this image
        # If per-image size needed, calculate here, else use the global one
        current_input_res = input_processing_resolution
        # If input_processing_resolution is None, the processor/model uses defaults

        if process_and_save_mask(
            image_path_str=image_path_str,
            model=model,
            image_processor=image_processor,
            device=device,
            output_resolution=final_output_resolution,
            input_processing_resolution=current_input_res, # Pass the determined input size
            output_path=output_mask_path,
            overwrite=args.overwrite
        ):
            success_count += 1
        else:
            fail_count += 1

    print("\n--- Processing Summary ---")
    print(f"Successfully processed/skipped: {success_count}")
    print(f"Failed to process: {fail_count}")
    print("Mask generation finished.")

if __name__ == "__main__":
    main()


# python mask2former_matrix_process.py \
#     --json-path /path/to/your/aggregated_sequences.json \
#     --dataset-type ade20k \
#     --output-resolution 512x512 \
#     --metadata-dir /path/to/your/output_metadata \
#     # --input-res 512x512 # Optional: Override default input size for ADE20k

