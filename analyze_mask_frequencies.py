# analyze_mask_frequencies.py

import os
import json
import argparse
import sys
from pathlib import Path
import random
from collections import defaultdict
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

# --- Helper Function (copied from generate_segmentation_masks.py if needed) ---
def derive_output_path(image_path_str: str, base_output_dir: str = None) -> Path:
    """
    Derives the expected output path for the mask based on the input image path.
    Handles both relative placement and placement within a base_output_dir.
    """
    image_path = Path(image_path_str)
    original_parent_name = image_path.parent.name
    mask_parent_name = original_parent_name + "_mask"

    if base_output_dir:
        # If base_output_dir was used during generation, masks are inside it
        output_parent_dir = Path(base_output_dir) / mask_parent_name
    else:
        # Otherwise, mask dir is alongside the original image directory's parent
        output_parent_dir = image_path.parent.parent / mask_parent_name

    output_path = output_parent_dir / image_path.name.replace(image_path.suffix, '.png') # Ensure mask is png
    return output_path

# --- Core Analysis Functions ---

def find_mask_paths(input_source: str, is_json: bool, base_mask_dir: str = None) -> list[Path]:
    """Finds all mask PNG files based on the input source."""
    mask_paths = []
    if is_json:
        print(f"Reading image paths from JSON: {input_source}")
        try:
            with open(input_source, 'r') as f:
                sequences_data = json.load(f)
        except FileNotFoundError:
            print(f"Error: Input JSON file not found at {input_source}")
            sys.exit(1)
        except json.JSONDecodeError:
            print(f"Error: Could not decode JSON from {input_source}")
            sys.exit(1)

        if not isinstance(sequences_data, list):
             print(f"Error: Expected a list of sequences in {input_source}, found {type(sequences_data)}. Exiting.")
             sys.exit(1)

        seen_mask_paths = set()
        for sequence in sequences_data:
             if not isinstance(sequence, list):
                 print(f"Warning: Expected a list for a sequence, found {type(sequence)}. Skipping.")
                 continue
             for item in sequence:
                 if isinstance(item, dict) and 'image_path' in item:
                     image_path_str = item['image_path']
                     # Derive the corresponding mask path
                     mask_path = derive_output_path(image_path_str, base_mask_dir)
                     if mask_path not in seen_mask_paths:
                         if mask_path.is_file(): # Check if the derived mask path actually exists
                            mask_paths.append(mask_path)
                            seen_mask_paths.add(mask_path)
                         else:
                             # This can happen if generation skipped images or base_mask_dir is wrong
                             # print(f"Warning: Derived mask path not found: {mask_path}")
                             pass # Avoid flooding console, check final count later
                 # else: print(f"Warning: Invalid item format in sequence: {item}")

    else: # Input source is a directory containing *_mask folders
        print(f"Scanning for '*_mask' directories in: {input_source}")
        base_dir = Path(input_source)
        if not base_dir.is_dir():
             print(f"Error: Input directory not found: {base_dir}")
             sys.exit(1)
        for potential_mask_dir in base_dir.iterdir():
            if potential_mask_dir.is_dir() and potential_mask_dir.name.endswith("_mask"):
                print(f"  Scanning masks in: {potential_mask_dir.name}")
                for item in potential_mask_dir.iterdir():
                     # Look for PNG files, assuming masks are saved as PNG
                     if item.is_file() and item.suffix.lower() == '.png':
                         mask_paths.append(item)

    if not mask_paths:
         print("Error: No mask files found. Check input path, format, or --base-output-dir-masks if using JSON.")
         sys.exit(1)

    print(f"Found {len(mask_paths)} potential mask files.")
    return mask_paths

def analyze_masks(mask_paths: list[Path], sample_size: int | None) -> dict[int, int]:
    """Analyzes masks to count the frequency of each class ID across images."""
    if sample_size is not None and sample_size < len(mask_paths):
        print(f"Sampling {sample_size} masks out of {len(mask_paths)}.")
        paths_to_process = random.sample(mask_paths, sample_size)
    else:
        print(f"Processing all {len(mask_paths)} found masks.")
        paths_to_process = mask_paths

    class_frequency = defaultdict(int) # Key: class_id (int), Value: count (int)
    processed_count = 0
    error_count = 0

    pbar = tqdm(paths_to_process, desc="Analyzing masks")
    for mask_path in pbar:
        try:
            # Open mask image (ensure mode allows reading raw pixel values)
            # 'L' for 8-bit grayscale, 'I' for 32-bit integer, etc.
            # Assuming masks were saved as 8-bit PNG ('L' mode)
            mask_image = Image.open(mask_path).convert('L') # Use convert('L') for safety
            mask_array = np.array(mask_image)

            # Find unique class IDs present in this mask
            unique_ids = np.unique(mask_array)

            # Increment frequency count for each unique ID found in this image
            for class_id in unique_ids:
                class_frequency[int(class_id)] += 1 # Ensure key is int

            processed_count += 1
        except FileNotFoundError:
             print(f"\nWarning: Mask file not found during analysis: {mask_path}. Skipping.")
             error_count += 1
        except Exception as e:
            print(f"\nWarning: Error processing mask {mask_path}: {e}. Skipping.")
            error_count += 1

    print(f"\nAnalysis finished. Processed: {processed_count}, Errors: {error_count}")
    return dict(class_frequency)

def plot_frequencies(frequency_data: list[dict], top_n: int, output_path: Path):
    """Generates and saves a bar chart of the top N class frequencies."""
    if not frequency_data:
        print("No frequency data to plot.")
        return

    # Sort data by frequency (descending) - should already be sorted, but double-check
    sorted_data = sorted(frequency_data, key=lambda x: x['frequency_count'], reverse=True)

    # Get top N items
    top_data = sorted_data[:top_n]

    if not top_data:
        print(f"No data available to plot for top {top_n}.")
        return

    class_names = [item['class_name'] for item in top_data]
    counts = [item['frequency_count'] for item in top_data]

    plt.style.use('seaborn-v0_8-darkgrid') # Use a nice style
    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.5))) # Adjust height based on N

    # Horizontal bar chart is usually better for category labels
    bars = ax.barh(class_names, counts, color='skyblue')

    ax.set_xlabel("Frequency (Number of Images Containing Class)")
    ax.set_ylabel("Class Name")
    ax.set_title(f"Top {min(top_n, len(class_names))} Most Frequent Semantic Classes in Sampled Images")
    ax.invert_yaxis()  # Display the most frequent class at the top

    # Add count labels to the bars
    ax.bar_label(bars, padding=3, fmt='%d')

    # Adjust layout and save
    plt.tight_layout()
    try:
        plt.savefig(output_path, dpi=150)
        print(f"Frequency plot saved to: {output_path}")
    except Exception as e:
        print(f"Error saving plot: {e}")
    finally:
        plt.close(fig) # Close the figure to free memory

# --- Main Execution ---

def main():
    parser = argparse.ArgumentParser(description="Analyze frequency of semantic classes in segmentation masks.")
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--mask-dir', type=str,
                             help="Directory containing '*_mask' subdirectories with mask PNG files.")
    input_group.add_argument('--json-path', type=str,
                             help="Path to the aggregated JSON file listing original image paths (masks will be derived).")

    parser.add_argument('--base-output-dir-masks', type=str, default=None,
                        help="[Optional] If using --json-path, specify the base output directory where '*_mask' folders were created by the generation script (passed to derive_output_path).")
    parser.add_argument('-m', '--metadata-json', type=str, required=True,
                        help="Path to the '_label_mapping.json' file corresponding to the masks.")
    parser.add_argument('-o', '--output-dir', type=str, required=True,
                        help="Directory to save the frequency JSON and plot.")
    parser.add_argument('-s', '--sample-size', type=int, default=None,
                        help="Number of masks to randomly sample for analysis. Processes all if omitted.")
    parser.add_argument('--top-n', type=int, default=10,
                        help="Number of top classes to include in the histogram plot.")
    parser.add_argument('--plot-filename', type=str, default="frequency_histogram.png",
                        help="Filename for the output plot.")
    parser.add_argument('--json-filename', type=str, default="class_frequencies.json",
                        help="Filename for the output frequency JSON.")

    args = parser.parse_args()

    # --- Validate Inputs ---
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = Path(args.metadata_json)
    if not metadata_path.is_file():
        print(f"Error: Metadata JSON not found at {metadata_path}")
        sys.exit(1)

    # --- Load Metadata ---
    print(f"Loading metadata from: {metadata_path}")
    try:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        # Ensure id2label keys are strings for lookup, as saved by the generation script
        id2label = metadata.get('id2label', {})
        if not id2label:
             print("Error: 'id2label' mapping not found or empty in metadata JSON.")
             sys.exit(1)
        print(f"Loaded {len(id2label)} class labels.")
    except Exception as e:
        print(f"Error loading or parsing metadata JSON: {e}")
        sys.exit(1)

    # --- Find Mask Files ---
    is_json_input = args.json_path is not None
    input_source = args.json_path if is_json_input else args.mask_dir
    mask_paths = find_mask_paths(input_source, is_json_input, args.base_output_dir_masks)

    # --- Analyze Frequencies ---
    class_counts = analyze_masks(mask_paths, args.sample_size) # Returns {int_id: count}

    # --- Prepare Output Data ---
    frequency_data = []
    for class_id, count in class_counts.items():
        # Metadata JSON has string keys for IDs
        class_name = id2label.get(str(class_id), f"Unknown ID [{class_id}]")
        frequency_data.append({
            "class_id": class_id,
            "class_name": class_name,
            "frequency_count": count
        })
        if str(class_id) not in id2label:
            print(f"Warning: Class ID {class_id} found in masks but not in metadata mapping.")

    # Sort by frequency (descending)
    frequency_data.sort(key=lambda x: x['frequency_count'], reverse=True)

    # --- Save Frequency JSON ---
    json_output_path = output_dir / args.json_filename
    try:
        with open(json_output_path, 'w') as f:
            json.dump(frequency_data, f, indent=2)
        print(f"Class frequency data saved to: {json_output_path}")
    except IOError as e:
        print(f"Error saving frequency JSON: {e}")

    # --- Generate and Save Plot ---
    plot_output_path = output_dir / args.plot_filename
    plot_frequencies(frequency_data, args.top_n, plot_output_path)

    print("\nAnalysis complete.")

if __name__ == "__main__":
    # Example Usage:
    # Using mask directory:
    # python analyze_mask_frequencies.py \
    #     --mask-dir /path/to/your/output_or_dataset_dir \
    #     --metadata-json /path/to/your/output_metadata/ade20k_label_mapping.json \
    #     --output-dir /path/to/your/analysis_results \
    #     --sample-size 5000 \
    #     --top-n 15

    # Using original JSON and derived paths (if masks stored relative to original images):
    # python analyze_mask_frequencies.py \
    #     --json-path /wekafs/ict/junyiouy/matrixcity_hf/small_city1.json \
    #     --metadata-json /wekafs/ict/junyiouy/matrixcity_hf/ade20k_label_mapping.json \
    #     --output-dir /wekafs/ict/junyiouy/matrixcity_hf/analysis_results \
    #     --sample-size 500

    # Using original JSON and derived paths (if masks stored under a base output dir):
    # python analyze_mask_frequencies.py \
    #     --json-path /path/to/your/aggregated_sequences.json \
    #     --base-output-dir-masks /output/masks \
    #     --metadata-json /path/to/your/output_metadata/ade20k_label_mapping.json \
    #     --output-dir /path/to/your/analysis_results \
    #     --sample-size 500

    # Ensure required libraries are installed:
    # pip install numpy Pillow matplotlib tqdm
    main()