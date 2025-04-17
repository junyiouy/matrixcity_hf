# --- Existing Imports ---
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
from PIL import Image, ImageDraw, ImageFont, ImageOps, ImageFilter # Added Resampling
import torch
import numpy as np
import os
from pathlib import Path
import imageio
import json
from tqdm import tqdm
from scipy.ndimage import center_of_mass
from collections import namedtuple # Needed for Cityscapes labels

# --- ADE20k Color Mapping (Keep for reference/use) ---
ade20k_label_colors = {
    1: (120, 120, 120), 2: (180, 120, 120), 3: (6, 230, 230), 4: (80, 50, 50), 5: (4, 200, 3),
    6: (120, 120, 80), 7: (140, 140, 140), 8: (204, 5, 255), 9: (230, 230, 230), 10: (4, 250, 7),
    11: (224, 5, 255), 12: (235, 255, 7), 13: (150, 5, 61), 14: (120, 120, 70), 15: (8, 255, 51),
    16: (255, 6, 82), 17: (143, 255, 140), 18: (204, 255, 4), 19: (255, 51, 7), 20: (204, 70, 3),
    21: (0, 102, 200), 22: (61, 230, 250), 23: (255, 6, 51), 24: (11, 102, 255), 25: (255, 7, 71),
    26: (255, 9, 224), 27: (9, 7, 230), 28: (220, 220, 220), 29: (255, 9, 92), 30: (112, 9, 255),
    31: (8, 255, 214), 32: (7, 255, 224), 33: (255, 184, 6), 34: (10, 255, 71), 35: (255, 41, 10),
    36: (7, 255, 255), 37: (224, 255, 8), 38: (102, 8, 255), 39: (255, 61, 6), 40: (255, 194, 7),
    41: (255, 122, 8), 42: (0, 255, 20), 43: (255, 8, 41), 44: (255, 5, 153), 45: (6, 51, 255),
    46: (235, 12, 255), 47: (160, 150, 20), 48: (0, 163, 255), 49: (140, 140, 140), 50: (250, 10, 15),
    51: (20, 255, 0), 52: (31, 255, 0), 53: (255, 31, 0), 54: (255, 224, 0), 55: (153, 255, 0),
    56: (0, 0, 255), 57: (255, 71, 0), 58: (0, 235, 255), 59: (0, 173, 255), 60: (31, 0, 255),
    61: (11, 200, 200), 62: (255, 82, 0), 63: (0, 255, 245), 64: (0, 61, 255), 65: (0, 255, 112),
    66: (0, 255, 133), 67: (255, 0, 0), 68: (255, 163, 0), 69: (255, 102, 0), 70: (194, 255, 0),
    71: (0, 143, 255), 72: (51, 255, 0), 73: (0, 82, 255), 74: (0, 255, 41), 75: (0, 255, 173),
    76: (10, 0, 255), 77: (173, 255, 0), 78: (0, 255, 153), 79: (255, 92, 0), 80: (255, 0, 255),
    81: (255, 0, 245), 82: (255, 0, 102), 83: (255, 173, 0), 84: (255, 0, 20), 85: (255, 184, 184),
    86: (0, 31, 255), 87: (0, 255, 61), 88: (0, 71, 255), 89: (255, 0, 204), 90: (0, 255, 194),
    91: (0, 255, 82), 92: (0, 10, 255), 93: (0, 112, 255), 94: (51, 0, 255), 95: (0, 194, 255),
    96: (0, 122, 255), 97: (0, 255, 163), 98: (255, 153, 0), 99: (0, 255, 10), 100: (255, 112, 0),
    101: (143, 255, 0), 102: (82, 0, 255), 103: (163, 255, 0), 104: (255, 235, 0), 105: (8, 184, 170),
    106: (133, 0, 255), 107: (0, 255, 92), 108: (184, 0, 255), 109: (255, 0, 31), 110: (0, 184, 255),
    111: (0, 214, 255), 112: (255, 0, 112), 113: (92, 255, 0), 114: (0, 224, 255), 115: (112, 224, 255),
    116: (70, 184, 160), 117: (163, 0, 255), 118: (153, 0, 255), 119: (71, 255, 0), 120: (255, 0, 163),
    121: (255, 204, 0), 122: (255, 0, 143), 123: (0, 255, 235), 124: (133, 255, 0), 125: (255, 0, 235),
    126: (245, 0, 255), 127: (255, 0, 122), 128: (255, 245, 0), 129: (10, 190, 212), 130: (214, 255, 0),
    131: (0, 204, 255), 132: (20, 0, 255), 133: (255, 255, 0), 134: (0, 153, 255), 135: (0, 41, 255),
    136: (0, 255, 204), 137: (41, 0, 255), 138: (41, 255, 0), 139: (173, 0, 255), 140: (0, 245, 255),
    141: (71, 0, 255), 142: (122, 0, 255), 143: (0, 255, 184), 144: (0, 92, 255), 145: (184, 255, 0),
    146: (0, 133, 255), 147: (255, 214, 0), 148: (25, 194, 194), 149: (102, 255, 0), 150: (92, 0, 255)
}

# --- Cityscapes Label Definitions ---
Label = namedtuple( 'Label' , [
    'name'        , 'id'          , 'trainId'     , 'category'    ,
    'categoryId'  , 'hasInstances', 'ignoreInEval', 'color'       ,
    ] )

cityscapes_labels_list = [
    Label(  'unlabeled'            ,  0 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'ego vehicle'          ,  1 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'rectification border' ,  2 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'out of roi'           ,  3 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'static'               ,  4 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'dynamic'              ,  5 ,      255 , 'void'            , 0       , False        , True         , (111, 74,  0) ),
    Label(  'ground'               ,  6 ,      255 , 'void'            , 0       , False        , True         , ( 81,  0, 81) ),
    Label(  'road'                 ,  7 ,        0 , 'flat'            , 1       , False        , False        , (128, 64,128) ),
    Label(  'sidewalk'             ,  8 ,        1 , 'flat'            , 1       , False        , False        , (244, 35,232) ),
    Label(  'parking'              ,  9 ,      255 , 'flat'            , 1       , False        , True         , (250,170,160) ),
    Label(  'rail track'           , 10 ,      255 , 'flat'            , 1       , False        , True         , (230,150,140) ),
    Label(  'building'             , 11 ,        2 , 'construction'    , 2       , False        , False        , ( 70, 70, 70) ),
    Label(  'wall'                 , 12 ,        3 , 'construction'    , 2       , False        , False        , (102,102,156) ),
    Label(  'fence'                , 13 ,        4 , 'construction'    , 2       , False        , False        , (190,153,153) ),
    Label(  'guard rail'           , 14 ,      255 , 'construction'    , 2       , False        , True         , (180,165,180) ),
    Label(  'bridge'               , 15 ,      255 , 'construction'    , 2       , False        , True         , (150,100,100) ),
    Label(  'tunnel'               , 16 ,      255 , 'construction'    , 2       , False        , True         , (150,120, 90) ),
    Label(  'pole'                 , 17 ,        5 , 'object'          , 3       , False        , False        , (153,153,153) ),
    Label(  'polegroup'            , 18 ,      255 , 'object'          , 3       , False        , True         , (153,153,153) ),
    Label(  'traffic light'        , 19 ,        6 , 'object'          , 3       , False        , False        , (250,170, 30) ),
    Label(  'traffic sign'         , 20 ,        7 , 'object'          , 3       , False        , False        , (220,220,  0) ),
    Label(  'vegetation'           , 21 ,        8 , 'nature'          , 4       , False        , False        , (107,142, 35) ),
    Label(  'terrain'              , 22 ,        9 , 'nature'          , 4       , False        , False        , (152,251,152) ),
    Label(  'sky'                  , 23 ,       10 , 'sky'             , 5       , False        , False        , ( 70,130,180) ),
    Label(  'person'               , 24 ,       11 , 'human'           , 6       , True         , False        , (220, 20, 60) ),
    Label(  'rider'                , 25 ,       12 , 'human'           , 6       , True         , False        , (255,  0,  0) ),
    Label(  'car'                  , 26 ,       13 , 'vehicle'         , 7       , True         , False        , (  0,  0,142) ),
    Label(  'truck'                , 27 ,       14 , 'vehicle'         , 7       , True         , False        , (  0,  0, 70) ),
    Label(  'bus'                  , 28 ,       15 , 'vehicle'         , 7       , True         , False        , (  0, 60,100) ),
    Label(  'caravan'              , 29 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0, 90) ),
    Label(  'trailer'              , 30 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0,110) ),
    Label(  'train'                , 31 ,       16 , 'vehicle'         , 7       , True         , False        , (  0, 80,100) ),
    Label(  'motorcycle'           , 32 ,       17 , 'vehicle'         , 7       , True         , False        , (  0,  0,230) ),
    Label(  'bicycle'              , 33 ,       18 , 'vehicle'         , 7       , True         , False        , (119, 11, 32) ),
    Label(  'license plate'        , -1 ,       -1 , 'vehicle'         , 7       , False        , True         , (  0,  0,142) ),
]

# Create a mapping from Cityscapes official NAME to the Label object
cityscapes_name2label = { label.name: label for label in cityscapes_labels_list }

def get_cityscapes_color_map(model_id2label):
    """
    Generates a color map mapping the model's output indices (0, 1, ...)
    to the corresponding Cityscapes RGB colors.
    """
    color_map = {}
    unknown_color = (0, 0, 0) # Default color for unknown/unmapped labels
    for index, name in model_id2label.items():
        cityscapes_label = cityscapes_name2label.get(name)
        if cityscapes_label:
            color_map[index] = cityscapes_label.color
        else:
            # Handle cases where model output name might differ slightly
            # or map background/unlabeled explicitly
            if name == "background" or name == "unlabeled":
                 color_map[index] = (0,0,0)
            else:
                 print(f"Warning: Model label name '{name}' (index {index}) not found in Cityscapes standard names. Using default color {unknown_color}.")
                 color_map[index] = unknown_color
    return color_map

# --- Global Settings ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Modified Function Signature ---
def process_sequence(
    image_paths,
    model,
    image_processor,
    id2label_map,
    color_map,
    output_gif_path="output.gif",
    output_resolution=None,
    input_processing_resolution=(1024, 1024) # Allow customizing input size
):
    """
    Processes a sequence of images, performs segmentation, adds labels correctly *after* resizing,
    and creates a GIF. Uses provided model, processor, id2label map, and color map.
    """

    images = []
    try:
        base_font_size = 15
        # Calculate font size based on the FINAL output resolution
        if output_resolution:
            # Ensure output_resolution is valid, default height to 512 if not
            ref_height = output_resolution[1] if isinstance(output_resolution, (list, tuple)) and len(output_resolution) == 2 and output_resolution[1] > 0 else 512
            scale_factor = ref_height / 512.0
            font_size = max(8, int(base_font_size * scale_factor))
        else:
             # If no output res, guess based on input size (might need adjustment)
             # Default to input_processing_resolution[1] if invalid
             ref_height = input_processing_resolution[1] if isinstance(input_processing_resolution, (list, tuple)) and len(input_processing_resolution) == 2 and input_processing_resolution[1] > 0 else 1024 # Default input height
             font_size = max(8, int(base_font_size * (ref_height / 512.0)))
        font = ImageFont.truetype("arial.ttf", font_size)
        print(f"Using font size: {font_size}")
    except IOError:
        print("Arial font not found, using default font.")
        font = ImageFont.load_default()
    except Exception as font_e:
        print(f"Error setting font size: {font_e}. Using default.")
        font = ImageFont.load_default()


    print(f"Processing images at input resolution: {input_processing_resolution}")

    for image_path in tqdm(image_paths, desc="Processing images"):
        try:
            image = Image.open(image_path).convert('RGB')
            original_image_size = image.size # Store original size if needed elsewhere

            # Resize for model input
            image_resized_for_input = image.resize(input_processing_resolution, Image.LANCZOS)

            inputs = image_processor(images=image_resized_for_input, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs)

            # Post-process to the *input processing* size
            predicted_semantic_map = image_processor.post_process_semantic_segmentation(
                outputs, target_sizes=[input_processing_resolution[::-1]] # (height, width)
            )[0]
            predicted_labels = predicted_semantic_map.cpu().numpy() # Still at input_processing_resolution

            # --- Create colored mask (at input_processing_resolution) ---
            height_proc, width_proc = predicted_labels.shape
            colored_mask_np = np.zeros((height_proc, width_proc, 3), dtype=np.uint8)
            unique_labels_in_image = np.unique(predicted_labels)

            # Store centers calculated at processing resolution
            processing_centers = {}
            for label_id in unique_labels_in_image: # Renamed from label_id_ade
                # --- Apply the correct color mapping ---
                color = color_map.get(label_id, (0, 0, 0)) # Use the passed color_map
                # --------------------------------------
                binary_mask = (predicted_labels == label_id)
                colored_mask_np[binary_mask] = color

                # --- Calculate center of mass at PROCESSING resolution ---
                class_name = id2label_map.get(label_id, "Unknown") # Use passed id2label_map
                # Filter out background/unknown classes explicitly if needed
                # (Cityscapes often has 'unlabeled' or similar, ADE20k might too)
                ignore_classes = {"background", "unknown", "unlabeled", "ego vehicle",
                                  "rectification border", "out of roi", "static", "dynamic",
                                  "ground", "parking", "rail track", "guard rail", "bridge",
                                  "tunnel", "polegroup", "caravan", "trailer", "license plate"} # Add more if needed
                if class_name.lower() not in ignore_classes:
                     # Adjust min segment size relative to processing resolution
                     min_segment_pixels = int(input_processing_resolution[0] * input_processing_resolution[1] * 0.00005) # 0.005% area
                     min_segment_pixels = max(50, min_segment_pixels) # Set a minimum floor
                     if binary_mask.sum() >= min_segment_pixels:
                         try:
                             # Calculate center of mass on the binary mask at processing resolution
                             cm = center_of_mass(binary_mask)
                             # Store processing resolution (y, x) coordinates
                             processing_centers[label_id] = (cm[0], cm[1])
                         except ValueError:
                             pass # Ignore if center cannot be calculated
                # ---------------------------------------------------------

            colored_mask_pil = Image.fromarray(colored_mask_np) # At input_processing_resolution

            # --- Determine Final Output Canvas Size ---
            if output_resolution:
                if not (isinstance(output_resolution, (list, tuple)) and len(output_resolution) == 2 and output_resolution[0] > 0 and output_resolution[1] > 0):
                    print(f"Warning: Invalid output_resolution {output_resolution}. Using native combined size.")
                    output_width = input_processing_resolution[0] * 2
                    output_height = input_processing_resolution[1]
                else:
                    output_width, output_height = output_resolution
            else:
                # Default to native combined size if no output resolution specified
                output_width = input_processing_resolution[0] * 2
                output_height = input_processing_resolution[1]
            # -------------------------------------------

            # --- Calculate target sizes for left and right panels based on output canvas ---
            target_left_width = output_width // 2
            target_right_width = output_width - target_left_width
            target_panel_height = output_height

            target_left_size = (target_left_width, target_panel_height)
            target_right_size = (target_right_width, target_panel_height)
            # -------------------------------------------------------

            # --- Resize image (from processing size) and UNLABELED mask (from processing size) separately ---
            # Resize the image that was fed into the model
            resized_image_panel = image_resized_for_input.resize(target_left_size, Image.LANCZOS)
            # Resize the colored mask generated at processing resolution
            resized_mask_panel = colored_mask_pil.resize(target_right_size, Image.LANCZOS) # Resize the unlabeled mask
            # ------------------------------------------------

            # --- Create final canvas and paste resized panels ---
            final_canvas = Image.new("RGB", (output_width, output_height))
            final_canvas.paste(resized_image_panel, (0, 0))
            final_canvas.paste(resized_mask_panel, (target_left_width, 0))
            # ---------------------------------------------------

            # --- Convert final canvas to RGBA for drawing text ---
            final_canvas_rgba = final_canvas.convert("RGBA")
            draw = ImageDraw.Draw(final_canvas_rgba)
            # ---------------------------------------------------

            # --- Draw Text Labels on Final Canvas using Scaled Coordinates ---
            for label_id, proc_center in processing_centers.items():
                class_name = id2label_map.get(label_id, "Unknown") # Use passed id2label_map

                # Scale the processing center coordinates to the resized mask panel
                proc_y, proc_x = proc_center
                # Scale factor = target_dimension / processing_dimension
                scale_x = target_right_width / width_proc
                scale_y = target_panel_height / height_proc

                # Calculate position within the *right panel*
                scaled_x_in_panel = int(proc_x * scale_x)
                scaled_y_in_panel = int(proc_y * scale_y)

                # Add the offset of the right panel to get coordinates on the final canvas
                final_text_x = target_left_width + scaled_x_in_panel
                final_text_y = scaled_y_in_panel
                text_pos = (final_text_x, final_text_y)

                try:
                    # Draw text background and text
                    bg_color = (0, 0, 0, 150)
                    # Use textbbox on the final draw object
                    bbox = draw.textbbox(text_pos, class_name, font=font, anchor="mm")
                    # Add padding to bbox
                    padding = max(1, font_size // 8)
                    bbox = (bbox[0]-padding, bbox[1]-padding, bbox[2]+padding, bbox[3]+padding)
                    draw.rectangle(bbox, fill=bg_color)
                    draw.text(text_pos, class_name, fill="white", font=font, anchor="mm")
                except Exception as draw_e:
                    print(f"Error drawing text for label {label_id} ('{class_name}') at {text_pos}: {draw_e}")
            # ------------------------------------------------------------------

            # --- Append the final image with text ---
            # Convert back to RGB if needed by imageio (usually handles RGBA ok)
            images.append(final_canvas_rgba.convert("RGB"))
            # ----------------------------------------

        except FileNotFoundError:
             print(f"Error: Image file not found at {image_path}. Skipping.")
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            import traceback
            traceback.print_exc()

    # Save as GIF
    if images:
        print(f"Saving GIF with {len(images)} frames...")
        try:
            # Ensure duration is reasonable (e.g., 10 frames per second)
            frame_duration = 0.1 # seconds per frame
            imageio.mimsave(output_gif_path, [np.array(img) for img in images], duration=frame_duration)
            print(f"GIF saved to {output_gif_path}")
        except Exception as gif_e:
            print(f"Error saving GIF: {gif_e}")
    else:
        print("No images were successfully processed.")


# --- Example Usage ---
if __name__ == "__main__":

    # --- Configuration ---
    # Choose dataset: "ade20k" or "cityscapes"
    TARGET_DATASET = "ade20k" # <--- CHANGE THIS TO SWITCH MODELS

    # Define model names and color maps
    if TARGET_DATASET == "ade20k":
        model_name = "facebook/mask2former-swin-large-ade-semantic"
        # For ADE20k, the model output 0-149 maps directly to the desired label IDs 1-150 used in the original dict
        # Need to adjust the dict keys to be 0-149
        # Note: This assumes the model's id2label maps index 0 to the class corresponding to ID 1, etc.
        # It's safer to build this map based on model.config like for cityscapes if unsure.
        # Let's create a 0-indexed map for consistency:
        active_color_map = {i: ade20k_label_colors[i+1] for i in range(150) if (i+1) in ade20k_label_colors}
        # Add a default for index 0 if it represents 'background' or similar and isn't in the 1-150 map
        active_color_map.setdefault(0, (0, 0, 0)) # Assuming index 0 might be background/wall
        input_res = (1024, 1024) # ADE20k models often trained on 512x512
        dataset_name_for_file = "ade"

    elif TARGET_DATASET == "cityscapes":
        # Use a Cityscapes fine-tuned model
        model_name = "facebook/mask2former-swin-large-cityscapes-semantic"
        # Color map needs to be generated based on the *specific model's* id2label config
        # We will generate this after loading the model.
        active_color_map = None # Will be generated later
        input_res = (512, 512) # Cityscapes aspect ratio often 2:1, common input size
        # input_res = (2048, 1024) # Or higher resolution if GPU memory allows
        dataset_name_for_file = "cityscapes"
    else:
        raise ValueError(f"Unsupported TARGET_DATASET: {TARGET_DATASET}")

    print(f"Using dataset: {TARGET_DATASET}")
    print(f"Loading model: {model_name}")

    # --- Load Model and Processor ---
    image_processor = AutoImageProcessor.from_pretrained(model_name)
    # Adjust processor size if needed, although Mask2Former handles variable sizes
    # image_processor.size = {"height": input_res[1], "width": input_res[0]} # Can override default

    model = Mask2FormerForUniversalSegmentation.from_pretrained(model_name)
    model.to(device)
    model.eval()

    # --- Get Model Specific Mappings ---
    model_id2label_map = model.config.id2label

    # --- Generate Cityscapes Color Map if Needed ---
    if TARGET_DATASET == "cityscapes":
        print("Generating Cityscapes color map from model config...")
        active_color_map = get_cityscapes_color_map(model_id2label_map)
        if not active_color_map:
             raise RuntimeError("Failed to generate Cityscapes color map.")

    # --- Paths and Parameters ---
    json_path = "/wekafs/ict/junyiouy/matrixcity_hf/small_city1.json" # Adjust if needed
    output_path = "/wekafs/ict/junyiouy/matrixcity_hf/" # Adjust if needed
    final_output_resolution = (1024, 512) # Example: HD aspect ratio, adjust as needed
    #final_output_resolution = None # Set to None to use native combined size (based on input_res * 2 width)

    print(f"Target input processing resolution: {input_res}")
    print(f"Target output resolution for GIF frames: {final_output_resolution if final_output_resolution else 'native combined'}")

    # --- Load Data ---
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: JSON file not found at {json_path}")
        exit()
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {json_path}")
        exit()


    num_sequences_to_process = 5
    images_per_sequence = 20

    # --- Process Sequences ---
    for i in range(0, num_sequences_to_process * images_per_sequence, images_per_sequence):
        if i >= len(data):
            print(f"Sequence index {i} out of bounds ({len(data)} items). Stopping.")
            break

        print(f"\nProcessing sequence starting at index {i}...")
        # Check if data[i] is the expected list format
        current_sequence_data = data[i]
        if isinstance(current_sequence_data, list):
             image_paths_current = [d['image_path'] for d in current_sequence_data[:images_per_sequence] if isinstance(d, dict) and 'image_path' in d]
        # Adapt if the structure is different, e.g., just a list of paths
        # elif isinstance(current_sequence_data, str): # Example: If data[i] is just a path
        #     image_paths_current = data[i:i+images_per_sequence] # This assumes data is a flat list of paths
        else:
             print(f"Warning: Expected list at data[{i}], found {type(current_sequence_data)}. Skipping.")
             continue

        if not image_paths_current:
             print(f"No valid image paths found for sequence index {i}. Skipping.")
             continue

        # Construct output filename
        res_str = f"{final_output_resolution[0]}x{final_output_resolution[1]}" if final_output_resolution else "native"
        output_gif_name = f"segmented_{dataset_name_for_file}_{i}_{res_str}.gif"
        full_output_path = os.path.join(output_path, output_gif_name)

        # Ensure output directory exists
        Path(output_path).mkdir(parents=True, exist_ok=True)

        process_sequence(
            image_paths=image_paths_current,
            model=model,
            image_processor=image_processor,
            id2label_map=model_id2label_map,
            color_map=active_color_map,
            output_gif_path=full_output_path,
            output_resolution=final_output_resolution,
            input_processing_resolution=input_res
        )

    print("\nProcessing finished.")