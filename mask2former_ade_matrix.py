from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
from PIL import Image, ImageDraw, ImageFont # Added ImageDraw, ImageFont
import torch
import numpy as np
import os
from pathlib import Path
import imageio
import json
from tqdm import tqdm
from scipy.ndimage import center_of_mass # Added scipy import

# --- Your label_colors dictionary remains the same ---
label_colors = {
    # ... (your label_colors dictionary remains the same) ...
    1: (120, 120, 120),  # wall
    # ... (rest of your colors)
    150: (92, 0, 255)   # flag
}

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Model Loading (Use the id2label mapping) ---
model_name = "facebook/mask2former-swin-large-ade-semantic"
image_processor = AutoImageProcessor.from_pretrained(model_name)
model = Mask2FormerForUniversalSegmentation.from_pretrained(model_name)
model.to(device)
model.eval()

# --- Get the ID to Label mapping from the model config ---
id2label = model.config.id2label
# -------------------------------------------------------

def process_sequence(image_paths, output_gif_path="output.gif"):
    """Processes a sequence of images, performs segmentation, adds labels, and creates a GIF."""

    images = []
    # --- Load a font (optional, default font can be used too) ---
    try:
        # Use a common TrueType font if available, adjust path if needed
        font = ImageFont.truetype("arial.ttf", 15)
    except IOError:
        print("Arial font not found, using default font.")
        font = ImageFont.load_default()
    # -------------------------------------------------------------

    for image_path in tqdm(image_paths, desc="Processing images"):
        try:
            image = Image.open(image_path).convert('RGB').resize((256,256))
            original_size = image.size[::-1] # (height, width)

            inputs = image_processor(images=image, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs)

            predicted_semantic_map = image_processor.post_process_semantic_segmentation(
                outputs, target_sizes=[original_size]
            )[0]
            predicted_labels = predicted_semantic_map.cpu().numpy() # Shape: (height, width)

            # --- Create colored mask (similar to before, check mapping) ---
            height, width = predicted_labels.shape
            colored_mask = np.zeros((height, width, 3), dtype=np.uint8)
            unique_labels_in_image = np.unique(predicted_labels)

            for label_id_ade in unique_labels_in_image: # Iterate only labels present
                # Map ADE20k ID (0-149) to your label_colors key (1-150) - VERIFY THIS MAPPING!
                label_id_mapped = label_id_ade + 1
                if label_id_mapped in label_colors:
                    color = label_colors[label_id_mapped]
                else:
                    color = (0, 0, 0) # Black for background or unknown
                colored_mask[predicted_labels == label_id_ade] = color

            colored_mask_pil = Image.fromarray(colored_mask)
            # -----------------------------------------------------------

            # --- Overlay Text Labels ---
            draw = ImageDraw.Draw(colored_mask_pil)
            for label_id_ade in unique_labels_in_image:
                # Get class name using the model's id2label mapping
                class_name = id2label.get(label_id_ade, "Unknown")

                # Skip background or very small segments if desired
                if class_name == "background" or class_name == "Unknown": # ADE20k often has no explicit 'background' label
                    continue

                # Create a binary mask for the current label
                binary_mask = (predicted_labels == label_id_ade)

                # Avoid labeling tiny noise segments
                if binary_mask.sum() < 50: # Heuristic: skip if less than 50 pixels
                    continue

                # Calculate center of mass for label placement
                try:
                    cm = center_of_mass(binary_mask)
                    # Coordinates are (row, col) -> (y, x) for PIL
                    center_y, center_x = int(cm[0]), int(cm[1])

                    # Draw text - adjust color/size as needed
                    # Add a small black outline for better visibility
                    text_pos = (center_x, center_y)
                    bbox = draw.textbbox(text_pos, class_name, font=font, anchor="mm") # Get bounding box
                    draw.rectangle(bbox, fill=(0,0,0,128)) # Semi-transparent black background
                    draw.text(text_pos, class_name, fill="white", font=font, anchor="mm") # anchor='mm' centers text

                except ValueError:
                    # Can happen if the segment is empty or causes issues
                    print(f"Could not calculate center of mass for label {label_id_ade}")
                    pass # Skip labeling this segment
            # --------------------------

            image_display = image
            combined_image = Image.new("RGB", (image_display.width + colored_mask_pil.width, image_display.height))
            combined_image.paste(image_display, (0, 0))
            combined_image.paste(colored_mask_pil, (image_display.width, 0)) # Paste the mask with text

            images.append(combined_image)

        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            import traceback
            traceback.print_exc()

    # Save as GIF (ensure consistent sizing)
    if images:
        first_img_size = images[0].size
        uniform_images = []
        for img in images:
            if img.size == first_img_size:
                uniform_images.append(np.array(img))
            else:
                print(f"Warning: Resizing image for GIF from {img.size} to {first_img_size}")
                uniform_images.append(np.array(img.resize(first_img_size)))

        if uniform_images:
             imageio.mimsave(output_gif_path, uniform_images, duration=0.1)
             print(f"GIF saved to {output_gif_path}")
        else:
            print("No images with consistent size found to save GIF.")
    else:
        print("No images were successfully processed.")


# --- Example Usage (Unchanged) ---
if __name__ == "__main__":
    json_path = "/wekafs/ict/junyiouy/matrixcity_hf/small_city1.json"
    output_path = "/wekafs/ict/junyiouy/matrixcity_hf/"
    output_gif_name = "segmented_labeled_sequence_mask2former.gif" # Changed name
    with open(json_path, 'r') as f:
        data = json.load(f)
    data=data[0]
    image_paths = [dict_['image_path'] for dict_ in data][:50] # Use fewer images for testing

    process_sequence(image_paths,os.path.join(output_path,output_gif_name))