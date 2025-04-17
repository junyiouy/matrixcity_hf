from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation
from PIL import Image, ImageDraw
import torch
import numpy as np
import os
from pathlib import Path
import imageio  # For GIF creation
import json
from tqdm import tqdm  # For progress bar

# Your provided label color data
label_colors = {
    1: (120, 120, 120),  # wall
    2: (180, 120, 120),  # building
    3: (6, 230, 230),    # sky
    4: (80, 50, 50),     # floor
    5: (4, 200, 3),      # tree
    6: (120, 120, 80),   # ceiling
    7: (140, 140, 140),  # road
    8: (204, 5, 255),    # bed
    9: (230, 230, 230),  # windowpane
    10: (4, 250, 7),     # grass
    11: (224, 5, 255),   # cabinet
    12: (235, 255, 7),   # sidewalk
    13: (150, 5, 61),    # person
    14: (120, 120, 70),  # earth
    15: (8, 255, 51),    # door
    16: (255, 6, 82),    # table
    17: (143, 255, 140), # mountain
    18: (204, 255, 4),   # plant
    19: (255, 51, 7),    # curtain
    20: (204, 70, 3),    # chair
    21: (0, 102, 200),   # car
    22: (61, 230, 250),  # water
    23: (255, 6, 51),    # painting
    24: (11, 102, 255),  # sofa
    25: (255, 7, 71),    # shelf
    26: (255, 9, 224),   # house
    27: (9, 7, 230),     # sea
    28: (220, 220, 220), # mirror
    29: (255, 9, 92),    # rug
    30: (112, 9, 255),   # field
    31: (8, 255, 214),   # armchair
    32: (7, 255, 224),   # seat
    33: (255, 184, 6),   # fence
    34: (10, 255, 71),   # desk
    35: (255, 41, 10),   # rock
    36: (7, 255, 255),   # wardrobe
    37: (224, 255, 8),   # lamp
    38: (102, 8, 255),   # bathtub
    39: (255, 61, 6),    # railing
    40: (255, 194, 7),   # cushion
    41: (255, 122, 8),   # base
    42: (0, 255, 20),    # box
    43: (255, 8, 41),    # column
    44: (255, 5, 153),   # signboard
    45: (6, 51, 255),    # chest
    46: (235, 12, 255),  # counter
    47: (160, 150, 20),  # sand
    48: (0, 163, 255),   # sink
    49: (140, 140, 140), # skyscraper
    50: (250, 10, 15),   # fireplace
    51: (20, 255, 0),    # refrigerator
    52: (31, 255, 0),    # grandstand
    53: (255, 31, 0),    # path
    54: (255, 224, 0),   # stairs
    55: (153, 255, 0),   # runway
    56: (0, 0, 255),     # case
    57: (255, 71, 0),    # pool table
    58: (0, 235, 255),   # pillow
    59: (0, 173, 255),   # screen door
    60: (31, 0, 255),    # stairway
    61: (11, 200, 200),  # river
    62: (255, 82, 0),    # bridge
    63: (0, 255, 245),   # bookcase
    64: (0, 61, 255),    # blind
    65: (0, 255, 112),   # coffee table
    66: (0, 255, 133),   # toilet
    67: (255, 0, 0),     # flower
    68: (255, 163, 0),   # book
    69: (255, 102, 0),   # hill
    70: (194, 255, 0),   # bench
    71: (0, 143, 255),   # countertop
    72: (51, 255, 0),    # stove
    73: (0, 82, 255),    # palm tree
    74: (0, 255, 41),    # kitchen island
    75: (0, 255, 173),   # computer
    76: (10, 0, 255),    # swivel chair
    77: (173, 255, 0),   # boat
    78: (0, 255, 153),   # bar
    79: (255, 92, 0),    # arcade machine
    80: (255, 0, 255),   # hovel
    81: (255, 0, 245),   # bus
    82: (255, 0, 102),   # towel
    83: (255, 173, 0),   # light source
    84: (255, 0, 20),    # truck
    85: (255, 184, 184), # tower
    86: (0, 31, 255),    # chandelier
    87: (0, 255, 61),    # awning
    88: (0, 71, 255),    # streetlight
    89: (255, 0, 204),   # booth
    90: (0, 255, 194),   # television
    91: (0, 255, 82),    # airplane
    92: (0, 10, 255),    # dirt track
    93: (0, 112, 255),   # apparel
    94: (51, 0, 255),    # pole
    95: (0, 194, 255),   # land
    96: (0, 122, 255),   # bannister
    97: (0, 255, 163),   # escalator
    98: (255, 153, 0),   # ottoman
    99: (0, 255, 10),    # bottle
    100: (255, 112, 0),  # buffet
    101: (143, 255, 0),  # poster
    102: (82, 0, 255),   # stage
    103: (163, 255, 0),  # van
    104: (255, 235, 0),  # ship
    105: (8, 184, 170),   # fountain
    106: (133, 0, 255),  # conveyer belt
    107: (0, 255, 92),   # canopy
    108: (184, 0, 255),  # washer
    109: (255, 0, 31),   # plaything
    110: (0, 184, 255),  # swimming pool
    111: (0, 214, 255),  # stool
    112: (255, 0, 112),  # barrel
    113: (92, 255, 0),    # basket
    114: (0, 224, 255),  # waterfall
    115: (112, 224, 255), # tent
    116: (70, 184, 160),  # bag
    117: (163, 0, 255),  # minibike
    118: (153, 0, 255),  # cradle
    119: (71, 255, 0),    # oven
    120: (255, 0, 163),  # ball
    121: (255, 204, 0),  # food
    122: (255, 0, 143),  # step
    123: (0, 255, 235),  # tank
    124: (133, 255, 0),  # trade name
    125: (255, 0, 235),  # microwave
    126: (245, 0, 255),  # pot flowerpot
    127: (255, 0, 122),  # animal
    128: (255, 245, 0),  # bicycle
    129: (10, 190, 212),  # lake
    130: (214, 255, 0),  # dishwasher
    131: (0, 204, 255),  # screen silver screen
    132: (20, 0, 255),  # blanket
    133: (255, 255, 0),  # sculpture
    134: (0, 153, 255),  # hood exhaust hood
    135: (0, 41, 255),  # sconce
    136: (0, 255, 204),  # vase
    137: (41, 0, 255),  # traffic light traffic signal stoplight
    138: (41, 255, 0),  # tray
    139: (173, 0, 255),  # ashcan trash can garbage can wastebin ash bin ash-bin ashbin dustbin trash barrel trash bin
    140: (0, 245, 255),  # fan
    141: (71, 0, 255),  # pier wharf wharfage dock
    142: (122, 0, 255),  # crt screen
    143: (0, 255, 184),  # plate
    144: (0, 92, 255),  # monitor monitoring device
    145: (184, 255, 0),  # bulletin board notice board
    146: (0, 133, 255),  # shower
    147: (255, 214, 0),  # radiator
    148: (25, 194, 194),  # glass drinking glass
    149: (102, 255, 0),  # clock
    150: (92, 0, 255)   # flag
}
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = len(label_colors) + 1

feature_extractor = SegformerFeatureExtractor.from_pretrained("nvidia/segformer-b5-finetuned-ade-640-640")
model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b5-finetuned-ade-640-640")
model.to(device)  # Move model to the appropriate device

model.eval()  # Set the model to evaluation mode

def process_sequence(image_paths, output_gif_path="output.gif"):
    """Processes a sequence of images, performs segmentation, and creates a GIF."""

    images = []  # To store processed images for GIF
    for image_path in tqdm(image_paths, desc="Processing images"):
        try:
            image = Image.open(image_path).convert('RGB').resize((256,256))
            inputs = feature_extractor(images=image, return_tensors="pt")
            inputs['pixel_values']=inputs['pixel_values'].to(device)  # Move inputs to the appropriate device
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits

            # Resize logits to original image size
            upsampled_logits = torch.nn.functional.interpolate(
                logits,
                size=image.size[::-1],
                mode="bilinear",
                align_corners=False,
            )

            # Get predicted class labels
            predicted_labels = torch.argmax(upsampled_logits.squeeze(), dim=0)
            predicted_labels = predicted_labels.cpu().numpy()

            # get unique labels
            unique_labels = np.unique(predicted_labels)

            # Apply custom colormap to predicted labels
            height, width = predicted_labels.shape
            colored_mask = np.zeros((height, width, 3), dtype=np.uint8)
            for i in range(height):
                for j in range(width):
                    label_id = predicted_labels[i, j]
                    if label_id in label_colors:
                        colored_mask[i, j] = label_colors[label_id]
                    else:
                        colored_mask[i, j] = (0, 0, 0)  # Black for unknown

            colored_mask_pil = Image.fromarray(colored_mask)

            # Combine original image and colored mask side by side
            combined_image = Image.new("RGB", (image.width + colored_mask_pil.width, image.height))
            combined_image.paste(image, (0, 0))
            combined_image.paste(colored_mask_pil, (image.width, 0))

            images.append(combined_image)

        except Exception as e:
            print(f"Error processing image {image_path}: {e}")

    # Save as GIF
    if images:
        imageio.mimsave(output_gif_path, [np.array(img) for img in images], duration=0.1)  # 100ms per frame
        print(f"GIF saved to {output_gif_path}")
    else:
        print("No images were successfully processed.")

# Example Usage:
if __name__ == "__main__":

    json_path = "/wekafs/ict/junyiouy/matrixcity_hf/small_city1.json"
    output_path = "/wekafs/ict/junyiouy/matrixcity_hf/"
    output_gif_name = "segmented_sequence.gif"
    with open(json_path, 'r') as f:
        data = json.load(f)
    data=data[0]
    image_paths = [dict_['image_path'] for dict_ in data][:200]

    process_sequence(image_paths,os.path.join(output_path,output_gif_name))