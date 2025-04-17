import numpy as np
import json
import os
from PIL import Image, UnidentifiedImageError # Need Pillow: pip install Pillow
import imageio        # Need imageio: pip install imageio
import sys
import argparse
# --- Helper Functions ---

def get_pose_info(frame_data):
    """Extracts position, forward, and right vectors from frame data."""
    # Ensure matrix is numpy array
    matrix = np.array(frame_data['rot_mat'])
    # Ensure matrix is 4x4, pad if necessary (though unlikely for rot_mat)
    if matrix.shape != (4, 4):
        # Attempt basic correction if it's 3x4 (common in some formats)
        if matrix.shape == (3, 4):
             matrix = np.vstack([matrix, [0, 0, 0, 1]])
        else:
             # Handle other unexpected shapes if necessary, or raise error
             raise ValueError(f"Unexpected matrix shape: {matrix.shape} for frame index {frame_data.get('frame_index', 'N/A')}")

    position = matrix[:3, 3]
    right_vec = matrix[:3, 0]
    # Z axis points backwards, so Forward is -Z
    forward_vec = -matrix[:3, 2]

    # Normalize vectors, handle potential zero vectors
    fwd_norm = np.linalg.norm(forward_vec)
    rgt_norm = np.linalg.norm(right_vec)

    if fwd_norm > 1e-8:
        forward_vec /= fwd_norm
    else: # Handle zero vector case if necessary
        forward_vec = np.array([0, 0, -1]) # Default forward if matrix is degenerate

    if rgt_norm > 1e-8:
        right_vec /= rgt_norm
    else: # Handle zero vector case
        right_vec = np.array([1, 0, 0]) # Default right if matrix is degenerate

    return position, forward_vec, right_vec

def calculate_rotation_angle_degrees(rot_mat1, rot_mat2):
    """Calculates the angle of rotation (in degrees) between two rotation matrices."""
    # Relative rotation: R_rel = rot_mat2 @ rot_mat1.T
    # We only need the rotation part (3x3)
    R1 = rot_mat1[:3, :3]
    R2 = rot_mat2[:3, :3]

    # Normalize R1 and R2 to ensure they are valid rotation matrices
    try: # Avoid division by zero if determinant is zero
      det1 = np.linalg.det(R1)
      det2 = np.linalg.det(R2)
      R1 /= (det1 ** (1/3))
      R2 /= (det2 ** (1/3))
    except np.linalg.LinAlgError:
       print("Warning: Singular matrix encountered during normalization in angle calculation.")
       return 180.0 # Indicate large rotation on error

    # Check if matrices are valid rotation matrices (optional but good practice)
    # if not np.allclose(np.linalg.det(R1), 1.0) or not np.allclose(np.linalg.det(R2), 1.0):
    #     print("Warning: Non-rotation matrix detected in angle calculation.")
        # Decide how to handle: return 0, raise error, etc.

    try:
        R_rel = R2 @ R1.T
        # Angle calculation from trace: angle = arccos((trace(R_rel) - 1) / 2)
        trace = np.trace(R_rel)
        # Clamp the value to [-1, 1] due to potential floating point inaccuracies
        cos_angle = np.clip((trace - 1.0) / 2.0, -1.0, 1.0)
        angle_rad = np.arccos(cos_angle)
        return np.degrees(angle_rad)
    except Exception as e:
        print(f"Error calculating rotation angle: {e}")
        # Return a large angle to indicate a problem or potential jump
        return 180.0 # Or some other indicator


def segment_sequences_by_jumps(frames_data, rotation_threshold_degrees=15.0, displacement_threshold=3.0, min_seq_len=2):
    """
    Segments frames into sequences, breaking whenever rotation or displacement
    between consecutive frames exceeds thresholds.
    """
    if not frames_data or len(frames_data) < 2:
        return []

    # Sort frames by index
    frames_data.sort(key=lambda f: f['frame_index'])

    potential_sequences = []
    current_sequence = []
    last_matrix = None
    last_pos = None

    for i, frame_data in enumerate(frames_data):
        frame_idx = frame_data['frame_index']
        matrix = np.array(frame_data['rot_mat'])
        pos = matrix[:3, 3]

        break_sequence = False
        if i > 0 and last_matrix is not None and last_pos is not None:
            # Calculate displacement
            displacement = np.linalg.norm(pos - last_pos)
            if displacement > displacement_threshold:
                # print(f"  Debug: Displacement break at frame {frame_idx} (Disp: {displacement:.2f} > {displacement_threshold})") # Optional debug
                break_sequence = True

            # Calculate rotation change only if not already broken by displacement
            if not break_sequence:
                rotation_change_deg = calculate_rotation_angle_degrees(last_matrix, matrix)
                if rotation_change_deg > rotation_threshold_degrees:
                    # print(f"  Debug: Rotation break at frame {frame_idx} (Rot: {rotation_change_deg:.2f} > {rotation_threshold_degrees})") # Optional debug
                    break_sequence = True

        # --- Sequence Management ---
        if break_sequence:
            # End the previous sequence if it's long enough
            if len(current_sequence) >= min_seq_len:
                potential_sequences.append(current_sequence)
            # Start a new sequence with the current frame
            current_sequence = [frame_idx]
        else:
            # Add the current frame to the sequence (handle first frame case implicitly)
            current_sequence.append(frame_idx)


        # Update state for the next iteration
        last_matrix = matrix
        last_pos = pos

    # Add the last sequence if it's valid
    if len(current_sequence) >= min_seq_len:
        potential_sequences.append(current_sequence)

    return potential_sequences

## MODIFIED ## Renamed function and added backward detection logic
def filter_sequences_by_motion_direction(sequences, frame_map,
                                         forward_vs_side_threshold=2.0,
                                         min_overall_forward_disp=0.2,
                                         min_overall_backward_disp=0.2): # ## NEW ## Parameter
    """
    Filters sequences, keeping those showing significant overall forward OR backward motion.
    Returns two lists: forward_sequences and backward_sequences.
    """
    forward_sequences = []
    backward_sequences = [] # ## NEW ## List for backward sequences
    if not sequences:
        return [], []

    for seq in sequences:
        if len(seq) < 2:
            continue # Need at least two frames to determine motion

        start_frame_idx = seq[0]
        end_frame_idx = seq[-1]

        if start_frame_idx not in frame_map or end_frame_idx not in frame_map:
            # print(f"Warning: Skipping sequence {seq} due to missing frame data in map.")
            continue

        try:
            start_pos, start_fwd, start_rgt = get_pose_info(frame_map[start_frame_idx])
            end_pos, _, _ = get_pose_info(frame_map[end_frame_idx])
        except ValueError as e:
            print(f"Warning: Skipping sequence {seq} due to pose info error: {e}")
            continue

        total_displacement = end_pos - start_pos
        total_disp_norm = np.linalg.norm(total_displacement)

        if total_disp_norm < 1e-5:
            # print(f"  Debug: Skipping sequence {seq} - negligible overall displacement.")
            continue # Skip sequences with almost zero overall movement

        # Project overall displacement onto the *starting* orientation of the sequence
        forward_component = np.dot(total_displacement, start_fwd)
        sideways_component = np.dot(total_displacement, start_rgt)

        # Check dominance of forward/backward motion over sideways motion
        is_dominant_fwd_bwd = False
        if abs(sideways_component) < 1e-5: # Negligible sideways motion
             is_dominant_fwd_bwd = True
        elif abs(forward_component) / (abs(sideways_component) + 1e-9) > forward_vs_side_threshold: # Ratio check
             is_dominant_fwd_bwd = True

        # ## MODIFIED ## Check for forward OR backward motion
        if is_dominant_fwd_bwd:
            if forward_component >= min_overall_forward_disp: # Check forward threshold
                forward_sequences.append(seq)
                # print(f"  Debug: Keeping FORWARD sequence {seq}. Fwd: {forward_component:.2f}, Side: {sideways_component:.2f}")
            elif forward_component <= -min_overall_backward_disp: # ## NEW ## Check backward threshold (negative direction)
                backward_sequences.append(seq)
                # print(f"  Debug: Keeping BACKWARD sequence {seq}. Fwd: {forward_component:.2f}, Side: {sideways_component:.2f}")
            # else:
                # print(f"  Debug: Filtering out sequence {seq} (dominant but not enough fwd/bwd). Fwd: {forward_component:.2f}, Side: {sideways_component:.2f}")
        # else:
            # print(f"  Debug: Filtering out sequence {seq} (sideways dominant). Fwd: {forward_component:.2f}, Side: {sideways_component:.2f}")

    return forward_sequences, backward_sequences


# --- New Helper for Subsampling (Keep as is) ---
def get_subsampled_indices(sequence_length, target_frames):
    """
    Calculates indices for temporal subsampling.
    Selects target_frames indices evenly spaced throughout the sequence.
    """
    if sequence_length <= 0:
        return []
    # If sequence is shorter or equal to target, use all frames
    if sequence_length <= target_frames:
        return np.arange(sequence_length)
    else:
        # Generate target_frames indices evenly spaced from 0 to sequence_length - 1
        indices_float = np.linspace(0, sequence_length - 1, target_frames)
        indices_int = sorted(list(set(np.round(indices_float).astype(int))))
        return np.array(indices_int)

# --- Main Logic (Modified) ---

## MODIFIED ## Added min_overall_backward_disp parameter
def create_sequence_gifs(json_file_path, dataset_dir, output_dir,
                         rotation_threshold_degrees=15.0,
                         displacement_threshold=3.0,
                         forward_vs_side_threshold=2.0,
                         min_overall_forward_disp=0.2,
                         min_overall_backward_disp=0.2, # ## NEW ## Parameter
                         num_sequences_to_save=5,
                         target_gif_frames=20,
                         gif_fps=10,
                         min_seq_len_initial=2):
    """
    Loads poses, segments sequences by jumps, filters for forward OR backward motion,
    saves all final sequence indices (reversing backward ones), selects top N longest,
    subsamples frames (reversing order for backward sequences), and saves them as GIFs.
    """
    # 1. Load JSON data (Robust loading from previous version)
    print(f"Loading JSON data from: {json_file_path}")
    try:
        with open(json_file_path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: JSON file not found at {json_file_path}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Could not decode JSON from {json_file_path}. Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred loading JSON: {e}")
        sys.exit(1)

    frames_data = data.get('frames', [])
    if not frames_data:
        print("Error: No 'frames' data found in the JSON file.")
        return

    # Create frame map and filter for valid frames (Robust handling from previous version)
    frame_map = {}
    valid_frames_data = []
    for i, frame in enumerate(frames_data):
        idx = frame.get('frame_index')
        if idx is not None:
            # Ensure rot_mat exists and is convertible to numpy array
            if 'rot_mat' in frame and isinstance(frame['rot_mat'], list):
                 try:
                      # Pre-validate matrix structure slightly
                      mat = np.array(frame['rot_mat'])
                      if mat.ndim != 2 or mat.shape[0] < 3 or mat.shape[1] < 4:
                           print(f"Warning: Invalid 'rot_mat' structure for frame_index {idx}. Skipping frame.")
                           continue
                 except Exception as e:
                      print(f"Warning: Cannot process 'rot_mat' for frame_index {idx}: {e}. Skipping frame.")
                      continue

                 if idx in frame_map:
                     print(f"Warning: Duplicate frame_index {idx} found. Using the last occurrence.")
                 frame_map[idx] = frame
                 valid_frames_data.append(frame)
            else:
                 print(f"Warning: Frame index {idx} missing 'rot_mat' or it's not a list. Skipping frame.")
        else:
             print(f"Warning: Frame at list index {i} is missing 'frame_index'. It will be ignored.")

    print(f"Loaded {len(frames_data)} frame entries, using {len(valid_frames_data)} with valid 'frame_index' and 'rot_mat'.")
    if not valid_frames_data:
         print("Error: No frames with valid 'frame_index' and 'rot_mat' found. Cannot proceed.")
         return

    # --- STAGE 1: Segment by Jumps ---
    print("\n--- Stage 1: Segmenting sequences based on jumps ---")
    print(f"Using Rotation Threshold: {rotation_threshold_degrees} degrees")
    print(f"Using Displacement Threshold: {displacement_threshold} units")
    print(f"Minimum sequence length after segmentation: {min_seq_len_initial}")
    potential_sequences = segment_sequences_by_jumps(
        valid_frames_data,
        rotation_threshold_degrees,
        displacement_threshold,
        min_seq_len_initial
    )
    print(f"Found {len(potential_sequences)} potential sequences based on jump segmentation.")

    if not potential_sequences:
        print("No potential sequences found after jump segmentation. Exiting.")
        return

    # --- STAGE 2: Filter for Forward & Backward Motion --- ## MODIFIED ##
    print("\n--- Stage 2: Filtering sequences for overall forward OR backward motion ---")
    print(f"Using Forward/Side Ratio Threshold: {forward_vs_side_threshold}")
    print(f"Using Minimum Overall Forward Displacement: {min_overall_forward_disp}")
    print(f"Using Minimum Overall Backward Displacement: {min_overall_backward_disp}") # ## NEW ## Log message
    ## MODIFIED ## Call new function and get two lists
    forward_sequences, backward_sequences = filter_sequences_by_motion_direction(
        potential_sequences,
        frame_map,
        forward_vs_side_threshold,
        min_overall_forward_disp,
        min_overall_backward_disp # ## NEW ## Pass parameter
    )
    print(f"Filtered down to {len(forward_sequences)} forward sequences and {len(backward_sequences)} backward sequences.")

    # 3. Create output directory
    try:
        os.makedirs(output_dir, exist_ok=True)
        print(f"\nEnsuring output directory exists: {output_dir}")
    except OSError as e:
        print(f"Error creating output directory '{output_dir}': {e}")
        sys.exit(1) # Exit if cannot create output dir

    ## --- NEW: Combine, Tag, and Sort Sequences --- ##
    tagged_sequences = []
    for seq in forward_sequences:
        tagged_sequences.append({'direction': 'forward', 'indices': seq})
    for seq in backward_sequences:
        tagged_sequences.append({'direction': 'backward', 'indices': seq})

    # Sort by original length (longest first)
    tagged_sequences.sort(key=lambda item: len(item['indices']), reverse=True)
    total_final_sequences = len(tagged_sequences)
    print(f"Total {total_final_sequences} final sequences (forward & backward) found.")

    # 5. Save ALL FINAL sequence indices to JSON (Reverse backward ones here)
    if tagged_sequences:
        base_input_name = os.path.splitext(os.path.basename(json_file_path))[0]
        # Update filename to reflect content
        output_sequences_filename = f"{base_input_name}_final_fwd_bwd_sequences_filtered.json" # ## MODIFIED ## Filename
        output_sequences_path = os.path.join(output_dir, output_sequences_filename)

        # Prepare list for JSON: reverse backward sequences
        sequences_to_save_json = []
        for item in tagged_sequences:
            seq_indices = item['indices']
            if item['direction'] == 'backward':
                sequences_to_save_json.append(list(reversed(seq_indices))) # Reverse backward ones
            else:
                sequences_to_save_json.append(seq_indices) # Keep forward ones as is

        print(f"\nSaving all {total_final_sequences} final sequence indices (backward reversed, sorted by length) to: {output_sequences_path}")
        try:
            with open(output_sequences_path, 'w') as f:
                # Save the list containing original forward and reversed backward sequences
                json.dump(sequences_to_save_json, f, indent=4)
            print("Successfully saved final sequence indices.")
        except IOError as e:
            print(f"  ERROR: Could not write sequence indices to {output_sequences_path}: {e}")
        except Exception as e:
             print(f"  ERROR: An unexpected error occurred while saving sequence indices: {e}")
    else:
        print("\nNo forward or backward sequences remained after filtering. Skipping sequence index file generation.")
        print("\nScript finished - no final sequences found.")
        return

    # 6. Select top N sequences to process for GIFs
    if num_sequences_to_save <= 0:
         print("\nnum_sequences_to_save is 0. Skipping GIF generation.")
         print("\nScript finished.")
         return

    # Select from the tagged and sorted list
    sequences_to_process_tagged = tagged_sequences[:num_sequences_to_save]
    print(f"\nSelected the top {len(sequences_to_process_tagged)} longest final sequences for GIF generation.")
    if not sequences_to_process_tagged:
        print("No sequences selected for GIF generation.")
        print("\nScript finished.")
        return

    # Print details of selected sequences
    print("Selected sequences (original frame indices, tagged):")
    print_limit = 10
    for i, item in enumerate(sequences_to_process_tagged):
        seq = item['indices']
        direction = item['direction']
        if i < print_limit:
             seq_str = f"{seq[:5]}...{seq[-5:]}" if len(seq) > 10 else str(seq)
             print(f"  Rank {i+1} ({direction}, orig len {len(seq)}): {seq_str}") # ## MODIFIED ## Show direction
        elif i == print_limit:
            print(f"  ... (omitting details of remaining {len(sequences_to_process_tagged) - print_limit} sequences)")
            break

    # 7. Process each selected sequence to create a GIF
    print(f"\nGenerating GIFs with target frame count ~{target_gif_frames}...")
    gifs_created_count = 0
    # ## MODIFIED ## Iterate through tagged sequences
    for i, item in enumerate(sequences_to_process_tagged):
        seq_rank = i + 1 # Use actual rank (1-based) after final sorting
        direction = item['direction']
        original_sequence_indices = item['indices']
        original_seq_len = len(original_sequence_indices)
        print(f"\n--- Processing Sequence Rank {seq_rank}/{len(sequences_to_process_tagged)} ({direction.upper()}, Original Length: {original_seq_len}) ---") # ## MODIFIED ##

        # --- ## NEW: Reverse sequence if backward BEFORE subsampling ---
        if direction == 'backward':
            processing_sequence_indices = list(reversed(original_sequence_indices))
            print(f"  Reversed sequence for backward motion processing.")
        else:
            processing_sequence_indices = original_sequence_indices # Use original order for forward

        # --- Subsampling Step (Use potentially reversed sequence) ---
        # Note: Subsampling uses the length of the (potentially reversed) sequence, which is the same as original.
        # The indices returned by get_subsampled_indices are relative to the processing_sequence_indices list.
        subsampled_relative_indices = get_subsampled_indices(len(processing_sequence_indices), target_gif_frames)
        if len(subsampled_relative_indices) == 0:
             print(f"  Skipping sequence rank {seq_rank}: No frames selected after subsampling (original length was {original_seq_len}).")
             continue

        # Get the actual frame indices from the potentially reversed list
        frames_to_load_indices = [processing_sequence_indices[idx] for idx in subsampled_relative_indices]
        num_frames_for_gif = len(frames_to_load_indices)
        print(f"  Subsampling resulted in {num_frames_for_gif} frames for the GIF.")

        # --- Load Images for Subsampled Frames ---
        sequence_images = []
        print(f"  Loading {num_frames_for_gif} images (order: {'reversed' if direction == 'backward' else 'original'})...") # ## MODIFIED ##
        loaded_image_count = 0
        for frame_index in frames_to_load_indices: # These indices are now in the correct order for the GIF
            if frame_index not in frame_map:
                print(f"  Warning: Frame index {frame_index} (from subsampled sequence) not found in JSON data mapping. Skipping.")
                continue

            frame_data = frame_map[frame_index]
            relative_image_path = frame_data.get('file_path')
            image_source_info = ""
            if relative_image_path:
                 full_image_path = os.path.join(dataset_dir, relative_image_path)
                 image_source_info = f"from 'file_path': {relative_image_path}"
            else:
                # Fallback image path construction (adjust if needed)
                # Use a robust formatting like f'{frame_index:06d}.png' or similar if needed
                image_filename = f"{frame_index:04d}.png" # Example, adjust if needed
                full_image_path = os.path.join(dataset_dir, image_filename)
                image_source_info = f"using fallback filename pattern" # More general
                print(f"  Warning: 'file_path' not found for frame {frame_index}. Trying path '{full_image_path}'.")


            try:
                img = Image.open(full_image_path).convert('RGB')
                sequence_images.append(np.array(img))
                loaded_image_count += 1
            except FileNotFoundError:
                print(f"  ERROR: Image file not found at '{full_image_path}' ({image_source_info}). Skipping frame {frame_index}.")
            except UnidentifiedImageError:
                 print(f"  ERROR: Could not identify image file at '{full_image_path}' ({image_source_info}). Skipping frame {frame_index}.")
            except Exception as e:
                print(f"  ERROR: Failed loading image {full_image_path} ({image_source_info}): {e}. Skipping frame {frame_index}.")

        print(f"  Successfully loaded {loaded_image_count} out of {num_frames_for_gif} requested images.")

        # --- Generate GIF ---
        actual_gif_frames = len(sequence_images)
        if actual_gif_frames >= 2:
            # ## MODIFIED ## Adjust filename based on direction
            output_gif_filename = f"{direction}_seq_rank_{seq_rank}_origlen_{original_seq_len}_frames_{actual_gif_frames}.gif"
            output_gif_path = os.path.join(output_dir, output_gif_filename)
            try:
                print(f"  Saving GIF with {actual_gif_frames} frames to: {output_gif_path}")
                imageio.mimsave(output_gif_path, sequence_images, fps=gif_fps)
                print(f"  Successfully saved GIF.")
                gifs_created_count += 1
            except Exception as e:
                print(f"  ERROR: Failed to save GIF {output_gif_path}: {e}")
        elif actual_gif_frames == 1 :
             print(f"  Skipping GIF generation for sequence rank {seq_rank}: Only 1 valid image loaded.")
        else:
             print(f"  Skipping GIF generation for sequence rank {seq_rank}: No valid images were loaded.")

    print(f"\nFinished processing. Generated {gifs_created_count} GIF files.")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate GIFs from image sequences based on motion analysis.")

    # --- Configuration Arguments ---
    parser.add_argument("--input_json_path", type=str, default='/wekafs/ict/junyiouy/matrixcity_hf/small_city_road_outside_dense_transforms.json',
                        help="Path to the input JSON file containing image transforms.")
    parser.add_argument("--dataset_dir", type=str, default='/wekafs/ict/junyiouy/matrixcity_hf/small_city_road_outside_dense',
                        help="Directory containing images/subdirs as referenced in the JSON file.")
    parser.add_argument("--output_gif_dir", type=str, default='./fwd_bwd_sequence_gifs_filtered_subsampled_bf',
                        help="Directory to save the generated GIFs and sequence list.")

    # --- Sequence Detection Parameters ---
    parser.add_argument("--rotation_threshold_degrees", type=float, default=5.0,
                        help="Threshold for rotation (in degrees) to consider images part of the same sequence.")
    parser.add_argument("--displacement_threshold", type=float, default=3.0,
                        help="Threshold for displacement to consider images part of the same sequence.")
    parser.add_argument("--min_seq_len_initial", type=int, default=3,
                        help="Minimum length of a sequence before filtering.")


    # --- Forward/Backward Motion Filtering Parameters ---
    parser.add_argument("--forward_vs_side_threshold", type=float, default=2.0,
                        help="Threshold to distinguish forward motion from side motion.")
    parser.add_argument("--min_overall_forward_disp", type=float, default=0.01,
                        help="Minimum overall forward displacement for a sequence to be considered forward moving.")
    parser.add_argument("--min_overall_backward_disp", type=float, default=0.01,
                        help="Minimum overall backward displacement for a sequence to be considered backward moving.")

    # --- GIF Generation Parameters ---
    parser.add_argument("--num_sequences_to_save", type=int, default=50,
                        help="Number of sequences to save as GIFs.")
    parser.add_argument("--target_gif_frames", type=int, default=25,
                        help="Target number of frames in each generated GIF.")
    parser.add_argument("--gif_fps", type=int, default=8,
                        help="Frames per second for the generated GIFs.")

    args = parser.parse_args()


    create_sequence_gifs(
        json_file_path=args.input_json_path,
        dataset_dir=args.dataset_dir,
        output_dir=args.output_gif_dir,
        rotation_threshold_degrees=args.rotation_threshold_degrees,
        displacement_threshold=args.displacement_threshold,
        forward_vs_side_threshold=args.forward_vs_side_threshold,
        min_overall_forward_disp=args.min_overall_forward_disp,
        min_overall_backward_disp=args.min_overall_backward_disp,
        num_sequences_to_save=args.num_sequences_to_save,
        target_gif_frames=args.target_gif_frames,
        gif_fps=args.gif_fps,
        min_seq_len_initial=args.min_seq_len_initial
    )

    print("\nScript finished.")