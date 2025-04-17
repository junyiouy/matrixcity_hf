# -*- coding: utf-8 -*-
import os
import glob
import subprocess
import tempfile
from pathlib import Path
import torch
import multiprocessing as mp # 导入多进程库
from transformers import Qwen2_5OmniModel, Qwen2_5OmniProcessor
# Assuming qwen_omni_utils.py is in the same directory or accessible
try:
    from qwen_omni_utils import process_mm_info
except ImportError:
    print("Error: Could not import 'process_mm_info' from 'qwen_omni_utils'. Make sure the file exists and is importable.")
    # You might need to define a placeholder or handle this case if the file is missing
    def process_mm_info(*args, **kwargs):
        print("Warning: 'process_mm_info' not found. Returning empty multimedia info.")
        return None, None, None # Placeholder return

import re
import time
import json
import queue # 用于处理队列超时
import argparse # For command-line arguments

# --- Constants and Initial Setup (Keep as before) ---
print(f"PyTorch 版本: {torch.__version__}")
print(f"CUDA 是否可用: {torch.cuda.is_available()}")
num_gpus = 0
if torch.cuda.is_available():
    num_gpus = torch.cuda.device_count()
    print(f"CUDA 版本: {torch.version.cuda}")
    print(f"GPU 数量: {num_gpus}")
    for i in range(num_gpus):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    bf16_supported = torch.cuda.is_bf16_supported()
    print(f"BF16 支持: {bf16_supported}")
else:
    bf16_supported = False
    print("警告: 未检测到 CUDA GPU。脚本将无法运行。")
    # exit() # Exit if no GPU

DEFAULT_OUTPUT_DIR = Path("./matrixcity_qwen_omni_captions")
DEFAULT_SUB_SEQUENCE_LENGTH = 100
DEFAULT_FRAME_SAMPLING_INTERVAL = 2 # Sample every 2nd frame for video creation
DEFAULT_FRAME_RATE = 10 # FPS for the temporary video
DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-Omni-7B"
DEFAULT_MAX_NEW_TOKENS = 150
DEFAULT_GPU_BATCH_SIZE = 4 # Adjust based on GPU memory
DEFAULT_TARGET_DTYPE = torch.bfloat16 if bf16_supported else torch.float16
DEFAULT_NUM_VIDEO_CREATOR_WORKERS = max(1, mp.cpu_count() // 4)
DEFAULT_QUEUE_MAX_SIZE_FACTOR = 4
DEFAULT_GPU_WORKER_BATCH_TIMEOUT = 1.0
FINAL_AGGREGATED_JSON_NAME = "matrixcity_aggregated_captions.json"
INTERMEDIATE_JSON_SUFFIX = "_subcaption.json"
# --- Minimum frames needed for 'normal' video (can be adjusted) ---
MIN_FRAMES_FOR_VIDEO = 2

# --- Argument Parser (Keep as before) ---
def parse_args():
    parser = argparse.ArgumentParser(description="Annotate MatrixCity sequences using Qwen-Omni.")
    parser.add_argument("--sequence_json_path", type=str, required=True, help="Path to the input MatrixCity sequence JSON file (e.g., small_city1.json).")
    parser.add_argument("--output_dir", type=str, default=str(DEFAULT_OUTPUT_DIR), help="Directory to save intermediate and final JSON outputs.")
    parser.add_argument("--sub_sequence_length", type=int, default=DEFAULT_SUB_SEQUENCE_LENGTH, help="Length of sub-sequences to split the long sequences into.")
    parser.add_argument("--frame_sampling_interval", type=int, default=DEFAULT_FRAME_SAMPLING_INTERVAL, help="Interval for sampling frames within a sub-sequence to create the video (e.g., 2 means every 2nd frame).")
    parser.add_argument("--frame_rate", type=int, default=DEFAULT_FRAME_RATE, help="Frame rate for the temporary videos created for captioning.")
    parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL_NAME, help="Name or path of the Qwen-Omni model.")
    parser.add_argument("--max_new_tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS, help="Maximum number of new tokens to generate for captions.")
    parser.add_argument("--gpu_batch_size", type=int, default=DEFAULT_GPU_BATCH_SIZE, help="Batch size for inference on each GPU.")
    parser.add_argument("--num_video_creator_workers", type=int, default=DEFAULT_NUM_VIDEO_CREATOR_WORKERS, help="Number of CPU workers for creating temporary videos.")
    parser.add_argument("--gpu_worker_batch_timeout", type=float, default=DEFAULT_GPU_WORKER_BATCH_TIMEOUT, help="Max time (seconds) a GPU worker waits to fill a batch.")
    parser.add_argument("--force_regenerate", action='store_true', help="Force regeneration of captions even if intermediate JSON files exist.")
    parser.add_argument("--skip_aggregation", action='store_true', help="Skip the final aggregation step (only generate intermediate files).")
    parser.add_argument("--keep_intermediate", action='store_true', help="Keep the intermediate sub-sequence JSON files after aggregation.")
    # New optional arg to limit sequence processing for debugging
    parser.add_argument("--limit_sequences", type=int, default=None, help="Process only the first N sequences (for debugging).")


    args = parser.parse_args()

    # Convert path strings to Path objects
    args.output_dir = Path(args.output_dir)
    args.sequence_json_path = Path(args.sequence_json_path)
    args.target_dtype = DEFAULT_TARGET_DTYPE # Set based on detection

    if not args.sequence_json_path.is_file():
        raise FileNotFoundError(f"Input sequence JSON not found: {args.sequence_json_path}")

    print("--- Configuration ---")
    for k, v in vars(args).items():
        print(f"{k}: {v}")
    print("--------------------")
    return args

# --- Video Creation Function (Keep as before) ---
def create_video_from_image_list(image_paths: list[Path], output_path: Path, fps: int) -> bool:
    """Creates a video from a specific list of image paths (Path objects)."""
    if not image_paths:
        return False

    list_path = None
    absolute_output_path = output_path.resolve().as_posix()
    success = False
    try:
        fd, list_path = tempfile.mkstemp(suffix='.txt', text=True)
        with os.fdopen(fd, 'w') as file_list:
            valid_paths_written = 0
            for img_path in image_paths:
                if not isinstance(img_path, Path):
                     print(f"[Video Creation Warning] Expected Path object, got {type(img_path)}. Skipping.")
                     continue
                if not img_path.is_file():
                     # Print less verbosely for missing files during video creation, as main handles overall check
                     # print(f"[Video Creation Warning] Image file not found, skipping: {img_path}")
                     continue
                file_list.write(f"file '{img_path.resolve().as_posix()}'\n")
                file_list.write(f"duration {1/fps}\n")
                valid_paths_written += 1

        # Check based on valid paths written to the list
        if valid_paths_written == 0:
             # print(f"[Video Creation Error] No valid image files found for {output_path.name}. Cannot create video.")
             success = False
        else:
            # Only run ffmpeg if we have valid paths
            command = [
                'ffmpeg', '-hide_banner', '-loglevel', 'error',
                '-f', 'concat', '-safe', '0', '-i', list_path,
                '-r', str(fps), '-pix_fmt', 'yuv420p', '-y', absolute_output_path
            ]
            result = subprocess.run(command, check=True, capture_output=True, text=True, encoding='utf-8')
            success = True # Assume success if ffmpeg command runs without error

    except FileNotFoundError:
        print("[Video Creation Error] ffmpeg command not found. Ensure it's installed and in PATH.")
        success = False
    except subprocess.CalledProcessError as e:
        stderr_decoded = e.stderr.decode('utf-8', errors='replace') if isinstance(e.stderr, bytes) else e.stderr
        # Reduce verbosity of ffmpeg errors unless debugging
        # print(f"[Video Creation Error] {output_path.name} : {stderr_decoded}")
        success = False
    except Exception as e:
        print(f"[Video Creation Error] {output_path.name} unexpected error: {e}")
        success = False
    finally:
        if list_path and os.path.exists(list_path):
            try: os.remove(list_path)
            except OSError as e_rem: print(f"Warning: Failed to remove temp list file {list_path}: {e_rem}")
    return success


# --- Result Parsing Function (Keep as before) ---
def parse_and_clean_caption_output(raw_text: str, max_tokens: int) -> dict:
    """Parses model output, extracts 'assistant' reply, and cleans."""
    output_data = {}
    assistant_marker = "assistant\n"
    marker_index = raw_text.find(assistant_marker)

    if marker_index != -1:
        caption_text = raw_text[marker_index + len(assistant_marker):].strip()
    else:
        # Less alarming warning, as sometimes the model might omit the marker
        # print("    Info: 'assistant\\n' marker not found. Cleaning raw output.")
        caption_text = raw_text.strip()

    # Basic cleaning
    if caption_text.lower().startswith("the video showcases "):
        caption_text = caption_text[len("the video showcases "):].strip()
    if caption_text.lower().startswith("this video shows "):
        caption_text = caption_text[len("this video shows "):].strip()
    if caption_text.lower().startswith("caption:"):
        caption_text = caption_text[len("caption:"):].strip()

    caption_text = caption_text.replace("\n", " ").strip() # Remove newlines

    if caption_text:
        output_data["caption"] = caption_text
    else:
        output_data["raw_output"] = raw_text.strip() # Store raw if cleaning failed
        print(f"    Warning: Caption is empty after cleaning. Raw: '{raw_text[:100]}...'")

    return output_data

# --- Video Creator Worker (Keep as before) ---
def video_creator_worker(task_queue: mp.Queue, ready_queue: mp.Queue, temp_dir: Path, frame_rate: int):
    """
    Gets tasks (sub_seq_id, sampled_paths[Path], original_paths[Path]), creates temp video,
    puts (sub_seq_id, temp_video_path, original_paths[Path]) onto ready_queue.
    """
    pid = os.getpid()
    # print(f"[Video Creator {pid}] Started.") # Less verbose start
    while True:
        try:
            task = task_queue.get()
            if task is None: # Poison pill
                # print(f"[Video Creator {pid}] Received exit signal.")
                break

            sub_seq_id, sampled_image_paths, original_paths_in_sub = task
            temp_video_path = temp_dir / f"temp_{sub_seq_id}_{pid}_{int(time.time()*1000)}.mp4"

            success = create_video_from_image_list(
                sampled_image_paths, temp_video_path, frame_rate
            )

            if success and temp_video_path.exists() and temp_video_path.stat().st_size > 0:
                ready_queue.put((sub_seq_id, temp_video_path, original_paths_in_sub))
            else:
                 # Video creation failed (might be due to missing source frames, ffmpeg error etc.)
                 # Don't print error here, let the main aggregation report missing files later.
                 if temp_video_path.exists():
                     try: temp_video_path.unlink()
                     except OSError: pass # Ignore error during cleanup

        except Exception as e:
            # Avoid crashing the worker process if possible
            print(f"[Video Creator {pid}] Error: {e}")
            # import traceback # Enable for deep debugging
            # traceback.print_exc()

    # print(f"[Video Creator {pid}] Exiting.") # Less verbose exit


# --- GPU Inference Worker (Keep as before) ---
def gpu_inference_worker(gpu_id: int, ready_queue: mp.Queue, output_dir: Path, batch_size: int, timeout: float, model_name: str, max_new_tokens: int, target_dtype: torch.dtype):
    """
    Runs inference on gpu_id. Gets (sub_seq_id, temp_path, original_paths[Path]), batches, infers,
    saves intermediate JSON, deletes temp video.
    """
    pid = os.getpid()
    device = f"cuda:{gpu_id}"
    print(f"[GPU Worker {pid} - GPU {gpu_id}] Starting on {device}")

    try:
        torch.cuda.set_device(device)

        print(f"[GPU Worker {pid} - GPU {gpu_id}] Loading model {model_name}...")
        major, _ = torch.cuda.get_device_capability(gpu_id)
        attn_impl = "flash_attention_2" if major >= 8 else "eager"
        print(f"[GPU Worker {pid} - GPU {gpu_id}] Using attention implementation: {attn_impl}")

        model = Qwen2_5OmniModel.from_pretrained(
            model_name,
            torch_dtype=target_dtype,
            device_map=device, # Assign model directly to this GPU
            enable_audio_output=False,
            attn_implementation=attn_impl
        ).eval() # Set model to evaluation mode
        processor = Qwen2_5OmniProcessor.from_pretrained(model_name)
        print(f"[GPU Worker {pid} - GPU {gpu_id}] Model loaded.")

        batch_sub_seq_ids = []
        batch_temp_video_paths = []
        batch_original_paths_lists = [] # This will now hold lists of Path objects
        batch_conversations = []

        while True:
            try:
                # Non-blocking get initially to check for poison pill first
                try:
                    item = ready_queue.get(block=False)
                except queue.Empty:
                    # If empty, try blocking get with timeout
                    if not batch_sub_seq_ids: # Only block if no current batch
                        try:
                            item = ready_queue.get(timeout=timeout)
                        except queue.Empty:
                             # Timeout occurred, process current batch if any
                            if batch_sub_seq_ids:
                                process_gpu_batch(model, processor, batch_sub_seq_ids, batch_temp_video_paths,
                                                  batch_original_paths_lists, batch_conversations, output_dir, device, max_new_tokens)
                                batch_sub_seq_ids, batch_temp_video_paths, batch_original_paths_lists, batch_conversations = [], [], [], []
                            continue # Continue waiting
                    else:
                        # Have a partial batch, process it before potentially waiting again
                        process_gpu_batch(model, processor, batch_sub_seq_ids, batch_temp_video_paths,
                                          batch_original_paths_lists, batch_conversations, output_dir, device, max_new_tokens)
                        batch_sub_seq_ids, batch_temp_video_paths, batch_original_paths_lists, batch_conversations = [], [], [], []
                        continue # Go back to non-blocking get


                if item is None: # Poison pill
                    print(f"[GPU Worker {pid} - GPU {gpu_id}] Received exit signal.")
                    if batch_sub_seq_ids:
                        print(f"[GPU Worker {pid} - GPU {gpu_id}] Processing final batch ({len(batch_sub_seq_ids)} items)...")
                        process_gpu_batch(model, processor, batch_sub_seq_ids, batch_temp_video_paths,
                                          batch_original_paths_lists, batch_conversations, output_dir, device, max_new_tokens)
                    break # Exit loop

                sub_seq_id, temp_video_path, original_paths_in_sub = item

                # --- Build conversation ---
                video_input_path_str = temp_video_path.resolve().as_posix()
                # More concise prompt?
                user_prompt_instruction = "Describe the visual scene in the video focusing on objects, environment, and actions. Be detailed but concise."
                system_prompt = f"You are a video analysis expert providing detailed scene descriptions, limited to about {max_new_tokens} tokens."

                conversation = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": [
                        {"type": "video", "video": video_input_path_str},
                        {"type": "text", "text": user_prompt_instruction }
                    ]},
                ]

                batch_sub_seq_ids.append(sub_seq_id)
                batch_temp_video_paths.append(temp_video_path)
                batch_original_paths_lists.append(original_paths_in_sub)
                batch_conversations.append(conversation)

                if len(batch_sub_seq_ids) >= batch_size:
                    process_gpu_batch(model, processor, batch_sub_seq_ids, batch_temp_video_paths,
                                      batch_original_paths_lists, batch_conversations, output_dir, device, max_new_tokens)
                    batch_sub_seq_ids, batch_temp_video_paths, batch_original_paths_lists, batch_conversations = [], [], [], []


            except Exception as e_item: # Catch broader exceptions during item handling
                print(f"[GPU Worker {pid} - GPU {gpu_id}] Error processing item loop: {e_item}")
                # Clear current partial batch on error to avoid reprocessing bad data?
                # batch_sub_seq_ids, batch_temp_video_paths, batch_original_paths_lists, batch_conversations = [], [], [], []
                # import traceback # Enable for debugging
                # traceback.print_exc()
                # Maybe sleep briefly to avoid tight error loops
                time.sleep(0.5)


    except Exception as e_load:
        print(f"[GPU Worker {pid} - GPU {gpu_id}] Fatal error during setup or main loop: {e_load}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up model and processor explicitly? (Might help release GPU memory)
        del model
        del processor
        torch.cuda.empty_cache()
        print(f"[GPU Worker {pid} - GPU {gpu_id}] Exiting.")


# --- GPU Batch Processing Function (Keep as before) ---
def process_gpu_batch(model, processor, batch_sub_seq_ids, batch_temp_video_paths, batch_original_paths_lists, batch_conversations, output_dir, device, max_new_tokens):
    """Processes a batch on a single GPU, saves intermediate JSONs."""
    batch_start_time = time.time()
    processed_in_batch = 0
    skipped_in_batch = 0
    batch_size = len(batch_sub_seq_ids)

    if not batch_sub_seq_ids:
        return 0, 0

    # print(f"  [GPU {device}] Processing batch (Size: {batch_size}) IDs: {batch_sub_seq_ids[0]}...{batch_sub_seq_ids[-1]}") # Verbose

    try:
        USE_AUDIO_IN_VIDEO = False
        text_prompts = processor.apply_chat_template(batch_conversations, add_generation_prompt=True, tokenize=False)

        audios, images, videos = process_mm_info(batch_conversations, use_audio_in_video=USE_AUDIO_IN_VIDEO)
        inputs = processor(text=text_prompts, audios=audios, images=images, videos=videos, return_tensors="pt", padding=True, use_audio_in_video=USE_AUDIO_IN_VIDEO)
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k,v in inputs.items()}

        generation_start_time = time.time()
        with torch.no_grad():
            gen_kwargs = {"max_new_tokens": max_new_tokens, "do_sample": False}
            # Ensure generate does not use cache if running into memory issues, though it might be slower
            # gen_kwargs["use_cache"] = False
            text_ids = model.generate(**inputs, use_audio_in_video=USE_AUDIO_IN_VIDEO, return_audio=False, **gen_kwargs)
        generation_time = time.time() - generation_start_time

        batch_generated_texts = processor.batch_decode(text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

        # --- Save Intermediate JSON Outputs ---
        for i, sub_seq_id in enumerate(batch_sub_seq_ids):
            generated_text = batch_generated_texts[i]
            original_paths_str = [str(p.resolve()) for p in batch_original_paths_lists[i]] # p IS a Path object

            output_data = parse_and_clean_caption_output(generated_text, max_new_tokens)
            caption = output_data.get("caption")

            if caption:
                intermediate_data = {
                    "caption": caption,
                    "original_image_paths": original_paths_str # Save the list of resolved path strings
                }
                output_json_path = output_dir / f"{sub_seq_id}{INTERMEDIATE_JSON_SUFFIX}"
                try:
                    with open(output_json_path, 'w', encoding='utf-8') as f:
                        json.dump(intermediate_data, f, ensure_ascii=False, indent=2) # Smaller indent
                    processed_in_batch += 1
                except Exception as e_save:
                    print(f"    [GPU {device}] Error saving JSON {output_json_path.name}: {e_save}")
                    skipped_in_batch += 1
            else:
                 # Log only if raw output exists, otherwise it was likely an empty generation
                 raw = output_data.get('raw_output', '')
                 if raw: print(f"    [GPU {device}] Captioning failed or empty for {sub_seq_id}. Raw: '{raw[:50]}...'")
                 skipped_in_batch += 1

    # Catch specific potential errors like CUDA OOM
    except torch.cuda.OutOfMemoryError:
        print(f"!!!!!!!! [GPU {device}] CUDA Out of Memory Error during batch processing !!!!!!!!")
        print(f"  Failed Batch Sub-sequence IDs: {batch_sub_seq_ids}")
        print(f"  Attempting to clear cache...")
        torch.cuda.empty_cache()
        skipped_in_batch += len(batch_sub_seq_ids) # Mark all as skipped
        processed_in_batch = 0
    except Exception as e_batch:
        print(f"!!!!!!!! [GPU {device}] Error processing batch !!!!!!!!")
        print(f"  Failed Batch Sub-sequence IDs: {batch_sub_seq_ids}")
        print(f"  Error: {e_batch}")
        # import traceback # Enable for debugging
        # traceback.print_exc()
        skipped_in_batch += len(batch_sub_seq_ids)
        processed_in_batch = 0

    finally:
        # --- Clean up temp videos for this batch ---
        for temp_video_path in batch_temp_video_paths:
            if temp_video_path is not None and temp_video_path.exists():
                try: temp_video_path.unlink()
                except OSError: pass # Ignore cleanup error

    batch_duration = time.time() - batch_start_time
    # print(f"  [GPU {device}] Batch finished. Time: {batch_duration:.2f}s. Processed: {processed_in_batch}, Skipped: {skipped_in_batch}") # Verbose
    return processed_in_batch, skipped_in_batch

# --- Modified Final Aggregation Function (Includes double check) ---
def aggregate_results(output_dir: Path, final_json_name: str, keep_intermediate: bool, all_input_paths_set: set):
    """Aggregates results and performs a double-check against input paths."""
    print("\n--- Starting Final Aggregation & Double Check ---")
    aggregated_captions = {}
    intermediate_files = list(output_dir.glob(f"*{INTERMEDIATE_JSON_SUFFIX}"))
    print(f"Found {len(intermediate_files)} intermediate JSON files to aggregate.")

    processed_files = 0
    skipped_files = 0
    output_paths_set = set() # Collect paths found in intermediate files

    for json_path in intermediate_files:
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            caption = data.get("caption")
            original_paths = data.get("original_image_paths") # These are absolute path strings

            if caption and original_paths and isinstance(original_paths, list):
                for img_path_str in original_paths:
                    aggregated_captions[img_path_str] = caption # Use the string path as key
                    output_paths_set.add(img_path_str) # Add to the set of processed paths
                processed_files += 1
            else:
                # print(f"Warning: Skipping invalid intermediate file {json_path.name} (missing caption or paths).") # Less verbose
                skipped_files += 1

        except json.JSONDecodeError:
            print(f"Error: Failed to decode JSON from {json_path.name}. Skipping.")
            skipped_files += 1
        except Exception as e:
            print(f"Error processing intermediate file {json_path.name}: {e}. Skipping.")
            skipped_files += 1

    final_output_path = output_dir / final_json_name
    print(f"\nAggregation Summary:")
    print(f"  Processed {processed_files} intermediate files (skipped {skipped_files}).")
    print(f"  Found {len(output_paths_set)} unique captioned image paths.")
    print(f"  Total unique image paths in input JSON: {len(all_input_paths_set)}")

    # --- Perform the double check ---
    missing_paths = all_input_paths_set - output_paths_set
    extra_paths = output_paths_set - all_input_paths_set # Should be empty

    print("\nDouble Check Results:")
    if not missing_paths and not extra_paths:
        print("  ✅ Success: All images from the input JSON have corresponding captions.")
    else:
        if missing_paths:
            print(f"  ❌ Warning: {len(missing_paths)} images from input JSON are MISSING captions!")
            # List first few missing paths for diagnosis
            missing_sample = list(missing_paths)[:10]
            print(f"     Examples: {missing_sample}")
            # Suggest possible reasons
            print("     Possible reasons: Invalid path in JSON, file not found on disk, error during video creation/processing for the sub-sequence.")
        if extra_paths:
            # This indicates a potential bug in the script's logic
            print(f"  ❓ Error: {len(extra_paths)} captioned images were found that were NOT in the original input JSON!")
            extra_sample = list(extra_paths)[:10]
            print(f"     Examples: {extra_sample}")

    # --- Save aggregated results ---
    if not aggregated_captions:
         print("\nError: No captions were aggregated. Final JSON will not be saved.")
         # Keep intermediate files for debugging if aggregation failed
         keep_intermediate = True
    else:
        print(f"\nSaving aggregated captions ({len(aggregated_captions)} entries) to: {final_output_path}")
        try:
            # Use indent=None for smallest file size, or indent=2 for readability
            with open(final_output_path, 'w', encoding='utf-8') as f:
                json.dump(aggregated_captions, f, ensure_ascii=False, indent=2)
            print("Aggregation saving complete.")
        except Exception as e:
            print(f"Error saving final aggregated JSON: {e}")
            keep_intermediate = True # Keep intermediate if saving failed

    # --- Clean up intermediate files ---
    if not keep_intermediate and processed_files > 0: # Only clean if requested AND aggregation was successful
        print("Cleaning up intermediate files...")
        deleted_count = 0
        failed_delete_count = 0
        # Make sure we only delete files we processed
        files_to_delete = [f for f in intermediate_files if f.name.endswith(INTERMEDIATE_JSON_SUFFIX)]
        for json_path in files_to_delete:
             try:
                 json_path.unlink()
                 deleted_count +=1
             except OSError as e:
                 print(f"Warning: Failed to delete intermediate file {json_path.name}: {e}")
                 failed_delete_count += 1
        print(f"Deleted {deleted_count} intermediate files (failed: {failed_delete_count}).")
    elif keep_intermediate:
         print("Keeping intermediate files as requested or due to errors.")


# --- Modified Main Function (Collects all input paths) ---
def main():
    args = parse_args()

    if num_gpus == 0:
        print("Error: No CUDA GPUs detected. Exiting.")
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    temp_dir_base = Path(tempfile.gettempdir()) / f"matrixcity_qwen_omni_{int(time.time())}"
    temp_dir_base.mkdir(exist_ok=True)
    print(f"Using temporary directory: {temp_dir_base}")

    print(f"Loading sequence data from: {args.sequence_json_path}")
    try:
        with open(args.sequence_json_path, 'r') as f:
            all_sequences_data = json.load(f) 
        print(f"Loaded {len(all_sequences_data)} long sequences.")
        # Apply sequence limit if specified
        if args.limit_sequences is not None and args.limit_sequences > 0:
             all_sequences_data = all_sequences_data[:args.limit_sequences]
             print(f"Limiting processing to the first {len(all_sequences_data)} sequences.")

    except Exception as e:
        print(f"Error loading or parsing sequence JSON: {e}")
        return

    # --- Collect ALL unique absolute input paths ---
    all_input_image_paths_absolute = set()
    print("Collecting all unique image paths from input JSON...")
    input_path_errors = 0
    for seq_idx, long_sequence in enumerate(all_sequences_data):
         if not isinstance(long_sequence, list):
             continue # Already warned during task prep
         for frame_info in long_sequence:
             if isinstance(frame_info, dict) and "image_path" in frame_info:
                 try:
                     abs_path_str = str(Path(frame_info["image_path"]).resolve())
                     all_input_image_paths_absolute.add(abs_path_str)
                 except Exception as e_path:
                     print(f"Warning: Error resolving input path '{frame_info.get('image_path', 'N/A')}' in seq {seq_idx}: {e_path}")
                     input_path_errors += 1
             else:
                 # This indicates format error in the input json itself
                 print(f"Warning: Invalid frame_info format in seq {seq_idx}: {frame_info}")
                 input_path_errors += 1
    print(f"Collected {len(all_input_image_paths_absolute)} unique absolute image paths from input.")
    if input_path_errors > 0:
        print(f"Encountered {input_path_errors} errors while resolving input paths.")
    # --- End Collection ---


    tasks_to_process = []
    total_sub_sequences = 0
    skipped_existing = 0
    processed_short = 0
    skipped_invalid = 0

    print(f"Splitting sequences into sub-sequences of length {args.sub_sequence_length}...")
    # This loop now focuses only on preparing tasks, not collecting all paths
    for seq_idx, long_sequence in enumerate(all_sequences_data):
        if not isinstance(long_sequence, list):
             # print(f"Warning: Item at index {seq_idx} is not a list, skipping.") # Less verbose
             skipped_invalid += 1
             continue

        num_frames_in_long_seq = len(long_sequence)
        for sub_seq_start_idx in range(0, num_frames_in_long_seq, args.sub_sequence_length):
            sub_seq_end_idx = min(sub_seq_start_idx + args.sub_sequence_length, num_frames_in_long_seq)
            sub_sequence = long_sequence[sub_seq_start_idx:sub_seq_end_idx]

            if not sub_sequence: continue

            original_paths_in_sub = []
            valid_sub_sequence = True
            for i, frame_info in enumerate(sub_sequence):
                if not isinstance(frame_info, dict) or "image_path" not in frame_info:
                    # print(f"Warning: Invalid frame data in sub-sequence {seq_idx}-{sub_seq_start_idx} at index {i}. Skipping.") # Less verbose
                    valid_sub_sequence = False
                    skipped_invalid += 1
                    break
                try:
                    abs_path = Path(frame_info["image_path"]).resolve()
                    # --- Check if file exists HERE ---
                    if not abs_path.is_file():
                         print(f"Warning: Input image file NOT FOUND: {abs_path} (in sub-seq {seq_idx}-{sub_seq_start_idx}). Sub-sequence might fail.")
                         # Decide whether to skip the whole sub-sequence or just let video creation fail later
                         # For now, let's still add it, aggregation check will catch missing paths.
                    original_paths_in_sub.append(abs_path) # Store Path object
                except Exception as e_path:
                     # print(f"Warning: Error processing path '{frame_info.get('image_path', 'N/A')}' in sub-sequence {seq_idx}-{sub_seq_start_idx}: {e_path}") # Less verbose
                     valid_sub_sequence = False
                     skipped_invalid += 1
                     break

            if not valid_sub_sequence: continue

            if not original_paths_in_sub: # Skip if somehow no valid paths were collected
                 skipped_invalid += 1
                 continue

            sampled_indices_in_sub = list(range(0, len(sub_sequence), args.frame_sampling_interval))
            num_sampled = len(sampled_indices_in_sub)
            sampled_image_paths = []

            if num_sampled < MIN_FRAMES_FOR_VIDEO:
                 # print(f"Info: Sub-sequence {seq_idx}-{sub_seq_start_idx} has {num_sampled} sampled frame(s). Processing first frame.") # Less verbose
                 sampled_image_paths = [original_paths_in_sub[0]] # Use Path of first frame
                 processed_short += 1 # Count this scenario
            else:
                 sampled_image_paths = [original_paths_in_sub[i] for i in sampled_indices_in_sub]


            sub_seq_id = f"seq{seq_idx}_sub{sub_seq_start_idx}"
            total_sub_sequences += 1

            intermediate_json_path = args.output_dir / f"{sub_seq_id}{INTERMEDIATE_JSON_SUFFIX}"
            if not args.force_regenerate and intermediate_json_path.exists():
                skipped_existing += 1
                continue

            # Add task only if sampled_image_paths is not empty
            if sampled_image_paths:
                 tasks_to_process.append((sub_seq_id, sampled_image_paths, original_paths_in_sub))
            else:
                 # This case implies original_paths_in_sub was empty or sampling failed unexpectedly
                 print(f"Warning: No valid sampled paths for sub-sequence {sub_seq_id}. Skipping task creation.")
                 skipped_invalid += 1



    print(f"\nTask Preparation Summary:")
    print(f"  Considered {total_sub_sequences} total sub-sequences.")
    print(f"  Prepared {len(tasks_to_process)} new tasks for processing.")
    print(f"  Skipped {skipped_existing} existing intermediate files.")
    print(f"  Will process {processed_short} sub-sequences using single-frame mode.")
    print(f"  Skipped {skipped_invalid} sub-sequences due to invalid format, path errors, or no valid frames.")

    if not tasks_to_process:
        print("\nNo new tasks to process.")
        if not args.skip_aggregation:
             if not all_input_image_paths_absolute:
                  print("Input path set is empty, cannot perform aggregation check.")
             elif not any(args.output_dir.glob(f"*{INTERMEDIATE_JSON_SUFFIX}")):
                  print("No intermediate files found for aggregation.")
             else:
                  # Perform aggregation even if no new tasks, using existing intermediate files
                  print("Attempting aggregation using existing intermediate files...")
                  aggregate_results(args.output_dir, FINAL_AGGREGATED_JSON_NAME, args.keep_intermediate, all_input_image_paths_absolute)
        else:
             print("Skipping aggregation as requested.")
        # Clean up empty temp dir
        try:
            if temp_dir_base.exists() and not any(temp_dir_base.iterdir()):
                temp_dir_base.rmdir()
        except Exception: pass
        return

    # --- Initialize Queues ---
    task_queue = mp.Queue()
    # Consider making queue size slightly larger relative to batch size/GPU count
    ready_queue_maxsize = num_gpus * args.gpu_batch_size * DEFAULT_QUEUE_MAX_SIZE_FACTOR * 2
    ready_queue = mp.Queue(maxsize=ready_queue_maxsize)
    print(f"\nReady queue max size: {ready_queue_maxsize}")

    # --- Start Workers ---
    video_creators = []
    print(f"Starting {args.num_video_creator_workers} video creator workers...")
    for _ in range(args.num_video_creator_workers):
        # Give temp_dir_base to worker
        p = mp.Process(target=video_creator_worker, args=(task_queue, ready_queue, temp_dir_base, args.frame_rate), daemon=True)
        video_creators.append(p)
        p.start()

    gpu_workers = []
    print(f"Starting {num_gpus} GPU inference workers...")
    for i in range(num_gpus):
        p = mp.Process(target=gpu_inference_worker, args=(i, ready_queue, args.output_dir, args.gpu_batch_size, args.gpu_worker_batch_timeout, args.model_name, args.max_new_tokens, args.target_dtype), daemon=True)
        gpu_workers.append(p)
        p.start()

    # --- Distribute Tasks ---
    print(f"Distributing {len(tasks_to_process)} tasks to video creators...")
    start_dist_time = time.time()
    for i, task in enumerate(tasks_to_process):
        while True: # Loop to handle full queue
             try:
                 task_queue.put(task, timeout=5) # Put with timeout
                 break # Exit inner loop if successful
             except queue.Full:
                 print(f"Warning: Task queue full, waiting... (Queued: {i}/{len(tasks_to_process)})")
                 time.sleep(2) # Wait before retrying
    end_dist_time = time.time()
    print(f"\nAll {len(tasks_to_process)} tasks queued in {end_dist_time - start_dist_time:.2f} seconds.")


    # --- Shutdown and Cleanup ---
    print("\n--- Starting Shutdown Sequence ---")
    # Signal Video Creators to Stop
    print("Sending stop signals to video creators...")
    for _ in range(args.num_video_creator_workers):
        task_queue.put(None) # Send poison pills

    # Wait for Video Creators to Finish
    print("Waiting for video creators to finish...")
    start_wait_vc = time.time()
    active_creators = args.num_video_creator_workers
    while active_creators > 0:
        if time.time() - start_wait_vc > 300: # Timeout for waiting
             print("\nWarning: Timeout waiting for video creators. Proceeding...")
             break
        active_creators = sum(1 for p in video_creators if p.is_alive())
        print(f"  Active video creators: {active_creators}, TaskQ: {task_queue.qsize()}, ReadyQ: {ready_queue.qsize()}   ", end='\r')
        time.sleep(2)
    print("\nVideo creators finished or timed out.                            ")


    # Wait for Ready Queue to Empty
    print("Waiting for GPU workers to process remaining tasks...")
    start_wait_gpu = time.time()
    processed_count = 0 # Estimate based on intermediate files? No, too complex here.
    while True:
        q_size = ready_queue.qsize()
        active_gpus = sum(1 for p in gpu_workers if p.is_alive())
        print(f"  Waiting for ready queue (Size: {q_size}), Active GPUs: {active_gpus}    ", end='\r')

        # Exit conditions
        if q_size == 0 and active_gpus == num_gpus: # Queue empty, all GPUs running
             # Give time for final batch processing
             final_wait_start = time.time()
             while time.time() - final_wait_start < max(10.0, args.gpu_worker_batch_timeout * 3):
                 q_size = ready_queue.qsize()
                 active_gpus = sum(1 for p in gpu_workers if p.is_alive())
                 if q_size > 0 or active_gpus < num_gpus: # If queue fills or GPU dies, break inner wait
                      print("\nChange detected, resuming check...")
                      break
                 print(f"  Final check: ReadyQ empty, GPUs active. Waiting {int(max(10.0, args.gpu_worker_batch_timeout * 3) - (time.time() - final_wait_start))}s... ", end='\r')
                 time.sleep(1)
             if q_size == 0 and active_gpus == num_gpus: # If still stable after wait
                 print("\nReady queue appears stable and empty, proceeding.")
                 break

        if active_gpus == 0: # All GPUs died
             if q_size > 0: print("\nError: All GPU workers exited but ready queue is not empty!")
             else: print("\nAll GPU workers exited.")
             break

        # Timeout for the whole GPU processing phase
        if time.time() - start_wait_gpu > 1800: # 30 min timeout for GPU processing
             print("\nWarning: Timeout waiting for GPUs to process queue. Proceeding with shutdown.")
             break

        time.sleep(5)


    print("\nGPU processing finished or timed out.                       ")


    # Signal GPU Workers to Stop
    print("Sending stop signals to GPU workers...")
    for _ in range(num_gpus):
        try: ready_queue.put(None, block=False)
        except queue.Full: pass # Ignore if full
        except Exception as e: print(f"Error putting stop signal: {e}")


    # Wait for GPU Workers to Terminate
    print("Waiting for GPU workers to terminate...")
    for i, p in enumerate(gpu_workers):
        p.join(timeout=90) # Increase join timeout
        if p.is_alive():
            print(f"Warning: GPU worker {i} (PID {p.pid}) did not terminate gracefully. Attempting terminate.")
            try: p.terminate() # Force terminate if stuck
            except Exception as e_term: print(f"Error terminating GPU worker {i}: {e_term}")


    print("All worker processes stopped or terminated.")

    # Clean up Temporary Directory
    try:
        remaining_files = list(temp_dir_base.glob('*'))
        if not remaining_files:
            temp_dir_base.rmdir()
            print(f"Cleaned up empty temporary directory: {temp_dir_base}")
        else:
             print(f"Warning: Temporary directory {temp_dir_base} is not empty ({len(remaining_files)} files remain). Manual cleanup might be needed.")
    except Exception as e:
        print(f"Error cleaning up temporary directory {temp_dir_base}: {e}")


    # Final Aggregation Step (with check)
    if not args.skip_aggregation:
        if not all_input_image_paths_absolute:
             print("\nInput path set is empty, cannot perform aggregation check.")
        else:
             aggregate_results(args.output_dir, FINAL_AGGREGATED_JSON_NAME, args.keep_intermediate, all_input_image_paths_absolute)
    else:
        print("\nSkipping final aggregation step as requested.")


    print("\n--- Script Finished ---")
    final_intermediate_count = len(list(args.output_dir.glob(f"*{INTERMEDIATE_JSON_SUFFIX}")))
    print(f"Total intermediate JSON files in output directory: {final_intermediate_count}")
    if not args.skip_aggregation:
         final_agg_path = args.output_dir / FINAL_AGGREGATED_JSON_NAME
         if final_agg_path.exists():
              # Final count confirmation from the aggregated file itself
              try:
                  with open(final_agg_path, 'r') as f: final_data = json.load(f)
                  print(f"Final aggregated JSON contains {len(final_data)} entries: {final_agg_path}")
              except Exception as e_read:
                   print(f"Could not read final aggregated JSON to confirm count: {e_read}")
         else:
              print("Final aggregated JSON was not created (check logs for errors).")


if __name__ == "__main__":
    try:
        current_method = mp.get_start_method(allow_none=True)
        if current_method != 'spawn':
             mp.set_start_method('spawn', force=True)
             print("Set multiprocessing start method to 'spawn'.")
    except RuntimeError as e:
        print(f"Warning: Could not set start method to 'spawn': {e}. Using default: {mp.get_start_method(allow_none=True)}")
        if mp.get_start_method(allow_none=True) != 'spawn':
             print("ERROR: Non-spawn start method may cause CUDA issues in child processes!")

    main()


# # Example command
# python /wekafs/ict/junyiouy/matrixcity_hf/captioning.py \
#     --sequence_json_path "/wekafs/ict/junyiouy/matrixcity_hf/small_city1.json" \
#     --output_dir "./matrixcity_captions_output" \
#     --sub_sequence_length 100 \
#     --frame_sampling_interval 2 \
#     --gpu_batch_size 4 \
#     --num_video_creator_workers 8 \
#     --model_name "Qwen/Qwen2.5-Omni-7B" \
#     --max_new_tokens 150 \
#     # --force_regenerate  # Add this if you want to re-run everything
#     # --keep_intermediate # Add this if you want to keep the seqX_subY_subcaption.json files