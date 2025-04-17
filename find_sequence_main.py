# run_multi_dataset_processing.py

import os
import json
import sys
import argparse
# 导入我们之前确认的模块中的核心函数
try:
    from find_sequence_module import extract_sequences_and_poses
except ImportError:
    print("错误：无法导入 'find_sequence_module.py'。请确保该文件存在于 Python 路径或当前目录中。")
    sys.exit(1)

def main(dataset_list, output_json_path, verbose=False, **kwargs):
    """
    处理多个数据集，聚合序列（包含路径和姿态），并将结果保存为 JSON。

    Args:
        dataset_list (list): 一个元组列表，每个元组是 (dataset_dir, input_json_path)。
        output_json_path (str): 保存聚合后序列数据的 JSON 文件的路径。
        verbose (bool): 是否打印详细进度。
        **kwargs: 传递给 extract_sequences_and_poses 的其他参数 (例如阈值, GIF 设置)。
    """

    all_sequences_paths_aggregated = []
    all_path_to_pose_map_aggregated = {}

    print(f"准备处理 {len(dataset_list)} 个数据集...")

    # 准备传递给模块的参数
    module_params = {
        'rotation_threshold_degrees': kwargs.get('rotation_threshold_degrees', 5.0),
        'displacement_threshold': kwargs.get('displacement_threshold', 3.0),
        'forward_vs_side_threshold': kwargs.get('forward_vs_side_threshold', 2.0),
        'min_overall_forward_disp': kwargs.get('min_overall_forward_disp', 0.01),
        'min_overall_backward_disp': kwargs.get('min_overall_backward_disp', 0.01),
        'min_seq_len_initial': kwargs.get('min_seq_len_initial', 3),
        'num_gifs_to_save': kwargs.get('num_gifs_to_save', 0), # 聚合时默认不生成 GIF
        'target_gif_frames': kwargs.get('target_gif_frames', 25),
        'gif_fps': kwargs.get('gif_fps', 8),
        'verbose': verbose
    }

    output_base_dir = os.path.dirname(output_json_path)
    if not output_base_dir: output_base_dir = "."

    for i, (dataset_dir, input_json_path) in enumerate(dataset_list):
        print(f"\n--- 数据集 {i+1}/{len(dataset_list)} ---")
        if not os.path.isdir(dataset_dir):
            print(f"警告: 数据集目录未找到: {dataset_dir}。跳过。")
            continue
        if not os.path.isfile(input_json_path):
            print(f"警告: 输入 JSON 未找到: {input_json_path}。跳过数据集 {dataset_dir}。")
            continue

        # 为当前数据集确定输出目录 (如果需要模块输出 GIF 等)
        current_output_dir_for_module = os.path.join(output_base_dir, f"dataset_{i+1}_output")
        if module_params['num_gifs_to_save'] > 0 : # 仅当需要时创建
             try:
                 os.makedirs(current_output_dir_for_module, exist_ok=True)
             except OSError as e:
                 print(f"警告：无法为数据集 {i+1} 创建输出目录 '{current_output_dir_for_module}': {e}")

        # --- 调用模块的处理函数 ---
        dataset_sequences_paths, dataset_poses_map = extract_sequences_and_poses(
            json_file_path=input_json_path,
            dataset_dir=dataset_dir,
            output_dir=current_output_dir_for_module, # 传递给模块
            **module_params
        )

        # 聚合结果 (路径列表和姿态映射)
        all_sequences_paths_aggregated.extend(dataset_sequences_paths)
        all_path_to_pose_map_aggregated.update(dataset_poses_map)

        if verbose:
             print(f"数据集处理完成: {dataset_dir} | "
                   f"序列: {len(dataset_sequences_paths)} | "
                   f"新姿态: {len(dataset_poses_map)}") # 输出当前数据集结果


    # --- 数据结构转换 ---
    print("\n--- 正在转换数据结构以匹配输出格式 ---")
    final_output_data = []
    sequences_with_missing_poses = 0
    frames_with_missing_poses = 0

    for path_sequence in all_sequences_paths_aggregated:
        sequence_data = []
        valid_sequence = True
        for image_path in path_sequence:
            if image_path in all_path_to_pose_map_aggregated:
                normalized_pose = all_path_to_pose_map_aggregated[image_path]
                # 确保姿态是列表形式
                if not isinstance(normalized_pose, list):
                     normalized_pose = normalized_pose.tolist()

                sequence_data.append({
                    "image_path": image_path,
                    "normalized_pose": normalized_pose
                    # 可以根据需要添加其他信息，例如 frame_index (如果模块返回了的话)
                })
            else:
                # 如果在聚合的姿态映射中找不到某个图像的姿态，记录错误
                print(f"错误: 在聚合的姿态映射中未找到图像 '{image_path}' 的姿态。该帧将从序列中省略。")
                frames_with_missing_poses += 1
                # 可以选择将整个序列标记为无效
                # valid_sequence = False
                # break # 如果一个帧缺失则跳过整个序列

        # 只有当序列仍然有效（或我们允许部分序列）并且包含数据时才添加
        if valid_sequence and sequence_data:
            final_output_data.append(sequence_data)
        elif not sequence_data:
             pass # 如果序列变空则不添加
        else: # valid_sequence is False
             sequences_with_missing_poses += 1

    if frames_with_missing_poses > 0:
         print(f"警告: 由于姿态缺失，共有 {frames_with_missing_poses} 帧从序列中省略。")
    if sequences_with_missing_poses > 0:
         print(f"警告: 由于帧姿态缺失，共有 {sequences_with_missing_poses} 个序列被完全跳过。")

    # --- 保存最终聚合和转换后的数据 ---
    print(f"\n--- 聚合结果摘要 ---")
    print(f"最终聚合得到的序列数量: {len(final_output_data)}")
    print(f"收集到的唯一图像姿态总数: {len(all_path_to_pose_map_aggregated)}") # 仍然可以报告这个

    # 确保输出目录存在
    if output_base_dir and not os.path.exists(output_base_dir):
        try:
            os.makedirs(output_base_dir)
            print(f"已创建聚合输出目录: {output_base_dir}")
        except OSError as e:
            print(f"错误: 创建聚合输出目录 '{output_base_dir}' 失败: {e}。将在当前目录保存 JSON。")
            output_json_path = os.path.basename(output_json_path) # 回退

    print(f"正在将最终聚合的序列数据保存到: {output_json_path}")
    try:
        with open(output_json_path, 'w') as f:
            # 直接将 final_output_data (列表的列表的字典) dump 到 JSON
            json.dump(final_output_data, f, indent=2) # 使用 indent=2 使输出更可读
        print("成功保存聚合序列 JSON。")
    except IOError as e:
        print(f"错误: 无法将聚合序列 JSON 写入 {output_json_path}: {e}")
    except Exception as e:
        print(f"保存聚合序列 JSON 时发生意外错误: {e}")

    # --- 返回转换后的数据结构 ---
    return final_output_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="处理多个数据集以提取运动序列（含姿态）并保存为 JSON。")
    parser.add_argument('-o', '--output-json', type=str, required=True,
                        help="用于保存最终聚合序列数据（列表的列表的字典）的输出 JSON 文件路径。")
    parser.add_argument('-v', '--verbose', action='store_true',
                        help="启用详细的打印输出。")
    # 添加用于覆盖模块默认值的参数 (示例)
    parser.add_argument('--rot-thresh', type=float, help="覆盖模块中的旋转阈值")
    parser.add_argument('--disp-thresh', type=float, help="覆盖模块中的位移阈值")
    # ... 可以添加更多 ...

    args = parser.parse_args()
    

    # --- 定义数据集列表 ---
    # datasets_to_process = [
    #      ('/wekafs/ict/junyiouy/matrixcity_hf/small_city_road_outside_dense', '/wekafs/ict/junyiouy/matrixcity_hf/small_city_road_outside_dense_transforms.json'),
    #     # ('/path/to/dataset2', '/path/to/dataset2/transforms.json'),
    # ]
    from dataset_to_process import datasets_to_process # 从外部文件导入数据集列表
    # ---------------------

    if not datasets_to_process:
        print("错误: 'datasets_to_process' 列表为空。请编辑脚本并添加数据集。")
        sys.exit(1)

    # 收集要传递给 main 函数的 kwargs
    module_override_params = {}
    if args.rot_thresh is not None: module_override_params['rotation_threshold_degrees'] = args.rot_thresh
    if args.disp_thresh is not None: module_override_params['displacement_threshold'] = args.disp_thresh
    # ... 添加其他覆盖参数 ...

    # 调用主处理函数
    final_structured_sequences = main(
        dataset_list=datasets_to_process,
        output_json_path=args.output_json,
        verbose=args.verbose,
        **module_override_params # 传递覆盖参数
    )

    print("\n多数据集处理和数据结构转换完成。")
    # final_structured_sequences 变量现在包含所需格式的数据
    # print(f"脚本返回了 {len(final_structured_sequences)} 个结构化序列。")