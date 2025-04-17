# process_sequences_refactored.py

import numpy as np
import json
import os
from PIL import Image, UnidentifiedImageError # Need Pillow: pip install Pillow
import imageio        # Need imageio: pip install imageio
import sys
import argparse # Added for __main__ block

# --- 辅助函数 ---

def normalize_pose_matrix(matrix_4x4):
    """
    将 4x4 位姿矩阵归一化，使其 3x3 旋转部分成为有效的旋转矩阵 (det=1)，
    并返回 [3, :4] 部分。能处理可能的反射情况。
    """
    if not isinstance(matrix_4x4, np.ndarray):
        matrix_4x4 = np.array(matrix_4x4)

    # 确保输入是 4x4 或 3x4
    if matrix_4x4.shape != (4, 4):
        if matrix_4x4.shape == (3, 4):
             matrix_4x4 = np.vstack([matrix_4x4, [0, 0, 0, 1]])
        else:
             # 如果形状不符合预期，抛出错误
             raise ValueError(f"输入矩阵必须是 4x4 或 3x4, 实际为 {matrix_4x4.shape}")
        
    # normalize 3x3 部分, 

    R = matrix_4x4[:3, :3]
    t = matrix_4x4[:3, 3]

    # 使用 SVD 来稳健地处理可能的缩放/切变/反射
    try:
        U, S, Vt = np.linalg.svd(R)
        R_closest = U @ Vt
        # 确保行列式为 +1 (处理反射)
        if np.linalg.det(R_closest) < 0:
            Vt[-1, :] *= -1 # 翻转最后一个奇异向量
            R_closest = U @ Vt

        # 创建归一化的 3x4 矩阵
        normalized_pose_3x4 = np.hstack((R_closest, t.reshape(3, 1)))
        return normalized_pose_3x4

    except np.linalg.LinAlgError:
        # SVD 计算失败时的警告和回退
        print(f"警告: SVD 计算失败，返回原始矩阵[:3, :4]. Matrix:\n{matrix_4x4[:3,:4]}")
        return matrix_4x4[:3, :4]
    except Exception as e:
        # 其他归一化错误的处理
        print(f"警告: 归一化过程中出错: {e}. 返回原始矩阵[:3, :4]. Matrix:\n{matrix_4x4[:3,:4]}")
        return matrix_4x4[:3, :4]

def get_pose_info(frame_data):
    """从原始 frame data 中提取位置、前向和右向向量。"""
    # 确保 matrix 是 numpy 数组
    matrix = np.array(frame_data['rot_mat'])
    # 确保 matrix 是 4x4, 如果是 3x4 则填充
    if matrix.shape != (4, 4):
        if matrix.shape == (3, 4):
             matrix = np.vstack([matrix, [0, 0, 0, 1]])
        else:
             # 处理其他意外形状
             raise ValueError(f"意外的矩阵形状: {matrix.shape} for frame index {frame_data.get('frame_index', 'N/A')}")

    position = matrix[:3, 3]
    right_vec = matrix[:3, 0]
    # Z 轴指向后方, 所以前向是 -Z
    forward_vec = -matrix[:3, 2]

    # 归一化向量, 处理可能的零向量
    fwd_norm = np.linalg.norm(forward_vec)
    rgt_norm = np.linalg.norm(right_vec)

    if fwd_norm > 1e-8:
        forward_vec /= fwd_norm
    else: # 处理零向量情况 (如果矩阵退化)
        forward_vec = np.array([0, 0, -1]) # 默认前向

    if rgt_norm > 1e-8:
        right_vec /= rgt_norm
    else: # 处理零向量情况
        right_vec = np.array([1, 0, 0]) # 默认右向

    return position, forward_vec, right_vec

def calculate_rotation_angle_degrees(rot_mat1, rot_mat2):
    """计算两个旋转矩阵之间的旋转角度（度数）。"""
    # 确保输入是 numpy 数组并处理 3x4 -> 4x4
    mats = []
    for rot_mat in [rot_mat1, rot_mat2]:
        mat_np = np.array(rot_mat)
        if mat_np.shape == (3,4):
            mat_np = np.vstack([mat_np, [0,0,0,1]])
        elif mat_np.shape != (4,4):
             # 基于参考代码，这里应该直接使用，但在角度计算中保持严格比较好
             print(f"警告: 角度计算中遇到意外的矩阵形状 {mat_np.shape}。返回 180 度。")
             return 180.0
        mats.append(mat_np)

    R1 = mats[0][:3, :3]
    R2 = mats[1][:3, :3]

    # 尝试近似归一化以提高角度计算稳定性（基于参考代码）
    try:
      det1 = np.linalg.det(R1)
      det2 = np.linalg.det(R2)
      # 避免除以零或接近零的行列式
      if abs(det1) > 1e-9: R1 = R1 / (abs(det1) ** (1/3)) # 使用 abs() 避免复数
      if abs(det2) > 1e-9: R2 = R2 / (abs(det2) ** (1/3))
    except np.linalg.LinAlgError:
       # 参考代码中的处理方式
       print("警告: 角度计算中的归一化时遇到奇异矩阵。")
       # 可以在这里返回 180.0，但原参考代码似乎没有，我们先保持，让后续计算处理
       pass

    try:
        R_rel = R2 @ R1.T
        trace = np.trace(R_rel)
        # 由于浮点数精度问题，将值限制在 [-1, 1]
        cos_angle = np.clip((trace - 1.0) / 2.0, -1.0, 1.0)
        angle_rad = np.arccos(cos_angle)
        return np.degrees(angle_rad)
    except Exception as e:
        print(f"计算旋转角度时出错: {e}")
        # 返回一个大角度表示问题或跳跃
        return 180.0 # 参考代码的处理

def segment_sequences_by_jumps(frames_data, rotation_threshold_degrees=15.0, displacement_threshold=3.0, min_seq_len=2):
    """
    根据连续帧之间的旋转或位移跳跃来分割序列。
    """
    if not frames_data or len(frames_data) < 2:
        return []

    # 按索引排序帧
    frames_data.sort(key=lambda f: f['frame_index'])

    potential_sequences = []
    current_sequence = [] # 存储 frame_index
    last_matrix_4x4 = None # 存储上一帧的 4x4 矩阵
    last_pos = None      # 存储上一帧的位置

    for i, frame_data in enumerate(frames_data):
        frame_idx = frame_data['frame_index']
        # 在循环开始时处理当前帧的矩阵
        try:
            matrix = np.array(frame_data['rot_mat'])
            if matrix.shape == (3,4):
                matrix_4x4 = np.vstack([matrix, [0,0,0,1]])
            elif matrix.shape == (4,4):
                matrix_4x4 = matrix
            else:
                raise ValueError("矩阵形状无效")
            current_pos = matrix_4x4[:3, 3]
        except (ValueError, TypeError, KeyError):
             # 如果当前帧的矩阵有问题，跳过它
             print(f"警告: 分割序列时跳过帧 {frame_idx}，因为矩阵无效或缺失。")
             continue

        break_sequence = False
        # 只有在有前一帧的数据时才进行比较
        if i > 0 and last_matrix_4x4 is not None and last_pos is not None:
            # 计算位移
            displacement = np.linalg.norm(current_pos - last_pos)
            if displacement > displacement_threshold:
                break_sequence = True

            # 如果尚未因位移中断，则计算旋转变化
            if not break_sequence:
                # 使用已经处理好的 4x4 矩阵进行角度计算
                rotation_change_deg = calculate_rotation_angle_degrees(last_matrix_4x4, matrix_4x4)
                if rotation_change_deg > rotation_threshold_degrees:
                    break_sequence = True

        # --- 序列管理 ---
        if break_sequence:
            # 结束上一个序列（如果足够长）
            if len(current_sequence) >= min_seq_len:
                potential_sequences.append(current_sequence)
            # 用当前帧开始一个新序列
            current_sequence = [frame_idx]
        else:
            # 将当前帧添加到当前序列
            current_sequence.append(frame_idx)

        # 更新下一轮迭代的状态
        last_matrix_4x4 = matrix_4x4
        last_pos = current_pos

    # 添加最后一个序列（如果有效）
    if len(current_sequence) >= min_seq_len:
        potential_sequences.append(current_sequence)

    return potential_sequences

def filter_sequences_by_motion_direction(sequences, frame_map,
                                         forward_vs_side_threshold=2.0,
                                         min_overall_forward_disp=0.2,
                                         min_overall_backward_disp=0.2):
    """
    过滤序列，保留那些显示出显著整体前向或后向运动的序列。
    返回两个列表：forward_sequences 和 backward_sequences。
    """
    forward_sequences = []
    backward_sequences = []
    if not sequences:
        return [], []

    for seq in sequences: # seq 是 frame_index 的列表
        if len(seq) < 2:
            continue # 需要至少两帧来判断运动

        start_frame_idx = seq[0]
        end_frame_idx = seq[-1]

        # 确保起始帧和结束帧的数据在 map 中存在
        if start_frame_idx not in frame_map or end_frame_idx not in frame_map:
            # print(f"警告: 跳过序列 {seq} 因为在 map 中缺少帧数据。")
            continue

        try:
            # 使用 get_pose_info 获取方向信息（基于原始矩阵）
            start_pos, start_fwd, start_rgt = get_pose_info(frame_map[start_frame_idx])
            # 只需要结束帧的位置
            end_matrix = np.array(frame_map[end_frame_idx]['rot_mat'])
            if end_matrix.shape == (3,4): end_matrix = np.vstack([end_matrix, [0,0,0,1]])
            elif end_matrix.shape != (4,4): raise ValueError("结束帧矩阵形状无效")
            end_pos = end_matrix[:3, 3]

        except (ValueError, KeyError) as e:
            print(f"警告: 跳过序列 {seq} 因为位姿信息错误: {e}")
            continue

        total_displacement = end_pos - start_pos
        total_disp_norm = np.linalg.norm(total_displacement)

        if total_disp_norm < 1e-5:
            # print(f"  调试: 跳过序列 {seq} - 整体位移可忽略。")
            continue # 跳过几乎没有整体移动的序列

        # 将整体位移投影到序列的 *起始* 方向上
        forward_component = np.dot(total_displacement, start_fwd)
        sideways_component = np.dot(total_displacement, start_rgt)

        # 检查前/后向运动是否主导侧向运动
        is_dominant_fwd_bwd = False
        if abs(sideways_component) < 1e-5: # 侧向运动可忽略
             is_dominant_fwd_bwd = True
        # 检查比率，避免除以零
        elif abs(forward_component) / (abs(sideways_component) + 1e-9) > forward_vs_side_threshold:
             is_dominant_fwd_bwd = True

        # 检查前向或后向运动
        if is_dominant_fwd_bwd:
            if forward_component >= min_overall_forward_disp: # 检查前向阈值
                forward_sequences.append(seq)
            elif forward_component <= -min_overall_backward_disp: # 检查后向阈值 (负方向)
                backward_sequences.append(seq)

    return forward_sequences, backward_sequences


# --- 子采样辅助函数 (保持不变) ---
def get_subsampled_indices(sequence_length, target_frames):
    """计算时间子采样的索引。"""
    if sequence_length <= 0:
        return []
    if sequence_length <= target_frames:
        return np.arange(sequence_length)
    else:
        indices_float = np.linspace(0, sequence_length - 1, target_frames)
        # 四舍五入并去重，确保索引是整数且唯一
        indices_int = sorted(list(set(np.round(indices_float).astype(int))))
        return np.array(indices_int)


# --- 单个数据集的主要处理函数 ---

def extract_sequences_and_poses(json_file_path, dataset_dir, output_dir,
                                rotation_threshold_degrees=15.0,
                                displacement_threshold=3.0,
                                forward_vs_side_threshold=2.0,
                                min_overall_forward_disp=0.2,
                                min_overall_backward_disp=0.2,
                                min_seq_len_initial=2,
                                # 移除了 image_subdir 参数，遵循参考代码逻辑
                                num_gifs_to_save=0, # 默认不保存 GIF
                                target_gif_frames=25,
                                gif_fps=10,
                                verbose=False):
    """
    处理单个数据集。返回序列路径和姿态映射。可选择性地保存 GIF。
    """
    if verbose: print(f"\n--- 正在处理数据集: {dataset_dir} ---")
    if verbose: print(f"加载 JSON: {json_file_path}")

    # 1. 加载 JSON 数据 (稳健加载)
    try:
        with open(json_file_path, 'r') as f: data = json.load(f)
    except FileNotFoundError:
        print(f"错误: JSON 文件未找到于 {json_file_path}"); return [], {}
    except json.JSONDecodeError as e:
        print(f"错误: 无法解码 JSON {json_file_path}. 错误: {e}"); return [], {}
    except Exception as e:
        print(f"加载 JSON 时发生意外错误: {e}"); return [], {}

    frames_data = data.get('frames', [])
    if not frames_data:
        print(f"错误: 在 {json_file_path} 中未找到 'frames' 数据。"); return [], {}

    # 2. 创建 frame_map, 验证帧, 解析路径, 并归一化姿态
    frame_map = {} # key: frame_index, value: frame_data (包含 'rot_mat_np', 'abs_image_path')
    valid_frames_data_for_seg = [] # 仅存储用于分割的有效帧数据
    path_to_normalized_pose = {}   # key: abs_image_path, value: normalized_pose_list
    processed_paths = set()        # 跟踪已处理的路径以进行姿态归一化
    skipped_count = 0
    path_resolution_warnings = 0

    if verbose: print("正在验证帧，解析路径，并归一化姿态...")
    for i, frame in enumerate(frames_data):
        idx = frame.get('frame_index')
        mat_list = frame.get('rot_mat')

        # 基本检查：索引和矩阵列表是否存在
        if idx is None or not isinstance(mat_list, list):
            print(f"警告: 帧 {i} (索引: {idx}) 缺少 'frame_index' 或 'rot_mat' 不是列表。跳过。")
            skipped_count += 1; continue

        try:
            # --- 矩阵处理 ---
            mat_np_orig = np.array(mat_list) # 使用原始列表创建 numpy 数组
            if mat_np_orig.shape == (3,4):
                mat_np_4x4 = np.vstack([mat_np_orig, [0,0,0,1]])
            elif mat_np_orig.shape == (4,4):
                mat_np_4x4 = mat_np_orig
            else:
                # 基于参考代码，这里应该跳过而不是填充
                print(f"警告: 帧索引 {idx} 的 'rot_mat' 结构无效 ({mat_np_orig.shape})。跳过帧。")
                skipped_count += 1; continue

            # --- 路径解析 (严格遵循参考代码逻辑) ---
            abs_path = None
            full_image_path = None # 用于记录尝试的路径
            relative_path = frame.get('file_path')
            image_source_info = "" # 用于错误消息

            if relative_path:
                # JSON 中提供了 file_path
                full_image_path = os.path.join(dataset_dir, relative_path)
                image_source_info = f"来自 'file_path': {relative_path}"
                # 检查文件是否存在以确保路径有效
                if os.path.isfile(full_image_path):
                     abs_path = os.path.abspath(full_image_path)
                # else: 在这种严格模式下，如果指定路径无效，则失败

            else:
                # file_path 缺失，使用回退模式
                # !! 使用参考代码中的 :04d 格式 !!
                image_filename = f"{idx:04d}.png"
                full_image_path = os.path.join(dataset_dir, image_filename)
                image_source_info = f"使用回退模式: {image_filename}"
                if os.path.isfile(full_image_path):
                     abs_path = os.path.abspath(full_image_path)
                # else: 如果回退模式也找不到文件，则失败

            # 如果两种方式都无法解析到有效路径
            if not abs_path:
                if verbose and path_resolution_warnings < 10: # 限制警告数量
                    print(f"警告: 无法解析帧 {idx} 的图像路径 ({image_source_info})。尝试的路径: '{full_image_path}'. 跳过此帧。")
                path_resolution_warnings += 1
                skipped_count += 1
                continue

            # --- 姿态归一化 ---
            # 仅对首次遇到的路径进行归一化
            if abs_path not in processed_paths:
                 try:
                    # 使用已验证的 4x4 矩阵进行归一化
                    norm_pose = normalize_pose_matrix(mat_np_4x4)
                    path_to_normalized_pose[abs_path] = norm_pose.tolist() # 存储为列表以兼容 JSON
                    processed_paths.add(abs_path)
                 except Exception as e:
                    print(f"错误: 归一化帧 {idx} ({abs_path}) 的姿态时出错: {e}")
                    skipped_count += 1; continue # 如果归一化失败则跳过

            # --- 存储到 frame_map ---
            # 添加或覆盖 frame_map 中的条目
            # 保留原始 rot_mat 用于 get_pose_info 和角度计算 (如果需要)
            # 存储 4x4 矩阵和绝对路径以便后续使用
            frame_copy = frame.copy() # 避免修改原始数据列表中的字典
            frame_copy['rot_mat'] = mat_list # 保留原始列表形式
            frame_copy['rot_mat_np_4x4'] = mat_np_4x4 # 存储验证过的 4x4 numpy 矩阵
            frame_copy['abs_image_path'] = abs_path
            frame_map[idx] = frame_copy
            valid_frames_data_for_seg.append(frame_copy) # 添加到用于分割的列表

        except Exception as e:
             # 捕获处理单个帧时可能出现的其他错误
             if verbose: print(f"警告: 处理帧 {idx} 时发生错误: {e}。跳过。")
             skipped_count += 1

    if path_resolution_warnings > 10: print(f"警告: 已抑制 {path_resolution_warnings-10} 条额外的路径解析警告。")
    if verbose: print(f"处理了 {len(frames_data)} 帧。有效且路径已解析: {len(frame_map)}。跳过: {skipped_count}。唯一姿态: {len(path_to_normalized_pose)}。")
    if not frame_map: print("错误: 处理后未找到有效帧。"); return [], {}

    # 3. 根据跳跃分割序列 (使用验证后的帧数据列表)
    if verbose: print("正在根据跳跃分割序列...")
    potential_sequences_idx = segment_sequences_by_jumps(
        valid_frames_data_for_seg, # 使用包含验证后数据的列表
        rotation_threshold_degrees,
        displacement_threshold,
        min_seq_len_initial
    ) # 返回 frame_index 的列表组成的列表
    if verbose: print(f"找到 {len(potential_sequences_idx)} 个潜在序列。")
    # 即使没有潜在序列，也可能已经收集了一些姿态
    if not potential_sequences_idx: return [], path_to_normalized_pose

    # 4. 过滤前向和后向运动
    if verbose: print("正在过滤前向/后向运动...")
    forward_seq_idx, backward_seq_idx = filter_sequences_by_motion_direction(
        potential_sequences_idx,
        frame_map, # 传递包含原始 'rot_mat' 和处理后 'rot_mat_np_4x4' 的 map
        forward_vs_side_threshold,
        min_overall_forward_disp,
        min_overall_backward_disp
    )
    if verbose: print(f"找到 {len(forward_seq_idx)} 个前向序列, {len(backward_seq_idx)} 个后向序列。")

    # 5. 将序列从索引转换为路径，并反转后向序列
    all_sequences_paths = []
    final_tagged_sequences = [] # 用于 GIF 生成和可选的索引保存

    for seq_idx in forward_seq_idx:
        # 确保序列中的每个索引都在 frame_map 中，并获取其绝对路径
        path_seq = [frame_map[idx]['abs_image_path'] for idx in seq_idx if idx in frame_map]
        # 检查路径列表的长度是否与索引列表匹配（防止中途有帧被跳过）
        if len(path_seq) == len(seq_idx):
             all_sequences_paths.append(path_seq)
             final_tagged_sequences.append({'direction': 'forward', 'indices': seq_idx, 'paths': path_seq})

    for seq_idx in backward_seq_idx:
        path_seq = [frame_map[idx]['abs_image_path'] for idx in seq_idx if idx in frame_map]
        if len(path_seq) == len(seq_idx):
             # 反转路径列表用于后向序列
             reversed_path_seq = list(reversed(path_seq))
             all_sequences_paths.append(reversed_path_seq)
             final_tagged_sequences.append({'direction': 'backward', 'indices': seq_idx, 'paths': reversed_path_seq})

    if verbose: print(f"创建了 {len(all_sequences_paths)} 个最终路径序列。")

    # --- 可选：GIF 生成 ---
    if num_gifs_to_save > 0 and final_tagged_sequences: # 使用 final_tagged_sequences
        if verbose: print(f"\n--- 正在生成最长的 Top {num_gifs_to_save} 个序列的 GIF ---")

        # 确保输出目录存在
        try:
            os.makedirs(output_dir, exist_ok=True)
            if verbose: print(f"已确保输出目录存在: {output_dir}")
        except OSError as e:
            print(f"错误: 创建 GIF 输出目录 '{output_dir}' 失败: {e}。无法保存 GIF。")
            num_gifs_to_save = 0 # 禁用 GIF 保存

        if num_gifs_to_save > 0:
            # 按原始长度排序（使用索引列表长度）
            final_tagged_sequences.sort(key=lambda item: len(item['indices']), reverse=True)
            sequences_to_process_for_gif = final_tagged_sequences[:num_gifs_to_save]

            if verbose: print(f"已选择 Top {len(sequences_to_process_for_gif)} 个序列用于生成 GIF。")

            gifs_created_count = 0
            for i, item in enumerate(sequences_to_process_for_gif):
                seq_rank = i + 1
                direction = item['direction']
                original_sequence_indices = item['indices'] # 用于信息展示
                # !! 使用已经处理好（且为后向序列反转过）的路径列表 !!
                sequence_paths_for_gif = item['paths']
                original_seq_len = len(sequence_paths_for_gif) # 长度现在基于路径列表

                if verbose: print(f"\n正在处理 GIF {seq_rank}/{len(sequences_to_process_for_gif)} ({direction.upper()}, 原始长度: {original_seq_len})")

                # 子采样步骤（使用路径列表的长度）
                # get_subsampled_indices 返回的是相对于输入列表的索引
                subsampled_relative_indices = get_subsampled_indices(original_seq_len, target_gif_frames)
                if len(subsampled_relative_indices) == 0:
                     if verbose: print("  跳过: 子采样后无帧。")
                     continue

                # 从（可能已反转的）路径列表中选择子采样后的路径
                subsampled_paths = [sequence_paths_for_gif[rel_idx] for rel_idx in subsampled_relative_indices]
                num_frames_for_gif = len(subsampled_paths)
                if verbose: print(f"  子采样得到 {num_frames_for_gif} 帧。")

                # 为 GIF 加载图像
                sequence_images = []
                loaded_image_count = 0
                img_load_errors = 0
                # 直接遍历子采样后的路径列表
                print(f"  正在加载 {num_frames_for_gif} 张图片 (顺序: {'反转' if direction == 'backward' else '原始'})...") # 明确顺序
                for abs_path in subsampled_paths:
                    try:
                        img = Image.open(abs_path).convert('RGB')
                        sequence_images.append(np.array(img))
                        loaded_image_count += 1
                    # 捕获特定和通用的图像加载错误
                    except FileNotFoundError:
                         print(f"  错误: GIF 加载时文件未找到 '{abs_path}'。")
                         img_load_errors += 1
                    except UnidentifiedImageError:
                         print(f"  错误: 无法识别的图像文件 '{abs_path}'。")
                         img_load_errors += 1
                    except Exception as e:
                         if verbose: print(f"  警告: 为 GIF 加载图像 {abs_path} 失败: {e}")
                         img_load_errors += 1

                if verbose: print(f"  为 GIF 加载了 {loaded_image_count} 张图像 ({img_load_errors} 个错误)。")

                # 保存 GIF
                actual_gif_frames = len(sequence_images)
                if actual_gif_frames >= 2: # GIF 至少需要 2 帧
                    # 使用与参考代码一致的文件名格式
                    output_gif_filename = f"{direction}_seq_rank_{seq_rank}_origlen_{original_seq_len}_frames_{actual_gif_frames}.gif"
                    output_gif_path = os.path.join(output_dir, output_gif_filename)
                    try:
                        print(f"  正在保存 GIF ({actual_gif_frames} 帧) 到: {output_gif_path}")
                        imageio.mimsave(output_gif_path, sequence_images, fps=gif_fps)
                        gifs_created_count += 1
                    except Exception as e:
                        print(f"  错误: 保存 GIF {output_gif_path} 失败: {e}")
                else:
                    if verbose: print(f"  跳过 GIF 生成: 仅加载了 {actual_gif_frames} 张有效图像 (需要 >= 2)。")

            if verbose: print(f"\nGIF 生成完成。共创建 {gifs_created_count} 个 GIF。")


    # --- 返回主要结果 ---
    # all_sequences_paths: 包含绝对图像路径的列表的列表
    # path_to_normalized_pose: 绝对图像路径到其归一化 [3, :4] 姿态（列表形式）的映射
    return all_sequences_paths, path_to_normalized_pose


# --- 主执行块 ---
if __name__ == "__main__":
    # 依赖检查 (可选，但良好实践)
    try:
        # 检查必要的库是否可导入
        from PIL import Image, UnidentifiedImageError
        import imageio
        import numpy
    except ImportError as e:
        print(f"错误: 缺少必需的库。 {e}")
        print("请安装必需的库: pip install Pillow imageio numpy")
        sys.exit(1)

    # --- Argparse 设置 (基于参考代码调整) ---
    parser = argparse.ArgumentParser(description="处理单个数据集：提取运动序列，归一化姿态，并可选择性地生成 GIF。")

    # 输入/输出
    parser.add_argument('--input_json_path', type=str, default='/wekafs/ict/junyiouy/matrixcity_hf/small_city_road_outside_dense_transforms.json', # 使其必需，因为没有默认值通常更好
                        help="输入 JSON 文件的路径 (例如, transforms.json)。")
    parser.add_argument('--dataset_dir', type=str, default='/wekafs/ict/junyiouy/matrixcity_hf/small_city_road_outside_dense',
                        help="包含图像的根数据集目录的路径。")
    parser.add_argument('--output_dir', type=str, default='./fwd_bwd_sequence_gifs_filtered_subsampled_bf', # 重命名以匹配函数参数
                        help="用于保存输出 GIF 和可选的其他文件（如保存的姿态）的目录。")

    # 序列检测参数 (匹配参考代码默认值)
    parser.add_argument('--rotation_threshold_degrees', type=float, default=5.0,
                        help="用于序列中断的旋转阈值（度）。默认值: 5.0")
    parser.add_argument('--displacement_threshold', type=float, default=3.0,
                        help="用于序列中断的位移阈值（单位）。默认值: 3.0")
    parser.add_argument('--min_seq_len_initial', type=int, default=3,
                        help="初始跳跃分割后序列所需的最小帧数。默认值: 3")

    # 运动过滤参数 (匹配参考代码默认值)
    parser.add_argument('--forward_vs_side_threshold', type=float, default=2.0,
                        help="前/后向运动与侧向运动幅度之间的最小比率。默认值: 2.0")
    parser.add_argument('--min_overall_forward_disp', type=float, default=0.01,
                        help="序列被分类为 '前向' 所需的最小总前向位移。默认值: 0.01")
    parser.add_argument('--min_overall_backward_disp', type=float, default=0.01,
                        help="序列被分类为 '后向' 所需的最小总后向位移（幅度）。默认值: 0.01")

    # GIF 生成参数 (匹配参考代码默认值)
    parser.add_argument('--num_sequences_to_save', type=int, default=0, # 函数参数叫 num_gifs_to_save，这里保持一致
                        dest='num_gifs_to_save',
                        help="要另存为 GIF 的最长有效序列的最大数量。设置为 0 禁用。默认值: 50")
    parser.add_argument('--target_gif_frames', type=int, default=25,
                        help="每个输出 GIF 的目标帧数（通过子采样）。默认值: 25")
    parser.add_argument('--gif_fps', type=int, default=8,
                        help="输出 GIF 的帧率。默认值: 8")

    # 其他选项
    # 移除了 --image-subdir 参数
    parser.add_argument('-v', '--verbose', action='store_true',
                        help="启用详细的打印输出。")
    parser.add_argument('--save-poses', action='store_true',
                        help="将收集到的归一化姿态保存到输出目录中的 JSON 文件。")
    # 参考代码中有一个保存序列索引的逻辑，这里用 --save-sequence-indices 模拟
    parser.add_argument('--save-sequence-indices', action='store_true',
                         help="将最终过滤后的序列索引（后向反转）保存到输出目录的 JSON 文件。")


    args = parser.parse_args()

    # 如果需要保存任何内容，请确保输出目录存在
    if args.num_gifs_to_save > 0 or args.save_poses or args.save_sequence_indices:
        try:
            os.makedirs(args.output_dir, exist_ok=True)
        except OSError as e:
            print(f"错误: 无法创建输出目录 '{args.output_dir}': {e}")
            sys.exit(1)


    # --- 调用主处理函数 ---
    # 注意函数现在需要 output_dir
    sequences_paths, poses_map = extract_sequences_and_poses(
        json_file_path=args.input_json_path,
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir, # 传递输出目录
        rotation_threshold_degrees=args.rotation_threshold_degrees,
        displacement_threshold=args.displacement_threshold,
        forward_vs_side_threshold=args.forward_vs_side_threshold,
        min_overall_forward_disp=args.min_overall_forward_disp,
        min_overall_backward_disp=args.min_overall_backward_disp,
        min_seq_len_initial=args.min_seq_len_initial,
        # image_subdir 参数已移除
        num_gifs_to_save=args.num_gifs_to_save,
        target_gif_frames=args.target_gif_frames,
        gif_fps=args.gif_fps,
        verbose=args.verbose
    )

    print("\n--- 脚本执行摘要 ---")
    print(f"总共找到 {len(sequences_paths)} 个序列。")
    print(f"收集了 {len(poses_map)} 个唯一的归一化姿态。")

    # 可选：保存收集到的姿态
    if args.save_poses:
        base_input_name = os.path.splitext(os.path.basename(args.input_json_path))[0]
        output_poses_filename = f"{base_input_name}_normalized_poses.json"
        output_poses_path = os.path.join(args.output_dir, output_poses_filename)
        print(f"正在将归一化的姿态保存到: {output_poses_path}")
        try:
            with open(output_poses_path, 'w') as f:
                # 将 numpy 数组转换回列表以便 JSON 序列化
                poses_to_save = {k: v if isinstance(v, list) else v.tolist() for k, v in poses_map.items()}
                json.dump(poses_to_save, f, indent=4)
            print("成功保存姿态。")
        except Exception as e:
            print(f"错误: 保存姿态失败: {e}")

    # 可选：保存最终序列索引（模拟参考代码行为）
    if args.save_sequence_indices:
        # 需要重新从 final_tagged_sequences 构建用于保存的列表
        # 这个列表在 extract_sequences_and_poses 函数作用域内，需要修改函数使其返回
        # 或者在这里重新执行部分逻辑（不推荐）。
        # 为了简单起见，我们假设 extract_sequences_and_poses 返回了 final_tagged_sequences
        # 或者我们可以在 extract_sequences_and_poses 内部处理这个保存逻辑
        print("注意：保存序列索引的逻辑需要修改 extract_sequences_and_poses 函数或在此处重新计算。当前未实现。")
        # 如果需要实现：
        # 1. 修改 extract_sequences_and_poses 返回 final_tagged_sequences
        # 2. 在这里处理它:
        #    if final_tagged_sequences:
        #        sequences_to_save_json = []
        #        # 确保按长度排序
        #        final_tagged_sequences.sort(key=lambda item: len(item['indices']), reverse=True)
        #        for item in final_tagged_sequences:
        #            seq_indices = item['indices']
        #            if item['direction'] == 'backward':
        #                sequences_to_save_json.append(list(reversed(seq_indices)))
        #            else:
        #                sequences_to_save_json.append(seq_indices)
        #        base_input_name = os.path.splitext(os.path.basename(args.input_json_path))[0]
        #        output_seq_idx_filename = f"{base_input_name}_final_fwd_bwd_sequences_filtered.json"
        #        output_seq_idx_path = os.path.join(args.output_dir, output_seq_idx_filename)
        #        print(f"正在将最终序列索引保存到: {output_seq_idx_path}")
        #        try:
        #            with open(output_seq_idx_path, 'w') as f:
        #                json.dump(sequences_to_save_json, f, indent=4)
        #            print("成功保存序列索引。")
        #        except Exception as e:
        #            print(f"错误: 保存序列索引失败: {e}")


    # 也可以选择保存序列路径列表
    # output_seq_paths_filename = f"{base_input_name}_sequence_paths.json"
    # output_seq_paths_path = os.path.join(args.output_dir, output_seq_paths_filename)
    # try:
    #     with open(output_seq_paths_path, 'w') as f: json.dump(sequences_paths, f, indent=2)
    #     print(f"序列路径列表已保存到: {output_seq_paths_path}")
    # except Exception as e: print(f"错误: 保存序列路径列表失败: {e}")


    print("\n脚本完成。")