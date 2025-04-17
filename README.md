# 数据集处理

本部分介绍如何准备用于后续任务的数据集。

## 步骤

1.  **下载和处理数据集：**
    *   运行 `download_img.sh` 和 `download_depth.sh` 脚本下载图像和深度数据。
    *   **注意：** 脚本中的数据集链接可能不完整，并且可能需要手动下载相机参数。请确保检查并补全链接，并获取所需的相机参数文件。
2.  **配置数据集路径：**
    *   编辑 `dataset_to_process.py` 文件，根据您的实际情况修改数据集的根路径。
3.  **生成数据序列：**
    *   运行 `find_sequence_main.py` 脚本，该脚本会生成数据集的序列。
    *   一个序列是一个列表，其中每个元素都是一个字典，包含图像路径和相机参数。
    *   `find_sequence_main.py`  脚本依赖于  `find_sequence_module.py`  和  `find_sequences.py`  来提取和处理序列。它使用这些模块中的函数，如  `extract_sequences_and_poses`  (在`find_sequence_module.py`中) 和  `get_pose_info`, `calculate_rotation_angle_degrees`, `segment_sequences_by_jumps`, `filter_sequences_for_forward_motion`, `get_subsampled_indices`, `create_sequence_gifs` (在`find_sequences.py`中) 来分析和组织数据。
4.  **生成图像描述 (Captioning)：**
    *   运行 `captioning.py` 脚本，该脚本会为数据集生成图像描述。
    *   `captioning.py` 脚本使用 `parse_args` 函数解析命令行参数，使用 `process_gpu_batch` 函数处理批量数据，并使用 `parse_and_clean_caption_output` 函数解析和清理模型的输出文本。

## 脚本说明

*   **`download_img.sh` 和 `download_depth.sh`:**  用于下载图像和深度数据的 Shell 脚本。
*   **`dataset_to_process.py`:**  Python 脚本，用于指定数据集的路径。
*   **`find_sequence_main.py`:**  Python 脚本，用于生成数据集的序列。该脚本调用 `find_sequence_module.py`和`find_sequences.py`中的模块。
*   **`captioning.py`:**  Python 脚本，用于生成数据集的图像描述。

## 序列提取相关文件说明

*   **`find_sequence_module.py`:** 包含从原始帧数据中提取姿态信息的函数，如 `get_pose_info`。
*   **`find_sequences.py`:** 包含用于计算旋转角度、分割序列、过滤序列以及创建 GIF 动画的函数，例如 `calculate_rotation_angle_degrees`, `segment_sequences_by_jumps`, `filter_sequences_for_forward_motion`, `get_subsampled_indices`, 和 `create_sequence_gifs`。

# 语义分割

本部分介绍如何使用预训练模型对数据集进行语义分割。

## 方法

*   **SegFormer:**
    *   使用 `segformer_ade_matrix.py` 脚本，利用 SegFormer 模型对数据集进行语义分割。
    *   词汇表来自 ADE20K 数据集。
*   **Mask2Former:**
    *   使用 `mask2former_ade_matrix.py` 脚本，利用 Mask2Former 模型对数据集进行语义分割。
    *   词汇表同样来自 ADE20K 数据集。