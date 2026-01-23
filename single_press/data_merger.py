import os
import glob
import numpy as np
import config

def merge_npy_files(data_dir, timestamp, delete_original=config.DELETE_FRAMES_AFTER_VIDEO):
    """
    将多个npy文件合并为一个大的npy文件，每个文件形状为[17,17,3,3]，合并后为[n,17,17,3,3]
    
    参数:
    - data_dir: 包含npy文件的目录
    - timestamp: 时间戳，用于命名输出文件
    - delete_original: 是否删除原始文件
    """
    try:
        # 获取所有npy文件并按名称排序
        npy_files = sorted(glob.glob(os.path.join(data_dir, "*.npy")))
        
        if not npy_files:
            print("未找到npy文件，无法合并")
            return False

        print(f"找到 {len(npy_files)} 个npy文件")
        
        # 检查所有文件的兼容性（形状和数据类型）
        first_file = np.load(npy_files[0])
        first_shape = first_file.shape
        first_dtype = first_file.dtype
        
        print(f"第一个文件形状: {first_shape}, 数据类型: {first_dtype}")
        
        # 验证所有文件是否兼容
        compatible = True
        file_shapes = []
        
        for i, npy_file in enumerate(npy_files):
            data = np.load(npy_file)
            file_shapes.append(data.shape)
            
            # 检查数据类型是否一致
            if data.dtype != first_dtype:
                print(f"警告: 文件 {os.path.basename(npy_file)} 的数据类型不匹配")
                compatible = False
            
            # 检查形状是否一致（所有文件应该是[17,17,3,3]）
            if data.shape != (17, 17, 3, 3):
                print(f"错误: 文件 {os.path.basename(npy_file)} 的形状不是 [17,17,3,3]")
                compatible = False
                break
        
        if not compatible:
            print("文件不兼容，无法合并")
            return False
        
        # 确定合并后的形状 [n, 17, 17, 3, 3]
        n_files = len(npy_files)
        merged_shape = (n_files,) + first_shape  # (n, 17, 17, 3, 3)
        
        print(f"合并后的形状: {merged_shape}")
        print(f"总文件数: {n_files}")
        
        # 创建合并后的数组
        merged_data = np.empty(merged_shape, dtype=first_dtype)
        
        # 逐个加载并合并数据
        for i, npy_file in enumerate(npy_files):
            data = np.load(npy_file)
            
            # 将数据复制到合并数组的相应位置
            merged_data[i] = data
            
            print(f"已合并文件 {i+1}/{len(npy_files)}: {os.path.basename(npy_file)}")
        
        # 保存合并后的文件
        grid_dir = os.path.dirname(data_dir)
        grid_path = os.path.join(grid_dir, f"grid_{timestamp}.npy")
        np.save(grid_path, merged_data)
        print(f"✓ 数据已成功合并: {grid_path}")
        print(f"文件大小: {os.path.getsize(grid_path) / (1024 * 1024):.2f} MB")
        
        # 可选：删除原始文件
        if delete_original:
            for npy_file in npy_files:
                os.remove(npy_file)
            print(f"已删除 {len(npy_files)} 个原始npy文件")
        
        return True
        
    except Exception as e:
        print(f"✗ 合并npy文件时出错: {e}")
        import traceback
        traceback.print_exc()
        return False

def merge_npy_files_with_pattern(base_dir, pattern, output_filename, 
                                 delete_original=config.DELETE_FRAMES_AFTER_VIDEO):
    """
    使用文件模式匹配来合并npy文件
    
    参数:
    - base_dir: 基础目录
    - pattern: 文件匹配模式（如 "displacement_*.npy"）
    - output_filename: 输出文件名
    - delete_original: 是否删除原始文件
    """
    # 构建完整的搜索模式
    search_pattern = os.path.join(base_dir, pattern)
    npy_files = sorted(glob.glob(search_pattern))
    
    if not npy_files:
        print(f"未找到匹配模式 '{pattern}' 的npy文件")
        return False
    
    print(f"找到 {len(npy_files)} 个匹配的文件")
    return merge_npy_files(base_dir, output_filename, delete_original)

def merge_npy_files_by_timestamp(data_dir, timestamp, 
                                delete_original=config.DELETE_FRAMES_AFTER_VIDEO):
    """
    按时间戳合并npy文件（与视频生成函数类似）
    
    参数:
    - data_dir: 数据目录
    - timestamp: 时间戳，用于命名输出文件
    - delete_original: 是否删除原始文件
    """
    output_filename = os.path.join(data_dir, f"merged_displacement_{timestamp}.npy")
    return merge_npy_files(data_dir, output_filename, delete_original)

# 高级版本：支持不同的合并策略
def merge_npy_files_advanced(data_dir, output_filename, 
                           merge_axis=0,  # 沿哪个轴合并
                           delete_original=config.DELETE_FRAMES_AFTER_VIDEO,
                           compression=False):
    """
    高级npy文件合并函数，支持不同的合并策略
    
    参数:
    - data_dir: 包含npy文件的目录
    - output_filename: 输出文件名
    - merge_axis: 合并的轴（0=行方向，1=列方向等）
    - delete_original: 是否删除原始文件
    - compression: 是否使用压缩格式（npz）
    """
    try:
        npy_files = sorted(glob.glob(os.path.join(data_dir, "*.npy")))
        
        if not npy_files:
            print("未找到npy文件")
            return False
        
        print(f"找到 {len(npy_files)} 个npy文件")
        
        # 加载第一个文件以获取参考信息
        first_data = np.load(npy_files[0])
        first_shape = first_data.shape
        first_dtype = first_data.dtype
        
        # 检查所有文件的兼容性
        compatible = True
        for npy_file in npy_files[1:]:
            data = np.load(npy_file)
            
            # 检查除合并轴外的其他维度是否匹配
            for i in range(len(first_shape)):
                if i != merge_axis and data.shape[i] != first_shape[i]:
                    print(f"形状不匹配: {os.path.basename(npy_file)}")
                    compatible = False
                    break
            
            if not compatible:
                break
        
        if not compatible:
            print("文件形状不兼容，无法合并")
            return False
        
        # 计算合并后的形状
        total_size_along_axis = sum(np.load(f).shape[merge_axis] for f in npy_files)
        
        new_shape = list(first_shape)
        new_shape[merge_axis] = total_size_along_axis
        new_shape = tuple(new_shape)
        
        print(f"合并后的形状: {new_shape}")
        
        # 创建合并数组
        merged_data = np.empty(new_shape, dtype=first_dtype)
        
        # 执行合并
        current_position = 0
        for i, npy_file in enumerate(npy_files):
            data = np.load(npy_file)
            data_length = data.shape[merge_axis]
            
            # 构建切片索引
            slice_obj = [slice(None)] * len(new_shape)
            slice_obj[merge_axis] = slice(current_position, current_position + data_length)
            
            # 将数据复制到合并数组
            merged_data[tuple(slice_obj)] = data
            
            current_position += data_length
            # print(f"已合并文件 {i+1}/{len(npy_files)}")
        
        # 保存结果
        if compression:
            output_filename = output_filename.replace('.npy', '.npz')
            np.savez_compressed(output_filename, data=merged_data)
        else:
            np.save(output_filename, merged_data)
        
        print(f"✓ 数据已成功合并: {output_filename}")
        
        # 删除原始文件
        if delete_original:
            for npy_file in npy_files:
                os.remove(npy_file)
            print(f"已删除 {len(npy_files)} 个原始文件")
        
        return True
        
    except Exception as e:
        print(f"✗ 合并npy文件时出错: {e}")
        import traceback
        traceback.print_exc()
        return False