import os
import glob
import numpy as np

import config


def merge_npy_files(
    data_dir,
    timestamp,
    delete_original=config.DELETE_FRAMES_AFTER_VIDEO,
):
    """
    Merge multiple NumPy (.npy) files into a single array.

    Each input file is expected to have the same shape, e.g.:
        (H, W, C, D)

    The merged output will have shape:
        (N, H, W, C, D)

    Parameters
    ----------
    data_dir : str
        Directory containing .npy files to merge.
    timestamp : str
        Timestamp identifier used to name the output file.
    delete_original : bool
        Whether to delete original .npy files after merging.

    Returns
    -------
    bool
        True if merge succeeds, False otherwise.
    """
    try:
        npy_files = sorted(glob.glob(os.path.join(data_dir, "*.npy")))

        if not npy_files:
            print("No .npy files found. Merge aborted.")
            return False

        print(f"Found {len(npy_files)} .npy files")

        # Load reference file
        first_data = np.load(npy_files[0])
        ref_shape = first_data.shape
        ref_dtype = first_data.dtype

        print(f"Reference shape : {ref_shape}")
        print(f"Reference dtype : {ref_dtype}")

        # Validate compatibility
        for path in npy_files:
            data = np.load(path)
            if data.shape != ref_shape:
                print(f"Shape mismatch: {os.path.basename(path)}")
                return False
            if data.dtype != ref_dtype:
                print(f"Dtype mismatch: {os.path.basename(path)}")
                return False

        # Allocate merged array
        merged_shape = (len(npy_files),) + ref_shape
        merged_data = np.empty(merged_shape, dtype=ref_dtype)

        # Merge
        for i, path in enumerate(npy_files):
            merged_data[i] = np.load(path)
            print(f"Merged [{i + 1}/{len(npy_files)}]: {os.path.basename(path)}")

        # Save output
        output_dir = os.path.dirname(data_dir)
        output_path = os.path.join(output_dir, f"grid_{timestamp}.npy")
        np.save(output_path, merged_data)

        print(f"✓ Merged data saved to: {output_path}")
        print(f"File size: {os.path.getsize(output_path) / 1024**2:.2f} MB")

        # Cleanup
        if delete_original:
            for path in npy_files:
                os.remove(path)
            print(f"Deleted {len(npy_files)} original files")

        return True

    except Exception as e:
        print(f"✗ Failed to merge .npy files: {e}")
        import traceback
        traceback.print_exc()
        return False


def merge_npy_files_advanced(
    data_dir,
    output_path,
    merge_axis=0,
    delete_original=config.DELETE_FRAMES_AFTER_VIDEO,
    compression=False,
):
    """
    Advanced NumPy file merging with configurable merge axis.

    Parameters
    ----------
    data_dir : str
        Directory containing .npy files.
    output_path : str
        Output file path (.npy or .npz).
    merge_axis : int
        Axis along which arrays are concatenated.
    delete_original : bool
        Whether to delete original files after merging.
    compression : bool
        If True, save as compressed .npz file.

    Returns
    -------
    bool
        True if merge succeeds, False otherwise.
    """
    try:
        npy_files = sorted(glob.glob(os.path.join(data_dir, "*.npy")))

        if not npy_files:
            print("No .npy files found.")
            return False

        first_data = np.load(npy_files[0])
        ref_shape = first_data.shape
        ref_dtype = first_data.dtype

        # Validate shapes (except merge axis)
        for path in npy_files[1:]:
            data = np.load(path)
            for dim in range(len(ref_shape)):
                if dim != merge_axis and data.shape[dim] != ref_shape[dim]:
                    print(f"Shape mismatch: {os.path.basename(path)}")
                    return False

        # Compute merged shape
        total_size = sum(np.load(p).shape[merge_axis] for p in npy_files)
        new_shape = list(ref_shape)
        new_shape[merge_axis] = total_size
        new_shape = tuple(new_shape)

        merged_data = np.empty(new_shape, dtype=ref_dtype)

        # Perform merge
        cursor = 0
        for path in npy_files:
            data = np.load(path)
            length = data.shape[merge_axis]

            slices = [slice(None)] * len(new_shape)
            slices[merge_axis] = slice(cursor, cursor + length)
            merged_data[tuple(slices)] = data

            cursor += length

        # Save
        if compression:
            np.savez_compressed(output_path, data=merged_data)
        else:
            np.save(output_path, merged_data)

        print(f"✓ Advanced merge saved to: {output_path}")

        if delete_original:
            for path in npy_files:
                os.remove(path)
            print(f"Deleted {len(npy_files)} original files")

        return True

    except Exception as e:
        print(f"✗ Advanced merge failed: {e}")
        import traceback
        traceback.print_exc()
        return False
