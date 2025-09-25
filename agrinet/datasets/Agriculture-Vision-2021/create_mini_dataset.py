import os
import shutil


def create_mini_dataset(src_root, dst_root, n_per_folder=100, extensions=None):
    """
    Create a mini dataset by copying up to n_per_folder files
    from each folder in src_root to dst_root.


    Parameters:
    - src_root: path to the original dataset root
    - dst_root: path to the smaller dataset root
    - n_per_folder: number of files to copy per folder
    - extensions: list of file extensions to include (e.g., ['.jpg','.png'])
    or None for all
    """
    src_root = os.path.abspath(src_root)
    dst_root = os.path.abspath(dst_root)

    for dirpath, _, filenames in os.walk(src_root):
        # Sort file names
        filenames.sort()

        # Filter by extensions if needed
        if extensions:
            filenames = [
                f for f in filenames
                if os.path.splitext(f)[1].lower() in extensions
            ]

        # Pick first N items
        selected_files = filenames[:n_per_folder]

        # Compute corresponding destination folder path
        rel_path = os.path.relpath(dirpath, src_root)
        dst_dir = os.path.join(dst_root, rel_path)
        os.makedirs(dst_dir, exist_ok=True)

        # Copy files
        for fname in selected_files:
            src_file = os.path.join(dirpath, fname)
            dst_file = os.path.join(dst_dir, fname)
            shutil.copy2(src_file, dst_file)

        print(f"Copied {len(selected_files)} files from {rel_path}")


if __name__ == "__main__":
    SRC_FOLDER = "_dataset"
    DST_FOLDER = "_dataset_mini"
    N_ITEMS = 100

    create_mini_dataset(SRC_FOLDER, DST_FOLDER, n_per_folder=N_ITEMS)
