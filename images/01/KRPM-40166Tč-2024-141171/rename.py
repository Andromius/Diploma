import os

def add_suffix_to_all_files_in_folder(folder_path, suffix):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            directory, name = os.path.split(file_path)
            name, ext = os.path.splitext(name)
            new_filename = f"{name}{suffix}{ext}"
            new_file_path = os.path.join(directory, new_filename)
            os.rename(file_path, new_file_path)

# Example usage
add_suffix_to_all_files_in_folder(".", "_01")