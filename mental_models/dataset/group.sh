#!/bin/bash

if [ "$#" -ne 2 ]; then
    echo "usage: $0 <source_folder> <target_folder>"
    exit 1
fi

# extract source_folder and target_folder from command line arguments
source_folder="$1"
target_folder="$2"

if [ ! -d "$target_folder" ]; then
    # create the target folder if it doesn't exist
    mkdir -p "$target_folder"
fi

# init counter for creating task names
counter=0
# find all .hdf5 files in the source folder and its subdirectories
find "$source_folder" -type f -name "*.hdf5" -print0 | while IFS= read -r -d '' file; do
    # get the relative path of the file from the source folder
    relative_path="${file#$source_folder/}"
    # remove filename from path
    relative_dir="$(dirname "$relative_path")"
    # replace '/' with '_' in the relative path
    target_file="${relative_dir//\//_}"

    # copy the file to the target folder with the modified name
    cp "$file" "$target_folder/$target_file.hdf5"
    # display a msg for each file copied
    echo "copied file $file to $target_folder/$target_file.hdf5"

    # # can't use symlinks due to this issue: https://github.com/h5py/h5py/issues/860
    # # create symlink in the target folder
    # ln -s "$file" "$target_folder/$target_file.hdf5"
    # # display a msg for each symlink created
    # echo "Created symbolic link $target_folder/$target_file.hdf5 for $file"

    # increment counter for the next task
    ((counter++))
done

echo "finished"
