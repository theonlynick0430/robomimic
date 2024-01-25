# Mental Models

## Installation

Clone Robosuite repo, checkout the v1.4.1 branch, and install from source. See docs [here](https://robosuite.ai/docs/installation.html).  
Clone this repo and install requirements:
```
git clone https://github.com/theonlynick0430/mental-models
pip install -r requirements.txt
```
Install Robomimic from source using forked [repo](https://github.com/theonlynick0430/robomimic):
```
cd <PATH_TO_YOUR_INSTALL_DIRECTORY>
git clone https://github.com/theonlynick0430/robomimic.git
cd robomimic
pip install -e .
```

## Dataset 

Datasets are structured in the same raw format as [Robomimic](https://robomimic.github.io/docs/index.html). 

### Create you own dataset 

To create you own dataset, you can either use the `collect_human_demonstrations.py` script from Robosuite to manually collect demos or mimc the `sim/stack.py` script to automatically collect demos using a scripted policy. Both output datasets in HDF5 format. 

### Download existing human-collected datasets

From the Robomimic 2021 CoRL paper (good for initial experiments):
```
cd robomimic
python robomimic/scripts/download_datasets.py --dataset_types ph --tasks sim --hdf5_types raw --download_dir [download directory]
```
From the [MimicGen 2023 CoRL paper](https://mimicgen.github.io/) (for large variety of experiments):
```
cd <PATH_TO_YOUR_INSTALL_DIRECTORY>
git clone https://github.com/NVlabs/mimicgen_environments.git
cd mimicgen_environments
python mimicgen_envs/scripts/download_datasets.py --dataset_type core --tasks [list of tasks] --download_dir [download directory]
```

### Loading data

First, move all downloaded dataset files into a single target directory:
```
cd mental-models
chmod +x dataset/group.sh
./dataset/group.sh [download directory] [target directory]
```

The raw HDF5 files do not contain any observations by default. Instead, they allow you to choose observation modalities by "replaying" the simulation via Robosuite. Extract observations as follows: 
```
cd robomimic 
python robomimic/scripts/extract_obs.py --data_dir [target directory] --camera_names [list of camera views, ex: agentview] --camera_height [H] --camera_width [W] --depth --done_mode 2 --exclude-next-obs
```
Now, the datasets are ready for use. As a generic example of how to use them in Pytorch datasets, see `datasets/robomimic.ipynb`. 