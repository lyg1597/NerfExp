from pathlib import Path, PosixPath
from typing import List, Literal, Optional, OrderedDict, Tuple, Union
import os 
import pandas as pd 

ALLOWED_RAW_EXTS = ['.cr2']

def list_images(data: Path, recursive: bool = True) -> List[Path]:
    """Lists all supported images in a directory

    Args:
        data: Path to the directory of images.
        recursive: Whether to search check nested folders in `data`.
    Returns:
        Paths to images contained in the directory
    """
    allowed_exts = [".jpg", ".jpeg", ".png", ".tif", ".tiff"] + ALLOWED_RAW_EXTS
    glob_str = "**/[!.]*" if recursive else "[!.]*"
    image_paths = sorted([p for p in data.glob(glob_str) if p.suffix.lower() in allowed_exts])
    return image_paths

script_dir = os.path.dirname(os.path.realpath(__file__))
input_dir = PosixPath(os.path.join(script_dir, '../gazebo3/images/'))
output_dir = PosixPath(os.path.join(script_dir, '../data/gazebo3/images'))

input_fns = list_images(input_dir)
output_fns = list_images(output_dir)

messages = []
for i in range(len(input_fns)):
    msg_dict = {}
    msg_dict['input_fns'] = input_fns[i]
    msg_dict['output_fns'] = output_fns[i]
    messages.append(msg_dict)

df = pd.DataFrame(messages)
csv_file = os.path.join(script_dir, 'matches.csv')
df.to_csv(csv_file, index=False)