#!/usr/bin/env python
import argparse 

parser = argparse.ArgumentParser()

parser.add_argument("test_image_set", type=str)
parser.add_argument("ORIGINAL_YAML_PATH", type=str) 
parser.add_argument("DCN_INFERENCE_PATH", type=str)
parser.add_argument("TEST_YAML_DIR", type=str)  
parser.add_argument("FLIP_YAML_DIR", type=str) 
parser.add_argument("SCALES_YAML_DIR", type=str) 

args = parser.parse_args() 

import re 
import os 

ORIGINAL_YAML_PATH = args.ORIGINAL_YAML_PATH
DCN_INFERENCE_PATH = args.DCN_INFERENCE_PATH
TEST_YAML_DIR      = args.TEST_YAML_DIR
FLIP_YAML_DIR      = args.FLIP_YAML_DIR
SCALES_YAML_DIR    = args.SCALES_YAML_DIR

def EditYAML(yaml_file, scale=None, test_set=None, output_path=None, output_file=None): 
    with open(yaml_file, "r") as f: 
        yaml_file_lines = f.readlines() 
    # Change image scale
    if scale is not None:
        index_of_scale_line = yaml_file_lines.index("SCALES:\n")
        yaml_file_lines[index_of_scale_line+1] = "- {}\n".format(scale)
        yaml_file_lines[index_of_scale_line+2] = "- {}\n".format(scale)
    # Change image test set
    if test_set is not None: 
        index_of_test_set = yaml_file_lines.index([_ for _ in yaml_file_lines if re.search("test_image_set:", _)][0])
        yaml_file_lines[index_of_test_set] = "  test_image_set: {}\n".format(test_set) 
    # Generate new YAML file
    if output_file is None:
        yaml_file_dir = yaml_file.split("/")
        if len(yaml_file_dir) == 1: 
            yaml_file_dir = "." 
        else:  
            yaml_file_dir = "/".join(yaml_file_dir[:-1]) 
        new_output_file = yaml_file.split("/")[-1]
        new_output_file = new_output_file.replace(".yaml", "")
        new_output_file = "{}/{}_scale{}.yaml".format(yaml_file_dir, new_output_file, scale) 
    else: 
        new_output_file = output_file 
    with open(new_output_file, "w") as f: 
        for each_line in yaml_file_lines: 
            f.write(each_line) 
    return new_output_file 

# Read in the original test YAML config file 
with open(ORIGINAL_YAML_PATH, "r") as f: 
    yaml_file = [line.strip() for line in f.readlines()] 

test_image_set = args.test_image_set

# Append suffix for flipped images 
flip_test_image_set = "{}_flip".format(test_image_set) 

# Fetch the output path
output_path = [_ for _ in yaml_file if re.search("output_path:", _)][0]
output_path = output_path.replace("output_path: ", "") 
output_path = output_path.replace('"', "") 
output_path = output_path.replace("'", "") 

############################
# PREDICT ON ORIGINAL TEST #
############################

if not os.path.exists(TEST_YAML_DIR): os.makedirs(TEST_YAML_DIR)

TEST_YAML_PATH = EditYAML(ORIGINAL_YAML_PATH, test_set=test_image_set, 
                          output_file=os.path.join(TEST_YAML_DIR, ORIGINAL_YAML_PATH.split("/")[-1]))

# Run inference
os.system("python {} --cfg {} --ignore_cache".format(DCN_INFERENCE_PATH, TEST_YAML_PATH))

# Need to append the name of the YAML config file to the end of
# the output path
yaml_name = ORIGINAL_YAML_PATH.split("/")[-1].split(".")[0] 

# Rename the detections JSON file so it doesn't get overwritten
JSON_PATH = os.path.join(output_path, yaml_name, test_image_set, "results", "detections_{}_results.json".format(test_image_set))
os.system("mv {} {}_original.json".format(JSON_PATH, JSON_PATH.split(".")[0]))

###########################
# PREDICT ON FLIPPED TEST #
###########################

# Assumes that there is a dataset of flipped data with suffix "_flip"
# Edit YAML test_image_set in YAML file

if not os.path.exists(FLIP_YAML_DIR): os.makedirs(FLIP_YAML_DIR)

FLIP_YAML_PATH = EditYAML(ORIGINAL_YAML_PATH, test_set=flip_test_image_set,
                          output_file=os.path.join(FLIP_YAML_DIR, ORIGINAL_YAML_PATH.split("/")[-1])) 

# Run inference
os.system("python {} --cfg {} --ignore_cache".format(DCN_INFERENCE_PATH, FLIP_YAML_PATH))

# Rename the detections JSON file so it doesn't get overwritten
FLIP_JSON_PATH = os.path.join(output_path, yaml_name, flip_test_image_set, "results", "detections_{}_results.json".format(flip_test_image_set))
os.system("mv {} {}_original.json".format(FLIP_JSON_PATH, FLIP_JSON_PATH.split(".")[0]))

###############################
# PREDICT ON DIFFERENT SCALES #
###############################
import numpy as np 
def round_up_to_even(f):
    return int(np.ceil(f / 2.) * 2)

# Get scales from YAML
original_scale = int(yaml_file[yaml_file.index("SCALES:") + 1].replace("- ", ""))
new_scales = [round_up_to_even(original_scale * 0.8), 
              round_up_to_even(original_scale * 1.2)]

SCALES080_YAML_DIR = os.path.join(SCALES_YAML_DIR, "080") 
SCALES120_YAML_DIR = os.path.join(SCALES_YAML_DIR, "120") 

if not os.path.exists(SCALES080_YAML_DIR): os.makedirs(SCALES080_YAML_DIR)
if not os.path.exists(SCALES120_YAML_DIR): os.makedirs(SCALES120_YAML_DIR)

SCALE080_YAML_PATH = EditYAML(ORIGINAL_YAML_PATH, scale=new_scales[0], test_set=test_image_set,
                              output_file=os.path.join(SCALES080_YAML_DIR, ORIGINAL_YAML_PATH.split("/")[-1]))
SCALE120_YAML_PATH = EditYAML(ORIGINAL_YAML_PATH, scale=new_scales[1], test_set=test_image_set,
                              output_file=os.path.join(SCALES120_YAML_DIR, ORIGINAL_YAML_PATH.split("/")[-1]))

# Run inference
os.system("python {} --cfg {} --ignore_cache".format(DCN_INFERENCE_PATH, SCALE080_YAML_PATH))
# Rename file 
os.system("mv {} {}_scale080.json".format(JSON_PATH, JSON_PATH.split(".")[0]))

# Run inference
os.system("python {} --cfg {} --ignore_cache".format(DCN_INFERENCE_PATH, SCALE120_YAML_PATH))
# Rename file 
os.system("mv {} {}_scale120.json".format(JSON_PATH, JSON_PATH.split(".")[0]))

###########################################
# PREDICT ON FLIPPED FOR DIFFERENT SCALES #
###########################################
FLIP_SCALES080_YAML_DIR = os.path.join(SCALES_YAML_DIR, "_flip-080") 
FLIP_SCALES120_YAML_DIR = os.path.join(SCALES_YAML_DIR, "_flip-120") 

if not os.path.exists(FLIP_SCALES080_YAML_DIR): os.makedirs(FLIP_SCALES080_YAML_DIR)
if not os.path.exists(FLIP_SCALES120_YAML_DIR): os.makedirs(FLIP_SCALES120_YAML_DIR)

FLIP_SCALE080_YAML_PATH = EditYAML(ORIGINAL_YAML_PATH, scale=new_scales[0], test_set=flip_test_image_set,
                                   output_file=os.path.join(FLIP_SCALES080_YAML_DIR, ORIGINAL_YAML_PATH.split("/")[-1]))
FLIP_SCALE120_YAML_PATH = EditYAML(ORIGINAL_YAML_PATH, scale=new_scales[1], test_set=flip_test_image_set,
                                   output_file=os.path.join(FLIP_SCALES120_YAML_DIR, ORIGINAL_YAML_PATH.split("/")[-1]))
 
# Run inference
os.system("python {} --cfg {} --ignore_cache".format(DCN_INFERENCE_PATH, FLIP_SCALE080_YAML_PATH))
# Rename file 
os.system("mv {} {}_scale080.json".format(FLIP_JSON_PATH, FLIP_JSON_PATH.split(".")[0]))

# Run inference
os.system("python {} --cfg {} --ignore_cache".format(DCN_INFERENCE_PATH, FLIP_SCALE120_YAML_PATH))
# Rename file 
os.system("mv {} {}_scale120.json".format(FLIP_JSON_PATH, FLIP_JSON_PATH.split(".")[0]))


