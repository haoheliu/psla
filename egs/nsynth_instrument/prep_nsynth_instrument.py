import os
import json
import numpy as np

PATH = "/mnt/fast/nobackup/scratch4weeks/hl01486/datasets/nsynth"

def write_json(my_dict, fname):
    # print("Save json file at "+fname)
    json_str = json.dumps(my_dict)
    with open(fname, 'w') as json_file:
        json_file.write(json_str)

def load_json(fname):
    with open(fname,'r') as f:
        data = json.load(f)
        return data

label_set = np.loadtxt('./data/nsynth_instrument_class_labels_indices.csv', delimiter=',', dtype='str')
label_map = {}
for i in range(1, len(label_set)):
    label_map[eval(label_set[i][2])] = label_set[i][0]
print(label_map)

for subfolder in ["test","train","valid"]:
    subpath = os.path.join(PATH, "nsynth-%s" % subfolder)
    metadata = load_json(os.path.join(subpath, "examples.json"))
    filepath = os.path.join(subpath, "audio")
    
    wav_list = []
    
    for k in metadata.keys():
        inst = metadata[k]["instrument_family_str"]
        cur_label = label_map[inst]
        cur_dict = {"wav": os.path.join(filepath, k+".wav"), "labels": '/m/nsyninst'+cur_label.zfill(2)}
        wav_list.append(cur_dict)
    np.random.shuffle(wav_list)
    write_json({'data': wav_list}, "data/datafiles/nsynth_instrument_%s.json" % subfolder)


        
    