import os

PATH = "/mnt/fast/nobackup/scratch4weeks/hl01486/project/psla/egs/audioset/exp"

for folder in os.listdir(PATH):
    if(not os.path.exists(os.path.join(PATH, folder, "stats_40.pickle"))):
        cmd = "rm -rf %s" % os.path.join(PATH, folder)
        print(cmd)
        os.system(cmd)