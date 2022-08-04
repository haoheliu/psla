import os

PATH = "/media/Disk_HDD/haoheliu/projects/psla/egs/audioset/exp"

for folder in os.listdir(PATH):
    if(not os.path.exists(os.path.join(PATH, folder, "stats_50.pickle"))):
        cmd = "rm -rf %s" % os.path.join(PATH, folder)
        print(cmd)
        os.system(cmd)