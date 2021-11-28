import sys
import os
import numpy as np


cwd = os.getcwd()
os.chdir("./common/")

sys.path.append(cwd)
cmd ='python glm_enr.py total_distance'
os.system(cmd) 

