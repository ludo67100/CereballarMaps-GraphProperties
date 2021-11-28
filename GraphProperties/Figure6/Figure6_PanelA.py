import sys
import os
import numpy as np


cwd = os.getcwd()
os.chdir("./common/")

sys.path.append(cwd)
cmd ='python glm_enr.py slope'
os.system(cmd) 

