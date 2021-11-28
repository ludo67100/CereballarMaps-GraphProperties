import sys
import os
import numpy as np


cwd = os.getcwd()
os.chdir("./common/")

sys.path.append(cwd)
cmd ='python glm_lcls.py post_op15'
os.system(cmd) 

