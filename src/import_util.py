import os
import sys

cur_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
gmflow_dir = os.path.join(cur_dir, 'deps/gmflow')
controlnet_dir = os.path.join(cur_dir, 'deps/ControlNet')
sys.path.insert(0, gmflow_dir)
sys.path.insert(0, controlnet_dir)

import deps.ControlNet.share  # noqa: F401 E402
