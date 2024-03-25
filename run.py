import subprocess
from subprocess import STDOUT
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"


subprocess.run(['python', 'run_tuning.py', 
                '--config', 'configs/rabbit-jump-tune.yaml', 
                '--dependent',
                '--num_frames', '8',
                '--window_size', '8',
                '--decay_rate', '0.1',
                '--eta', '0.1',
                ],stderr=STDOUT)

# subprocess.run(['python', 'run_videop2p.py', 
#                 '--config', 'configs/rabbit-jump-p2p.yaml', 
#                 '--fast'
#                 ],stderr=STDOUT)
