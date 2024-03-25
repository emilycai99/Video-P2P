import subprocess
from subprocess import STDOUT
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

## baseline
# subprocess.run(['python', 'run_tuning.py', 
#                 '--config', 'configs/rabbit-jump-tune.yaml', 
#                 # '--dependent',
#                 '--num_frames', '8',
#                 '--window_size', '8',
#                 '--decay_rate', '0.1',
#                 '--eta', '0.0',
#                 # '--dependent_weights', '0.01'
#                 ],stderr=STDOUT)

# subprocess.run(['python', 'run_videop2p.py', 
#                 '--config', 'configs/rabbit-jump-p2p.yaml', 
#                 '--fast',
#                 # '--dependent',
#                 # '--dependent_p2p',
#                 '--num_frames', '8',
#                 '--window_size', '8',
#                 '--decay_rate', '0.1',
#                 '--eta', '0.0',
#                 # '--dependent_weights', '0.01'      
#                 ],stderr=STDOUT)

decay_rate_list = [0.1, 0.3, 0.5, 0.7]
eta_list = [0.1, 0.3, 0.5]
dependent_weights_list = [0.01, 0.05, 0.1]

for d in decay_rate_list:
    for e in eta_list:
        for dw in dependent_weights_list:
            subprocess.run(['python', 'run_tuning.py', 
                            '--config', 'configs/car-drive-tune.yaml', 
                            '--dependent',
                            '--num_frames', '8',
                            '--window_size', '8',
                            '--decay_rate', str(d),
                            '--eta', str(e),
                            '--dependent_weights', str(dw),
                            ],stderr=STDOUT)

            subprocess.run(['python', 'run_videop2p.py', 
                            '--config', 'configs/car-drive-p2p.yaml', 
                            '--fast',
                            '--dependent',
                            '--dependent_p2p',
                            '--num_frames', '8',
                            '--window_size', '8',
                            '--decay_rate', str(d),
                            '--eta', str(e),
                            '--dependent_weights', str(dw),    
                            ],stderr=STDOUT)
