#!/bin/bash
cd ~/Collab-Overcooked/src 
python_dir=$(which python)

/usr/bin/env ${python_dir}  -- evaluation.py --test_mode build_in
/usr/bin/env ${python_dir}  -- organize_result.py
/usr/bin/env ${python_dir}  -- convert_result.py 
