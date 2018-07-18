#!/bin/bash

# activate virtual env
source activate cdms2
echo '============ cdms2 environment activated ============'

python run_CLUS_tool.py input_CLUStool.in

source deactivate
echo '============ cdms2 environment deactivated ============'

tail log*.log
