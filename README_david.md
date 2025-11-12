1. System-wide Versions
- Python 3.6.15
- Cuda compilation tools, release 11.8, V11.8.89
- python -m pip install --upgrade "pip<22" "setuptools<60" "wheel<0.38"

2. Python Module Versions
- You'll need to install rawpy following the instruction from the ELD repo
- For others, you can install from requirements.txt

3. In order to run inference. You'll need to place the SID dataset into ExposureDiffusion/datasets/SID/Sony and then run bash run_infaerence.sh