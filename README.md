## setup
conda create - NeRF python=3.9
conda activate NeRF
pip install -r requirements.txt
conda install pytorch=1.13.0 torchvision pytorch-cuda=11.6 -c pytorch -c nvidia
git+https://github.com/facebookresearch/pytorch3d.git

## ToDo
Build nerf on custom 6D pose dataset - zB Tless, etc