#conda create -n tajima_relighting python=3.6
#conda activate tajima_relighting
conda install pytorch==1.8.1 torchvision==0.9.1 torchaudio==0.8.1 cudatoolkit=10.2 -c pytorch
pip install opencv-python
conda install numpy scikit-image
pip install torch-optimizer tqdm
pip install pyshtools