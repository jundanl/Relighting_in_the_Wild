#conda create --yes -n tajima_relighting python=3.8
#conda activate tajima_relighting
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
pip install numpy scikit-image
pip install torch-optimizer tqdm
pip install pyshtools==4.7.1
pip install opencv-python==4.5.5.64
pip install gdown
mkdir trained_models
cd trained_models
gdown --fuzzy https://drive.google.com/file/d/1bHYQ3gtLvcW2epqh7H_if9XejkfRtR_8/view?usp=drive_link
gdown --fuzzy https://drive.google.com/file/d/17Kd3shagrtvPCmm1kgYX-JeKgQhR7skp/view?usp=drive_link