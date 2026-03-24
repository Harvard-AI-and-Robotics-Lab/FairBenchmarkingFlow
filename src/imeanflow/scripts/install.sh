conda install -c nvidia cudnn=9.8

pip install pillow clu tensorflow==2.15.0 "keras<3" tensorflow_datasets matplotlib==3.9.2
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install orbax-checkpoint==0.11.32 ml-dtypes==0.5.0 tensorstore==0.1.67
pip install diffusers dm-tree cached_property
pip install huggingface_hub
pip install 'transformers==4.43.4'
pip install "torchmetrics[image]"
pip install torchmetrics[multimodal]
pip install image-reward
pip install openai-clip
pip install -U xformers --index-url https://download.pytorch.org/whl/cu126
pip install -U "jax[cuda12]" 
pip install gdown
pip install nvidia-nvjitlink-cu12
