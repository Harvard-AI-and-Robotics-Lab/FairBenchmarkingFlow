conda install -c nvidia cudnn=9.8

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install pillow clu tensorflow==2.15.0 "keras<3" tensorflow_datasets matplotlib==3.9.2
pip install orbax-checkpoint==0.4.4 ml-dtypes==0.5.0 tensorstore==0.1.67
pip install diffusers dm-tree cached_property
pip install huggingface_hub
pip install 'transformers==4.43.4'
pip install "torchmetrics[image]" torchmetrics[multimodal] openai-clip
pip install image-reward
pip install -U xformers --index-url https://download.pytorch.org/whl/cu126


# test with jax and jaxlib 0.6.2, replace `jax.tree_leaves` with `jax.tree.leaves` in code
pip install -U "jax[cuda12]" 
pip install gdown