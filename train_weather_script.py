import os
os.system("python train_hfrm.py")
os.system("CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m  torch.distributed.launch --nproc_per_node=8 train_diffusion.py --config raindrop_wavelet.yml --test_set raindrop_wavelet  --world_size=8")