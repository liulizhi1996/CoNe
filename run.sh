# Neural
python main.py --dataset_name Neural --walk_length 64 --rwpe_use_weight 0 --hidden_channels 64 --num_layers 2 --lr 0.0005 --lr_decay 0.9 --weight_decay 0.1 --batch_size 256

# Celegans
python main.py --dataset_name Celegans --walk_length 64 --rwpe_use_weight 1 --hidden_channels 64 --num_layers 4 --lr 0.005 --lr_decay 0.8 --weight_decay 0.1 --batch_size 64

# Netscience
python main.py --dataset_name Netscience --walk_length 64 --rwpe_use_weight 0 --hidden_channels 64 --num_layers 3 --lr 0.01 --lr_decay 0.95 --weight_decay 0.02 --batch_size 512

# Pblog
python main.py --dataset_name Pblog --walk_length 64 --rwpe_use_weight 0 --hidden_channels 64 --num_layers 3 --lr 0.01 --lr_decay 0.9 --weight_decay 0.05 --batch_size 256

# UCsocial
python main.py --dataset_name UCsocial --walk_length 64 --rwpe_use_weight 0 --hidden_channels 64 --num_layers 3 --lr 0.05 --lr_decay 0.65 --weight_decay 0.2 --batch_size 256

# Condmat
python main.py --dataset_name Condmat --walk_length 16 --rwpe_use_weight 1 --hidden_channels 16 --num_layers 3 --lr 0.001 --lr_decay 0.95 --weight_decay 0.05 --batch_size 512

# Astro
python main.py --dataset_name Astro --walk_length 32 --rwpe_use_weight 0 --hidden_channels 32 --num_layers 2 --lr 0.001 --lr_decay 0.95 --weight_decay 0.05 --batch_size 512

# Collaboration
python main.py --dataset_name Collaboration --walk_length 64 --rwpe_use_weight 0 --hidden_channels 64 --num_layers 3 --lr 0.0001 --lr_decay 0.95 --weight_decay 0.1 --batch_size 256

# Congress
python main.py --dataset_name Congress --walk_length 32 --rwpe_use_weight 0 --hidden_channels 32 --num_layers 3 --lr 0.0005 --lr_decay 0.99 --weight_decay 0.1 --batch_size 1024

# Usair
python main.py --dataset_name Usair --walk_length 128 --rwpe_use_weight 0 --hidden_channels 128 --num_layers 3 --lr 0.001 --lr_decay 0.95 --weight_decay 0.3 --batch_size 64
