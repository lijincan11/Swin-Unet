
# ------------------test-
# python /home/ljc/source/Swin-Unet-main/Swin-Unet/train.py --dataset Synapse --cfg /home/ljc/source/Swin-Unet-main/Swin-Unet/configs/swin_tiny_patch4_window7_224_lite.yaml --root_path /home/ljc/source/Swin-Unet-main/data/Synapse --max_epochs 150 --output_dir /home/ljc/source/Swin-Unet-main/train_Synapse_test  --img_size 224 --base_lr 0.05 --batch_size 4
#----------------------

# python /home/ljc/source/Swin-Unet-main/Swin-Unet/train.py --dataset Synapse --cfg /home/ljc/source/Swin-Unet-main/Swin-Unet/configs/swin_tiny_patch4_window7_224_lite.yaml --root_path /home/ljc/source/Swin-Unet-main/data/Synapse --max_epochs 150 --output_dir /home/ljc/source/Swin-Unet-main/train_Synapse_0122_PM2  --img_size 224 --base_lr 0.05 --batch_size 24

# python Swin-Unet/test.py --dataset Synapse --cfg /home/ljc/source/Swin-Unet-main/Swin-Unet/configs/swin_tiny_patch4_window7_224_lite.yaml --is_saveni --volume_path data/Synapse --output_dir train_Synapse_0122_PM2 --max_epoch 150 --base_lr 0.05 --img_size 224 --batch_size 24

# nohup bash run.sh >train.log 2>&1&

# python /home/ljc/source/Swin-Unet-main/Swin-Unet_Mixer/train.py --dataset Synapse --cfg /home/ljc/source/Swin-Unet-main/Swin-Unet_Mixer/configs/swin_tiny_patch4_window7_224_lite.yaml --root_path /home/ljc/source/Swin-Unet-main/data/Synapse --max_epochs 150 --output_dir /home/ljc/source/Swin-Unet-main/train_Synapse_0124_iFormer  --img_size 224 --base_lr 0.05 --batch_size 24

# python Swin-Unet_Mixer/test.py --dataset Synapse --cfg /home/ljc/source/Swin-Unet-main/Swin-Unet_Mixer/configs/swin_tiny_patch4_window7_224_lite.yaml --is_saveni --volume_path data/Synapse --output_dir train_Synapse_0124_iFormer --max_epoch 150 --base_lr 0.05 --img_size 224 --batch_size 24

# python /home/ljc/source/Swin-Unet-main/Swin-Unet_Mixer/train.py --dataset Synapse --cfg /home/ljc/source/Swin-Unet-main/Swin-Unet_Mixer/configs/swin_tiny_patch4_window7_224_lite.yaml --root_path /home/ljc/source/Swin-Unet-main/data/Synapse --max_epochs 200 --output_dir /home/ljc/source/Swin-Unet-main/train_Synapse_0124_iFormer  --img_size 224 --base_lr 0.05 --batch_size 24

# python Swin-Unet_Mixer/test.py --dataset Synapse --cfg /home/ljc/source/Swin-Unet-main/Swin-Unet_Mixer/configs/swin_tiny_patch4_window7_224_lite.yaml --is_saveni --volume_path data/Synapse --output_dir train_Synapse_0124_iFormer --max_epoch 200 --base_lr 0.05 --img_size 224 --batch_size 24

# python /home/ljc/source/Swin-Unet-main/Swin-Unet/train.py --dataset Synapse --cfg /home/ljc/source/Swin-Unet-main/Swin-Unet/configs/swin_tiny_patch4_window7_224_lite.yaml --root_path /home/ljc/source/Swin-Unet-main/data/Synapse --max_epochs 150 --output_dir /home/ljc/source/Swin-Unet-main/train_Synapse_0124_UFormer  --img_size 224 --base_lr 0.05 --batch_size 24

# python Swin-Unet/test.py --dataset Synapse --cfg /home/ljc/source/Swin-Unet-main/Swin-Unet/configs/swin_tiny_patch4_window7_224_lite.yaml --is_saveni --volume_path data/Synapse --output_dir train_Synapse_0124_UFormer --max_epoch 150 --base_lr 0.05 --img_size 224 --batch_size 24

# python /home/ljc/source/Swin-Unet-main/Swin-Unet/train.py --dataset Synapse --cfg /home/ljc/source/Swin-Unet-main/Swin-Unet/configs/swin_tiny_patch4_window7_224_lite.yaml --root_path /home/ljc/source/Swin-Unet-main/data/Synapse --max_epochs 150 --output_dir /home/ljc/source/Swin-Unet-main/train_Synapse_0128_aug --img_size 224 --base_lr 0.05 --batch_size 24

# python Swin-Unet/test.py --dataset Synapse --cfg /home/ljc/source/Swin-Unet-main/Swin-Unet/configs/swin_tiny_patch4_window7_224_lite.yaml --is_saveni --volume_path data/Synapse --output_dir train_Synapse_0128_aug --max_epoch 150 --base_lr 0.05 --img_size 224 --batch_size 24

# python /home/ljc/source/Swin-Unet-main/Swin-Unet/train.py --dataset Synapse --cfg /home/ljc/source/Swin-Unet-main/Swin-Unet/configs/swin_tiny_patch4_window7_224_lite.yaml --root_path /home/ljc/source/Swin-Unet-main/data/Synapse --max_epochs 120 --output_dir /home/ljc/source/Swin-Unet-main/train_Synapse_0128_aug --img_size 224 --base_lr 0.05 --batch_size 24

# python Swin-Unet/test.py --dataset Synapse --cfg /home/ljc/source/Swin-Unet-main/Swin-Unet/configs/swin_tiny_patch4_window7_224_lite.yaml --is_saveni --volume_path data/Synapse --output_dir train_Synapse_0128_aug --max_epoch 120 --base_lr 0.05 --img_size 224 --batch_size 24


# python /home/ljc/source/Swin-Unet-main/Swin-Unet/train.py --dataset Synapse --cfg /home/ljc/source/Swin-Unet-main/Swin-Unet/configs/swin_tiny_patch4_window7_224_lite.yaml --root_path /home/ljc/source/Swin-Unet-main/data/Synapse --max_epochs 300 --output_dir /home/ljc/source/Swin-Unet-main/train_Synapse_0128_aug --img_size 224 --base_lr 0.05 --batch_size 24

# python Swin-Unet/test.py --dataset Synapse --cfg /home/ljc/source/Swin-Unet-main/Swin-Unet/configs/swin_tiny_patch4_window7_224_lite.yaml --is_saveni --volume_path data/Synapse --output_dir train_Synapse_0128_aug --max_epoch 300 --base_lr 0.05 --img_size 224 --batch_size 24
# python /home/ljc/source/Swin-Unet-main/Swin-Unet/train.py --dataset Synapse --cfg /home/ljc/source/Swin-Unet-main/Swin-Unet/configs/swin_tiny_patch4_window7_224_lite.yaml --root_path /home/ljc/source/Swin-Unet-main/data/Synapse --max_epochs 150 --output_dir /home/ljc/source/Swin-Unet-main/train_Synapse_0131_dropKey --img_size 224 --base_lr 0.05 --batch_size 24

# python Swin-Unet/test.py --dataset Synapse --cfg /home/ljc/source/Swin-Unet-main/Swin-Unet/configs/swin_tiny_patch4_window7_224_lite.yaml --is_saveni --volume_path data/Synapse --output_dir train_Synapse_0131_dropKey --max_epoch 150 --base_lr 0.05 --img_size 224 --batch_size 24



# python Swin-Unet_2/train.py --dataset Synapse --cfg /home/ljc/source/Swin-Unet-main/Swin-Unet/configs/swin_tiny_patch4_window7_224_lite.yaml --root_path /home/ljc/source/Swin-Unet-main/data/Synapse --max_epochs 250 --output_dir /home/ljc/source/Swin-Unet-main/train_Synapse_0205_CswIN_UFormer --img_size 224 --base_lr 0.05 --batch_size 24

# python Swin-Unet_2/test.py --dataset Synapse --cfg /home/ljc/source/Swin-Unet-main/Swin-Unet/configs/swin_tiny_patch4_window7_224_lite.yaml --is_saveni --volume_path data/Synapse --output_dir train_Synapse_0205_CswIN_UFormer --max_epoch 250 --base_lr 0.05 --img_size 224 --batch_size 24

# python /home/ljc/source/Swin-Unet-main/Swin-Unet/train.py --dataset Synapse --cfg /home/ljc/source/Swin-Unet-main/Swin-Unet/configs/swin_tiny_patch4_window7_224_lite.yaml --root_path /home/ljc/source/Swin-Unet-main/data/Synapse --max_epochs 150 --output_dir /home/ljc/source/Swin-Unet-main/train_Synapse_0205_CswIN_UFormer --img_size 224 --base_lr 0.05 --batch_size 24

# python Swin-Unet/test.py --dataset Synapse --cfg /home/ljc/source/Swin-Unet-main/Swin-Unet/configs/swin_tiny_patch4_window7_224_lite.yaml --is_saveni --volume_path data/Synapse --output_dir train_Synapse_0205_CswIN_UFormer --max_epoch 150 --base_lr 0.05 --img_size 224 --batch_size 24

# python Swin-Unet_3/train.py --dataset Synapse --cfg /home/ljc/source/Swin-Unet-main/Swin-Unet/configs/swin_tiny_patch4_window7_224_lite.yaml --root_path /home/ljc/source/Swin-Unet-main/data/Synapse --max_epochs 250 --output_dir /home/ljc/source/Swin-Unet-main/train_Synapse_0205_CswIN_UFormer --img_size 224 --base_lr 0.01 --batch_size 24

# python Swin-Unet_3/test.py --dataset Synapse --cfg /home/ljc/source/Swin-Unet-main/Swin-Unet/configs/swin_tiny_patch4_window7_224_lite.yaml --is_saveni --volume_path data/Synapse --output_dir train_Synapse_0205_CswIN_UFormer --max_epoch 250 --base_lr 0.01 --img_size 224 --batch_size 24

# python /home/ljc/source/Swin-Unet-main/Swin-Unet/train.py --dataset Synapse --cfg /home/ljc/source/Swin-Unet-main/Swin-Unet/configs/swin_tiny_patch4_window7_224_lite.yaml --root_path /home/ljc/source/Swin-Unet-main/data/Synapse --max_epochs 150 --output_dir /home/ljc/source/Swin-Unet-main/train_Synapse_0217_CNN --img_size 224 --base_lr 0.05 --batch_size 24

# python Swin-Unet/test.py --dataset Synapse --cfg /home/ljc/source/Swin-Unet-main/Swin-Unet/configs/swin_tiny_patch4_window7_224_lite.yaml --is_saveni --volume_path data/Synapse --output_dir train_Synapse_0217_CNN --max_epoch 150 --base_lr 0.05 --img_size 224 --batch_size 24

# python /home/ljc/source/Swin-Unet-main/Swin-Unet/train.py --dataset Synapse --cfg /home/ljc/source/Swin-Unet-main/Swin-Unet/configs/swin_tiny_patch4_window7_224_lite.yaml --root_path /home/ljc/source/Swin-Unet-main/data/Synapse --max_epochs 150 --output_dir /home/ljc/source/Swin-Unet-main/train_Synapse_0217_CNN --img_size 224 --base_lr 0.05 --batch_size 24

# python Swin-Unet/test.py --dataset Synapse --cfg /home/ljc/source/Swin-Unet-main/Swin-Unet/configs/swin_tiny_patch4_window7_224_lite.yaml --is_saveni --volume_path data/Synapse --output_dir train_Synapse_0217_CNN --max_epoch 150 --base_lr 0.05 --img_size 224 --batch_size 24


#--
# python /home/ljc/source/Swin-Unet-main/Swin-Unet/train.py --dataset Synapse --cfg /home/ljc/source/Swin-Unet-main/Swin-Unet/configs/swin_tiny_patch4_window7_224_lite.yaml --root_path /home/ljc/source/Swin-Unet-main/data/Synapse --max_epochs 150 --output_dir /home/ljc/source/Swin-Unet-main/train_Synapse_0219 --img_size 224 --base_lr 0.05 --batch_size 24

# python Swin-Unet/test.py --dataset Synapse --cfg /home/ljc/source/Swin-Unet-main/Swin-Unet/configs/swin_tiny_patch4_window7_224_lite.yaml --is_saveni --volume_path data/Synapse --output_dir train_Synapse_0219 --max_epoch 150 --base_lr 0.05 --img_size 224 --batch_size 24
# #--
# python Swin-Unet_2/train.py --dataset Synapse --cfg Swin-Unet/configs/swin_tiny_patch4_window7_224_lite.yaml --root_path /home/ljc/source/Swin-Unet-main/data/Synapse --max_epochs 150 --output_dir /home/ljc/source/Swin-Unet-main/train_Synapse_0219 --img_size 224 --base_lr 0.05 --batch_size 24

# python Swin-Unet_2/test.py --dataset Synapse --cfg /home/ljc/source/Swin-Unet-main/Swin-Unet/configs/swin_tiny_patch4_window7_224_lite.yaml --is_saveni --volume_path data/Synapse --output_dir train_Synapse_0219 --max_epoch 150 --base_lr 0.05 --img_size 224 --batch_size 24
# # #--
# python Swin-Unet_3/train.py --dataset Synapse --cfg /home/ljc/source/Swin-Unet-main/Swin-Unet/configs/swin_tiny_patch4_window7_224_lite.yaml --root_path /home/ljc/source/Swin-Unet-main/data/Synapse --max_epochs 150 --output_dir /home/ljc/source/Swin-Unet-main/train_Synapse_0219 --img_size 224 --base_lr 0.05 --batch_size 24

# python Swin-Unet_3/test.py --dataset Synapse --cfg /home/ljc/source/Swin-Unet-main/Swin-Unet/configs/swin_tiny_patch4_window7_224_lite.yaml --is_saveni --volume_path data/Synapse --output_dir train_Synapse_0219 --max_epoch 150 --base_lr 0.05 --img_size 224 --batch_size 24
# #--
# python Swin-Unet_4/train.py --dataset Synapse --cfg /home/ljc/source/Swin-Unet-main/Swin-Unet/configs/swin_tiny_patch4_window7_224_lite.yaml --root_path /home/ljc/source/Swin-Unet-main/data/Synapse --max_epochs 150 --output_dir /home/ljc/source/Swin-Unet-main/train_Synapse_0219 --img_size 224 --base_lr 0.05 --batch_size 24

# python Swin-Unet_4/test.py --dataset Synapse --cfg /home/ljc/source/Swin-Unet-main/Swin-Unet/configs/swin_tiny_patch4_window7_224_lite.yaml --is_saveni --volume_path data/Synapse --output_dir train_Synapse_0219 --max_epoch 150 --base_lr 0.05 --img_size 224 --batch_size 24


# python /home/ljc/source/Swin-Unet-main/Swin-Unet/train.py --dataset Synapse --cfg /home/ljc/source/Swin-Unet-main/Swin-Unet/configs/swin_tiny_patch4_window7_224_lite.yaml --root_path /home/ljc/source/Swin-Unet-main/data/Synapse --max_epochs 150 --output_dir /home/ljc/source/Swin-Unet-main/train_Synapse_0223_Mixer --img_size 224 --base_lr 0.05 --batch_size 24

# python Swin-Unet/test.py --dataset Synapse --cfg /home/ljc/source/Swin-Unet-main/Swin-Unet/configs/swin_tiny_patch4_window7_224_lite.yaml --is_saveni --volume_path data/Synapse --output_dir train_Synapse_0223_Mixer --max_epoch 150 --base_lr 0.05 --img_size 224 --batch_size 24


# python /home/ljc/source/Swin-Unet-main/Swin-Unet/train.py --dataset Synapse --cfg /home/ljc/source/Swin-Unet-main/Swin-Unet/configs/swin_tiny_patch4_window7_224_lite.yaml --root_path /home/ljc/source/Swin-Unet-main/data/Synapse --max_epochs 200 --output_dir /home/ljc/source/Swin-Unet-main/train_Synapse_0223_Mixer --img_size 224 --base_lr 0.05 --batch_size 24

# python Swin-Unet/test.py --dataset Synapse --cfg /home/ljc/source/Swin-Unet-main/Swin-Unet/configs/swin_tiny_patch4_window7_224_lite.yaml --is_saveni --volume_path data/Synapse --output_dir train_Synapse_0223_Mixer --max_epoch 200 --base_lr 0.05 --img_size 224 --batch_size 24


# python /home/ljc/source/Swin-Unet-main/Swin-Unet/train.py --dataset Synapse --cfg /home/ljc/source/Swin-Unet-main/Swin-Unet/configs/swin_tiny_patch4_window7_224_lite.yaml --root_path /home/ljc/source/Swin-Unet-main/data/Synapse --max_epochs 200 --output_dir /home/ljc/source/Swin-Unet-main/train_Synapse_0228_liu --img_size 224 --base_lr 0.05 --batch_size 24

# python Swin-Unet/test.py --dataset Synapse --cfg /home/ljc/source/Swin-Unet-main/Swin-Unet/configs/swin_tiny_patch4_window7_224_lite.yaml --is_saveni --volume_path data/Synapse --output_dir train_Synapse_0228_liu --max_epoch 200 --base_lr 0.05 --img_size 224 --batch_size 24

# python /home/ljc/source/Swin-Unet-main/Swin-Unet/train.py --dataset Synapse --cfg /home/ljc/source/Swin-Unet-main/Swin-Unet/configs/swin_tiny_patch4_window7_224_lite.yaml --root_path /home/ljc/source/Swin-Unet-main/data/Synapse --max_epochs 200 --output_dir /home/ljc/source/Swin-Unet-main/train_Synapse_0223_Mixer --img_size 224 --base_lr 0.1 --batch_size 24

# python Swin-Unet/test.py --dataset Synapse --cfg /home/ljc/source/Swin-Unet-main/Swin-Unet/configs/swin_tiny_patch4_window7_224_lite.yaml --is_saveni --volume_path data/Synapse --output_dir train_Synapse_0223_Mixer --max_epoch 200 --base_lr 0.1 --img_size 224 --batch_size 24


# python /home/ljc/source/Swin-Unet-main/Swin-Unet_2/train.py --dataset Synapse --cfg /home/ljc/source/Swin-Unet-main/Swin-Unet/configs/swin_tiny_patch4_window7_224_lite.yaml --root_path /home/ljc/source/Swin-Unet-main/data/Synapse --max_epochs 200 --output_dir /home/ljc/source/Swin-Unet-main/train_Synapse_0221_Mixer --img_size 224 --base_lr 0.05 --batch_size 24

# python Swin-Unet_2/test.py --dataset Synapse --cfg /home/ljc/source/Swin-Unet-main/Swin-Unet_2/configs/swin_tiny_patch4_window7_224_lite.yaml --is_saveni --volume_path data/Synapse --output_dir train_Synapse_0221_Mixer --max_epoch 200 --base_lr 0.05 --img_size 224 --batch_size 24

# python Swin-Unet_3/train.py --dataset Synapse --cfg /home/ljc/source/Swin-Unet-main/Swin-Unet_3/configs/swin_tiny_patch4_window7_224_lite.yaml --root_path /home/ljc/source/Swin-Unet-main/data/Synapse --max_epochs 200 --output_dir /home/ljc/source/Swin-Unet-main/train_Synapse_0221_Mixer --img_size 224 --base_lr 0.05 --batch_size 24

# python Swin-Unet_3/test.py --dataset Synapse --cfg /home/ljc/source/Swin-Unet-main/Swin-Unet_3/configs/swin_tiny_patch4_window7_224_lite.yaml --is_saveni --volume_path data/Synapse --output_dir train_Synapse_0221_Mixer --max_epoch 200 --base_lr 0.05 --img_size 224 --batch_size 24

# python Swin-Unet/train.py --dataset ACDC --cfg Swin-Unet/configs/swin_tiny_patch4_window7_224_lite.yaml --root_path data/ACDC --max_epochs 150 --output_dir /home/ljc/source/Swin-Unet-main/train_ACDC_0228_BaseLine --img_size 224 --base_lr 0.0001 --batch_size 24

# python Swin-Unet/test.py --dataset ACDC --cfg Swin-Unet/configs/swin_tiny_patch4_window7_224_lite.yaml --is_saveni --volume_path data/ACDC --output_dir train_ACDC_0228_BaseLine --max_epoch 150 --base_lr 0.0001 --img_size 224 --batch_size 24

python Swin-Unet_2/train.py --dataset ACDC --cfg Swin-Unet/configs/swin_tiny_patch4_window7_224_lite.yaml --root_path data/ACDC --max_epochs 200 --output_dir /home/ljc/source/Swin-Unet-main/train_ACDC_0228_Mixer --img_size 224 --base_lr 0.0001 --batch_size 24

python Swin-Unet_2/test.py --dataset ACDC --cfg Swin-Unet/configs/swin_tiny_patch4_window7_224_lite.yaml --is_saveni --volume_path data/ACDC --output_dir train_ACDC_0228_Mixer --max_epoch 200 --base_lr 0.0001 --img_size 224 --batch_size 24

# python Swin-Unet_3/train.py --dataset ACDC --cfg Swin-Unet/configs/swin_tiny_patch4_window7_224_lite.yaml --root_path data/ACDC --max_epochs 150 --output_dir /home/ljc/source/Swin-Unet-main/train_ACDC_0228_liu --img_size 224 --base_lr 0.05 --batch_size 24

# python Swin-Unet_3/test.py --dataset ACDC --cfg Swin-Unet/configs/swin_tiny_patch4_window7_224_lite.yaml --is_saveni --volume_path data/ACDC --output_dir train_ACDC_0228_liu --max_epoch 150 --base_lr 0.05 --img_size 224 --batch_size 24

