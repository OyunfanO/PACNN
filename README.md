# PACNN

requirements:
python == 3.6.9
pytorch == 1.7.1
scipy == 1.1.0

The network architecture is defined in file model/PACNN11.py

The implementation of loss function is in folder loss

The pre-trained denoising models for gaussian noise with sigma =50 and SIDD validation dataset can be downloaded at https://gocuhk-my.sharepoint.com/:f:/g/personal/fanjia_cuhk_edu_hk/EsQzDn3_rxFGthDNgC-_DCQBqWRdnBkkDWGAUWV-jFfcvA?e=1gkT5B.


After you download the PACNN_C50 model to folder experiment, you can test the model with commond:
python3 main.py --model PACNN11 --save PACNN11_C50 --scale 50 --n_feats 128 --save_results --print_model --n_colors 3 --save_models --task_type denoising --resume -1 --pre_train experiment/PACNN_C50/ --data_test BSD68 --test_only  --img_ext .png

