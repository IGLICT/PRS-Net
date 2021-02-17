# python train.py --dataroot ./datasets/thing --name thing --tf_log --num_quat 0 --load_pretrain net99_norot_w100_bs32 --weights 100 --batchSize 32

python train.py --dataroot ./datasets/all --name test --tf_log --num_quat 3 --num_plane 3 --batchSize 32 --niter 100 --niter_decay 100