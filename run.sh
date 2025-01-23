export IRDST_DATASETS=''
export MDFA_DATASETS=''
export IRSTD_DATASETS=''
export NUAA_DATASETS=''
export NUDT_DATASETS=''
python train_net.py --resume --num-gpus 8 --config-file configs/repvit.yaml