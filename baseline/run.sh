CUDA_VISIBLE_DEVICES=1 python train_opensource.py \
    --model_path="../ptlm/t5/" \
    --model_name="t5" \
    --dataset="opensource" \
    --lr=2e-5 \
    --batch_size=12 \
    --max_source_length=512 \
    --max_target_length=200 \
    --epoch=20 \
    --data_dir="../data/" \
    --split_dataset \

# You can change it by your pretrained model path,
# such as the further-pretrained model obtained from the opensource data
PTLM_PATH="../ptlm/t5/"

CUDA_VISIBLE_DEVICES=1 python train_fsl.py \
    --model_path=$PTLM_PATH \
    --model_name="t5" \
    --dataset="instruction" \
    --lr=2e-5 \
    --batch_size=12 \
    --max_source_length=512 \
    --max_target_length=200 \
    --epoch=2 \
    --eval_num=28 \
    --data_dir="../data/" \
    --split_dataset \
