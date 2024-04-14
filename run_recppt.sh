# for ml-1m dataset
MASTER_PORT=$(shuf -n`` 1 -i 10000-65535)
deepspeed --include localhost:0 --master_port ${MASTER_PORT} ./train.py \
    --mode train \
    --action none \
    --model recppt \
    --dataset  ml-1m \
    --hidden_units 768 \
    --seq_length 200 \
    --epoch 300 \
    --batch_size 128 \
    --learning_rate 0.001
