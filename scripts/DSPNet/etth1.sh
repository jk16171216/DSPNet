export CUDA_VISIBLE_DEVICES=0
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
seq_len=96
model_name=DSPNet

root_path_name=./dataset/ETT-small
data_path_name=ETTh1.csv
model_id_name=ETTh1
data_name=ETTh1

random_seed=2021
for pred_len in 96 192 336 720
do
    python -u run_longExp.py \
      --e_layers 3 \
      --down_sampling_method avg \
      --down_sampling_layers 3 \
      --down_sampling_window 2 \
      --random_seed $random_seed \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name_$seq_len'_'$pred_len \
      --model $model_name \
      --data $data_name \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 7 \
      --c_out 7 \
      --n_heads 4 \
      --d_model 16 \
      --d_ff 128 \
      --dropout 0.3 \
      --fc_dropout 0.3 \
      --head_dropout 0 \
      --patience 10 \
      --patch_len 16 \
      --stride 8 \
      --des 'Exp' \
      --train_epochs 10 \
      --itr 1 --batch_size 128 --learning_rate 0.00055 # >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log
done