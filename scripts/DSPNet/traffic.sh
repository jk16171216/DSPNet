export CUDA_VISIBLE_DEVICES=0
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
seq_len=96
model_name=DSPNet

root_path_name=./dataset/traffic/
data_path_name=traffic.csv
model_id_name=traffic
data_name=traffic

random_seed=2021
for pred_len in 720
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
      --enc_in 862 \
      --c_out 862 \
      --n_heads 16 \
      --d_model 128 \
      --d_ff 256 \
      --dropout 0.3 \
      --fc_dropout 0.3 \
      --head_dropout 0 \
      --patch_len 16 \
      --stride 8 \
      --des 'Exp' \
      --train_epochs 10 \
      --patience 5 \
      --lradj 'TST' \
      --pct_start 0.2 \
      --itr 1 --batch_size 8 --learning_rate 0.001 # >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log
done