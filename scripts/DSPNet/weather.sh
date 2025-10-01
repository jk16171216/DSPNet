export CUDA_VISIBLE_DEVICES=1
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
seq_len=96
model_name=DSPNet

root_path_name=./dataset/weather/
data_path_name=weather.csv
model_id_name=weather
data_name=weather

random_seed=2021
#python -u run_longExp.py \
#  --e_layers 2 \
#  --down_sampling_method avg \
#  --down_sampling_layers 3 \
#  --down_sampling_window 2 \
#  --random_seed $random_seed \
#  --is_training 1 \
#  --seq_len $seq_len \
#  --pred_len 96 \
#  --root_path $root_path_name \
#  --data_path $data_path_name \
#  --model_id $model_id_name_$seq_len'_'$pred_len \
#  --model $model_name \
#  --data $data_name \
#  --features M \
#  --enc_in 21 \
#  --c_out 21 \
#  --n_heads 16 \
#  --d_model 128 \
#  --d_ff 256 \
#  --dropout 0.3 \
#  --fc_dropout 0.3 \
#  --head_dropout 0 \
#  --patch_len 16 \
#  --stride 8 \
#  --des 'Exp' \
#  --train_epochs 10 \
#  --patience 20 \
#  --itr 1 --batch_size 64 --learning_rate 0.0003 # >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log

#python -u run_longExp.py \
#  --e_layers 2 \
#  --down_sampling_method avg \
#  --down_sampling_layers 3 \
#  --down_sampling_window 2 \
#  --random_seed $random_seed \
#  --is_training 1 \
#  --seq_len $seq_len \
#  --pred_len 192 \
#  --root_path $root_path_name \
#  --data_path $data_path_name \
#  --model_id $model_id_name_$seq_len'_192' \
#  --model $model_name \
#  --data $data_name \
#  --features M \
#  --enc_in 21 \
#  --c_out 21 \
#  --n_heads 16 \
#  --d_model 128 \
#  --d_ff 256 \
#  --dropout 0.25 \
#  --fc_dropout 0.25 \
#  --head_dropout 0 \
#  --patch_len 16 \
#  --stride 8 \
#  --des 'Exp' \
#  --train_epochs 10 \
#  --patience 20 \
#  --itr 1 --batch_size 128 --learning_rate 0.0003 # >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log

#python -u run_longExp.py \
#  --e_layers 2 \
#  --down_sampling_method avg \
#  --down_sampling_layers 3 \
#  --down_sampling_window 2 \
#  --random_seed $random_seed \
#  --is_training 1 \
#  --seq_len $seq_len \
#  --pred_len 336 \
#  --root_path $root_path_name \
#  --data_path $data_path_name \
#  --model_id $model_id_name_$seq_len'_336' \
#  --model $model_name \
#  --data $data_name \
#  --features M \
#  --enc_in 21 \
#  --c_out 21 \
#  --n_heads 16 \
#  --d_model 128 \
#  --d_ff 256 \
#  --dropout 0.3 \
#  --fc_dropout 0.3 \
#  --head_dropout 0 \
#  --patch_len 16 \
#  --stride 8 \
#  --des 'Exp' \
#  --train_epochs 10 \
#  --patience 20 \
#  --itr 1 --batch_size 128 --learning_rate 0.0003 # >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log

#python -u run_longExp.py \
#  --e_layers 2 \
#  --down_sampling_method avg \
#  --down_sampling_layers 2 \
#  --down_sampling_window 2 \
#  --random_seed $random_seed \
#  --is_training 1 \
#  --seq_len $seq_len \
#  --pred_len 720 \
#  --root_path $root_path_name \
#  --data_path $data_path_name \
#  --model_id $model_id_name_$seq_len'_720' \
#  --model $model_name \
#  --data $data_name \
#  --features M \
#  --enc_in 21 \
#  --c_out 21 \
#  --n_heads 16 \
#  --d_model 128 \
#  --d_ff 256 \
#  --dropout 0.3 \
#  --fc_dropout 0.3 \
#  --head_dropout 0 \
#  --patch_len 16 \
#  --stride 8 \
#  --des 'Exp' \
#  --train_epochs 10 \
#  --patience 20 \
#  --itr 1 --batch_size 256 --learning_rate 0.0003 # >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log