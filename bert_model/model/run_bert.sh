

fold=5
for((i=0;i<$fold;i++));
do
python ./run_bert.py \
--model_type bert \
--model_name_or_path ../model/bert  \
--do_train \
--do_eval \
--do_test \
--data_dir ../data/data_StratifiedKFold_42/data_origin_$i \
--output_dir ./model/finallmodel/bert/bert_$i \
--max_seq_length 512 \
--split_num 1 \
--lstm_hidden_size 512 \
--lstm_layers 1 \
--lstm_dropout 0.1 \
--eval_steps 200 \
--per_gpu_train_batch_size 4 \
--gradient_accumulation_steps 4 \
--warmup_steps 0 \
--per_gpu_eval_batch_size 64 \
--learning_rate 1e-5 \
--adam_epsilon 1e-6 \
--weight_decay 0 \
--train_steps 30000 \
--freeze 0 \
--op AdamW
done






echo "combine result"
python ./combine.py --model_prefix ./model/finallmodel/bert/bert_ --out_path ./model/finallmodel/bert/bert.csv --fold $fold
echo "done"