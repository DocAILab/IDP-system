python evaluate.py ^
    --model_path ./checkpoint/model_best ^
    --test_path ./data/dev.txt ^
    --device gpu ^
    --batch_size 16 ^
    --max_seq_len 512