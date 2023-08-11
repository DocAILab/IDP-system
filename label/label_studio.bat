python ./label_studio.py ^
    --label_studio_file ./data/label_studio.json ^
    --save_dir ./data ^
    --splits 0.76 0.24 0 ^
    --negative_ratio 3 ^
    --task_type ext