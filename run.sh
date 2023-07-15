# python gpt2.py --log-interval 5 --switch 190 --epochs 200 --decay-epoch 150 > output_decay.log 
# python gpt2.py --log-interval 5 --switch 190 --epochs 200 --decay-epoch 150 --lr 0.005 > output_decay_3.log  
# python gpt2_language.py --log-interval 5 --switch 390 --epochs 400 --decay-epoch 300 --depth 12 --embd 384 > output_decay_language_d12.log 
python gpt2_language.py --log-interval 5 --switch 390 --epochs 400 --decay-epoch 300 --depth 24 --embd 384 > output_decay_language_d24.log 
python gpt2_language.py --log-interval 5 --switch 390 --epochs 400 --decay-epoch 300 --depth 48 --embd 384 > output_decay_language_d48.log
python gpt2_language.py --log-interval 5 --switch 390 --epochs 400 --decay-epoch 300 --depth 96 --embd 384 > output_decay_language_d96.log
