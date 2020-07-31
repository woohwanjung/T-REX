## Requirements and Installation
python3

pytorch>=1.0

```
pip3 install -r requirements.txt
python3 settings.py
```

## preprocessing data
Download metadata from [TsinghuaCloud](https://cloud.tsinghua.edu.cn/d/99e1c0805eb64736af95/) or [GoogleDrive](https://drive.google.com/drive/folders/1Ri3LIILKKBi3aBJjUVCOBpGX5PpONHRK) for baseline method and put them into prepro_data folder.

__
```
python3 gen_data.py
```

## Training and Testing

### Training:
```
CUDA_VISIBLE_DEVICES=0 python3 train.py --model_name T-REX --save_name T-REX
```
--model_name (T-REX|CNN|LSTM|BiLSTM|ContextAware|BERT)

--learning_rate (default: 1e-5)

--train_prefix  (train_dev(default)|train)
  
  train_dev: supervised
  train: weakly (distantly) supervised
    
    

### Testing:
```
CUDA_VISIBLE_DEVICES=0 python3 test.py --model_name T-REX --save_name T-REX
```




