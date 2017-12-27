# Deep AutoEncoders for Collaborative Filtering
This is largely based on the research project described in the paper <a href="https://arxiv.org/abs/1708.01715">Training Deep AutoEncoders for Collaborative Filtering</a> and its official implementation in <a href="https://github.com/NVIDIA/DeepRecommender">Github</a>.

### The model
The model is based on deep AutoEncoders.

![AutEncoderPic](./AutoEncoder.png)

## Requirements
* Python 3.6
* [Pytorch](http://pytorch.org/)

## Getting Started

### Run unittests first
The code is intended to run on GPU. Last test can take a minute or two.
```
$ python -m unittest test/data_layer_tests.py
$ python -m unittest test/test_model.py
```
### Train the model
In this example, the model will be trained for 12 epochs. In paper we train for 102.
```
python run.py \
--path_to_train_data data/train \
--path_to_eval_data data/test \
--hidden_layers 512,512,1024 \
--non_linearity_type selu \
--batch_size 128 \
--logdir model_save \
--drop_prob 0.8 \
--optimizer momentum \
--lr 0.005 \
--weight_decay 0 \
--aug_step 1 \
--noise_prob 0 \
--num_epochs 2 \
--summary_frequency 1000
```

Note that you can run Tensorboard in parallel
```
$ tensorboard --logdir=model_save
```

### Run inference on the Test set
```
python infer.py \
--path_to_train_data data/train \
--path_to_eval_data data/test \
--hidden_layers 512,512,1024 \
--non_linearity_type selu \
--save_path model_save/model.epoch_11 \
--drop_prob 0.8 \
--predictions_path preds.txt
```

### Compute Test RMSE
```
python compute_RMSE.py --path_to_predictions=preds.txt
```
