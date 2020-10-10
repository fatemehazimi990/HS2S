# HS2S
Pytorch code for [Hybrid Sequence to Sequence Model for Video Object Segmentation]() <br />
Dependencie:
```
numpy==1.18.1,
sacred==0.8.1,
torch==1.5,
torchvision==0.6,
tqdm
```
For training, download the Youtube-VOS from [here](https://competitions.codalab.org/competitions/19544#participate-get_data).
Modify the paths in train.py and run the following command.
```
python train.py
```
For inference with the pretrained model, download the weights from [here]() and put them under the Model directory.

Modify the data paths in submit_ytvos.py and run the following command.

```
python submit_ytvos.py with model_name='weights_HS2S.pth'
```
In case of questions, please contact fatemeh.azimi@dfki.de.


#'/netscratch/azimi/code_dir/seq2seq_vos/sacred/training/S2S_offlin_training/pytorchseq2seq2018/sacred/icpr_resnet_noInit/V1/withRefEncoder/PlusMatchPrevious/LightV1_encoder50/skipmemnocrop/exp5/420.5samestart/'
