# HS2S
Pytorch code for [Hybrid Sequence to Sequence Model for Video Object Segmentation]() <br />
For more information, please visit our [project page]() <br />
Dependencie:
```
numpy==1.18.1,
sacred==0.8.1,
torch==1.5,
torchvision==0.6,
tqdm
```
For training on Youtube-VOS, download the dataset from [here](https://competitions.codalab.org/competitions/19544#participate-get_data) and modify the data paths in train.py.
Finally, run the following command.
```
python train.py
```
For inference with the pretrained model, download the weights from [here](https://drive.google.com/file/d/1qnB-BJJOCUwdRogrD5LNF_oMzeQ9Fdto/view?usp=sharing) and put them under the Model directory.

Modify the data paths in submit_ytvos.py and run the following command.

```
python submit_ytvos.py with model_name='weights_HS2S.pth'
```
In case of questions, please contact fatemeh.azimi@dfki.de.


