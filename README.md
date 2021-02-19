# kaggle Cassava Leaf Disease Classification 25th place Solution

## TLDR
1. OAV(online adversarial validation) -> conclusion: train test are split at random
2. Denoiser trained with `no margin cosine loss` -> clean training set and maintain validation set
3. trained 13 models blend/stack with CV 0.908 and 0.910, respectively -> both private LB 0.899 (excuse me?)
4. post processing: class4 (healthy) * 0.9 -> private LB 0.899 -> 0.901 
5. Quite 'interesting' competition that CV doesn't matter LOL

## Diagram
<img src='https://github.com/kagglesintracking/kaggle-Cassava-Leaf-Disease-Classification/blob/main/images/diagram.png'>


## Directories
 ┣ 📂configs  
 ┃ ┗ 📜config1.py  
 ┗ 📂src  
 ┃ ┣ 📜conf.py  
 ┃ ┣ 📜dataset.py  
 ┃ ┣ 📜main.py  
 ┃ ┣ 📜models.py  
 ┃ ┣ 📜outliers.py  
 ┃ ┣ 📜transforms.py  
 ┃ ┗ 📜utils.py    

## To train a model 
For example, if the config file is `config1.py`, do
```
cd src
python train.py --config config1
```
## Dependencies
```
pip install -r requirements.txt
```
