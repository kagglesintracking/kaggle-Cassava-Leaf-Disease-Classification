import pandas as pd
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from utils import *

# logged args
class args:
    image_size = 448
    n_workers = 24
    gpus = [0, 1]
    fold_id = 0
    init_lr = 1e-4 
    batch_size = 32
    n_epochs = 15
    seed = 42
    acc_step = 1

    distributed_backend = 'ddp' #'ddp' #None # 
    sync_batchnorm = True

    # snapmix args
    snap_mix = False
    SNAPMIX_PCT = 0.25 # percentage of snapmix
    SNAPMIX_ALPHA = 5

    # cutmix args
    cut_mix = False
    cutmix_prob = 0.25
    beta = 1 # 1 is used in the paper

    net_params = {'backbone':'nfnet_f1', 'out_dim':5, 'pool_type': 'none', 'pretrained': False}


# unlogged args
class other_args:

    meta = pd.read_csv('../final_train_data.csv')

    working_dir = '../'

    save_path = '../weights/'

    callbacks = [ModelCheckpoint(monitor='valid_acc', 
                                 save_top_k=1, 
                                 filename='{epoch}-{valid_acc:.4f}', 
                                 mode='max', 
                                 verbose=True, 
                                 save_weights_only=True)]

    loss =  SmoothCrossEntropyLoss(smoothing=0.1) # nn.CrossEntropyLoss() # # OUSMLoss(k=2)

    neptune_log = True