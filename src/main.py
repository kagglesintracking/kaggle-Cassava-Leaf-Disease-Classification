from transforms import get_train_transforms, get_valid_transforms
from outliers import outliers_list
from utils import *
from models import get_net
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from conf import *
from dataset import CompetitionDataset
from warnings import filterwarnings; filterwarnings("ignore")
import os
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.metrics.functional import accuracy
from pytorch_lightning.loggers import NeptuneLogger


def get_logger():
    neptune_logger = NeptuneLogger(
        api_key = open('../neptune_api.txt').read().strip(),
        project_name = 'kagglesin/cass-leaf',
        experiment_name= '0213',
        params= dict(vars(args)), 
    )
    return neptune_logger

class CLDData(pl.LightningDataModule):
    def __init__(self, df, fold_id, train_augs, valid_augs, bs, n_workers):
        super().__init__()
        self.df = df
        self.train_augs = train_augs
        self.valid_augs = valid_augs
        self.fold_id = fold_id
        self.bs = bs
        self.n_workers = n_workers
        print(f'Batch size is {bs}.')

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        
        df_train_this = self.df[(self.df['fold'] != self.fold_id)&(self.df['is_outlier'] == False)]
        df_valid_this = self.df[self.df['fold'] == self.fold_id]
        self.dataset_train = CompetitionDataset(df_train_this, 'train', transform= self.train_augs)
        self.dataset_valid = CompetitionDataset(df_valid_this, 'valid', transform= self.valid_augs)

    def train_dataloader(self):
        
        return torch.utils.data.DataLoader(self.dataset_train, batch_size=self.bs, shuffle=True,  num_workers= self.n_workers, drop_last=True)
    
    def val_dataloader(self):
        
        return torch.utils.data.DataLoader(self.dataset_valid, batch_size=self.bs, shuffle=False, num_workers= self.n_workers)

    def test_dataloader(self):
        return None

class CLDLightningModule(pl.LightningModule):
    
    def __init__(self, init_lr, net_params, n_epochs, loss, optimizer = None, scheduler = None):
        super().__init__()

        self.model = get_net(**net_params)
        self.init_lr = init_lr
        self.n_epochs = n_epochs
        self.loss = loss
        if args.snap_mix:
            self.snapmix_criterion = SnapMixLoss()
        
    def forward(self, x):
        x = self.model(x)
        return x

    def configure_optimizers(self):

        optimizer = optim.AdamW(self.model.parameters(), lr= self.init_lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max= self.n_epochs)
        return  (
            [optimizer],
            [{'scheduler': scheduler, 'interval': 'epoch'}],
        )
    
    def training_step(self, batch, batch_idx):
        
        images, targets = batch
        targets = targets.long()

        # snap mix
        if args.snap_mix:
            rand = np.random.rand()
            if rand > (1.0- args.SNAPMIX_PCT):
                X, ya, yb, lam_a, lam_b = snapmix(images, targets, args.SNAPMIX_ALPHA, args.image_size, self.model)
                logits, _ = self.model(X)
                loss = self.snapmix_criterion(self.loss, logits, ya, yb, lam_a, lam_b)
            else:
                logits, _ = self.model(images)
                loss = torch.mean(self.loss(logits, targets))
            return {'loss': loss}

        # cutmix
        if args.cut_mix:
            rand = np.random.rand()
            if rand < args.cutmix_prob:
                lam = np.random.beta(args.beta, args.beta)
                rand_index = torch.randperm(images.size()[0])
                target_a = targets
                target_b = targets[rand_index]
                bbx1, bby1, bbx2, bby2 = rand_bbox(images.size(), lam)
                images[:, :, bbx1:bbx2, bby1:bby2] = images[rand_index, :, bbx1:bbx2, bby1:bby2]
                # adjust lambda to exactly match pixel ratio
                lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (images.size()[-1] * images.size()[-2]))
                # compute output
                logits = self.model(images)
                loss = self.loss(logits, target_a) * lam + self.loss(logits, target_b) * (1. - lam)
            else:
                logits = self.model(images)
                loss = torch.mean(self.loss(logits, targets))
            return {'loss': loss}

        else:
            logits = self.model(images)
            loss = self.loss(logits, targets)
            score = accuracy(logits.argmax(1), targets)
            logs = {'train_loss': loss, f'train_accuracy': score}
            return {'loss': loss, 'logits': logits, 'targets': targets, f'train_accuracy': score, 'progress_bar': logs}
    
    def training_epoch_end(self, outputs):
        
        if args.snap_mix or args.cut_mix:
            avg_loss = torch.stack([x['loss'] for x in outputs]).mean()

            if other_args.neptune_log:
                self.logger.experiment.log_metric('train_loss', avg_loss)

        else:
            avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
            y_true = torch.cat([x['targets'] for x in outputs])
            y_pred = torch.cat([x['logits'] for x in outputs])
            score = accuracy(y_pred.argmax(1), y_true)

            if other_args.neptune_log:
                self.logger.experiment.log_metric('train_acc', score)
                self.logger.experiment.log_metric('train_loss', avg_loss)

            return {'train_loss': avg_loss, 'train_acc': score}

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        if args.snap_mix:
            logits, _ = self.model(images)
        else:
            logits = self.model(images)

        loss = self.loss(logits, targets.long())
        score = accuracy(logits.argmax(1), targets)
        logs = {'valid_loss': loss, f'valid_accuracy': score}

        return {'loss': loss, 'logits': logits, 'targets': targets, f'valid_accuracy': score}
    
    def validation_epoch_end(self, outputs):

        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        y_true = torch.cat([x['targets'] for x in outputs])
        y_pred = torch.cat([x['logits'] for x in outputs])
        score = accuracy(y_pred.argmax(1), y_true)

        if other_args.neptune_log:
            self.logger.experiment.log_metric('valid_acc', score)
            self.logger.experiment.log_metric('valid_loss', avg_loss)

        return {'valid_loss': avg_loss, 'valid_acc': score}

if __name__ == '__main__':

    if other_args.neptune_log:
        # get logger
        logger = get_logger()

    # set seed 
    seed_everything(args.seed) 

    # initialize lightning data module
    dm = CLDData(df = other_args.meta , 
                 fold_id = args.fold_id, 
                 train_augs=get_train_transforms(args.image_size), 
                 valid_augs=get_valid_transforms(args.image_size),
                 bs = args.batch_size // len(args.gpus), 
                 n_workers = args.n_workers) 

    # initialize lightning module
    lm = CLDLightningModule(init_lr = args.init_lr, 
                            net_params = args.net_params, 
                            n_epochs = args.n_epochs,
                            loss = other_args.loss) 

    # initialize trainer
    trainer = pl.Trainer(
        num_sanity_val_steps = 0,
        weights_summary = 'top',
        precision = 16, 
        gradient_clip_val = 5, 
        logger = logger if other_args.neptune_log else None, 
        deterministic = True,
        benchmark = False,
        gpus = args.gpus,
        distributed_backend = args.distributed_backend,
        max_epochs = args.n_epochs,
        sync_batchnorm= args.sync_batchnorm,
        callbacks = other_args.callbacks,
        weights_save_path = other_args.save_path,
        accumulate_grad_batches= args.acc_step, 
    )

    # start training
    trainer.fit(lm, dm)
