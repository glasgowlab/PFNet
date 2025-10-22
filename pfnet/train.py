import lightning as L
import torch
import wandb
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import DataLoader

from pfnet.data import HDXDataset, OnlineHDXDataset, custom_collate_fn
from pfnet.model import HDXModel

if __name__ == "__main__":
    
    import time
    
    start_time = time.time()
    
    config = {
        "batch_size": 64,
        "minibatch_size": 8,
        "epochs_per_chunk": 1,
        "learning_rate": 5e-5,
        'data_start_idx': 0,
        'data_end_idx':20000000,
        'chunk_size': 25000,
        "checkpoint_path": './PFNet/4x9q26nk/checkpoints/epoch=5-val_loss=5.ckpt',
        #"checkpoint_path":None
    }

    
    if config['checkpoint_path'] is not None:
        hdx_model = HDXModel.load_from_checkpoint(checkpoint_path=config['checkpoint_path'], strict=False, weights_only=True)
        hdx_model.learning_rate = config['learning_rate']
    else:
        hdx_model = HDXModel(learning_rate=config['learning_rate'])
    
    hdx_model.to(device='cuda')
    # hdx_model = torch.compile(hdx_model)
    # freeze the weights except for tp_mask_head
    # for name, param in hdx_model.named_parameters():
    #     if 'tp_mask_head' not in name:
    #         param.requires_grad = False

    wandb_logger = WandbLogger(project='PFNet', config=config, save_code=True)

    # num of gpus
    num_gpus = torch.cuda.device_count()

    trainer = L.Trainer(
        max_epochs=config['epochs_per_chunk'],  
        accumulate_grad_batches=config['batch_size']//config['minibatch_size']//num_gpus,
        callbacks=[
            LearningRateMonitor(logging_interval='step'),
            ModelCheckpoint(save_weights_only=False, 
                        filename='{epoch}-{val_loss:.0f}',  
                        monitor='val_loss', 
                        mode='min') 
        ],
        logger=wandb_logger,
        #gradient_clip_val=1.0,  
        num_sanity_val_steps=0,  
        accelerator="gpu",
        devices=-1,
        strategy='ddp_find_unused_parameters_true',
        precision='16-mixed'
    )

    # make chunks for training
    total_epochs = 0
    chunk_size = config['chunk_size']
    
    val_dataset = OnlineHDXDataset(data_start_idx=0, data_end_idx=5000, base_seed=22, generate_miss_id=True)
    val_loader = DataLoader(val_dataset, batch_size=config['minibatch_size'], shuffle=False, collate_fn=custom_collate_fn, num_workers=4,)
    
    laci_dataset = HDXDataset("./exp_data/laci_exp_pfnet_data/", data_start_idx=0, data_end_idx=5)
    laci_loader = DataLoader(laci_dataset, batch_size=config['minibatch_size'], shuffle=False, collate_fn=custom_collate_fn, num_workers=4,)

    for i in range(config['data_start_idx'], config['data_end_idx'], chunk_size):
        
        
        train_dataset = OnlineHDXDataset(data_start_idx=i, data_end_idx=i+chunk_size, base_seed=42, generate_miss_id=True)
        train_loader = DataLoader(train_dataset, batch_size=config['minibatch_size'], shuffle=True, collate_fn=custom_collate_fn, num_workers=4,)

        # update the total epochs
        total_epochs += config["epochs_per_chunk"]
        trainer.fit_loop.max_epochs=total_epochs

        trainer.fit(hdx_model, train_loader, [val_loader, laci_loader])
        
        del train_dataset,train_loader
        torch.cuda.empty_cache()
        
        
    # test_dataset = OnlineHDXDataset(data_start_idx=0, data_end_idx=10000, base_seed=108)
    # test_loader = DataLoader(test_dataset, batch_size=config['minibatch_size'], shuffle=False, collate_fn=custom_collate_fn, num_workers=4)
    # trainer.test(hdx_model, test_loader)
    wandb.finish()

    end_time = time.time()
    print(f"Total time taken: {end_time - start_time:.2f} seconds")