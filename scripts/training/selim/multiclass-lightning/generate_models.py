import os
from pytorch_lightning.loggers import TensorBoardLogger
import pytorch_lightning as pl

from classes import *
from utils import tagname_to_id

def get_loaders (train_dataset,
                val_dataset,
                train_params, 
                val_params,
                tagname_to_tagid, 
                tokenizer,
                max_len:int):
                
    training_set = CustomDataset(train_dataset, tagname_to_tagid, tokenizer, max_len)
    val_set = CustomDataset(val_dataset, tagname_to_tagid, tokenizer, max_len)
    
    training_loader = DataLoader(training_set, **train_params)
    val_loader = DataLoader(val_set, **val_params)
    return training_loader, val_loader

def train_on_specific_targets (train_dataset,
                                val_dataset,
                                name_classifier:str,
                                dirpath:str,
                                MODEL_NAME:str,
                                tokenizer,
                                early_stopping_callback,
                                checkpoint_callback,
                                dropout_rate:float,
                               train_params,
                               val_params,
                                gpu_nb:int,
                               MAX_EPOCHS:int,
                               weight_decay=0.01,
                               warmup_steps=500,
                               output_length=384,
                               max_len=128,
                               weight_classes=None,
                               learning_rate=3e-5,
                               pred_threshold:float=0.5):
    """
    main function used to train model
    """
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)

    train_dataset = train_dataset[['entry_id', 'excerpt', 'target']]
    val_dataset = val_dataset[['entry_id', 'excerpt', 'target']]

    logger = TensorBoardLogger("lightning_logs", name=name_classifier)

    tagname_to_tagid = tagname_to_id (train_dataset["target"])

    empty_dataset = CustomDataset(None, tagname_to_tagid, tokenizer, max_len)
    
    training_loader, val_loader = get_loaders (train_dataset,
                                               val_dataset,
                                               train_params,
                                               val_params,
                                               tagname_to_tagid,
                                               tokenizer,
                                               max_len
                                               )
    

    trainer = pl.Trainer(
        logger=logger,
        callbacks=[early_stopping_callback, checkpoint_callback],
        progress_bar_refresh_rate=30,
        profiler="simple",
        log_gpu_memory=True,
        weights_summary=None,
        gpus=gpu_nb,
        accumulate_grad_batches=1,
        max_epochs=MAX_EPOCHS,
        gradient_clip_val=1,
        gradient_clip_algorithm='norm'
        #overfit_batches=1,
        #limit_predict_batches=2,
        #limit_test_batches=2,
        #fast_dev_run=True,
        #limit_train_batches=1,
        #limit_val_batches=1,
        #limit_test_batches: Union[int, float] = 1.0,
    )


    model = Transformer(MODEL_NAME,
                        len(tagname_to_tagid),
                        empty_dataset,
                        training_loader,
                        val_loader,
                        weight_classes = weight_classes,
                        gpus=gpu_nb,
                        precision=16,
                        plugin='deepspeed_stage_3_offload',
                        accumulate_grad_batches=1,
                        max_epochs=MAX_EPOCHS,
                        dropout_rate=dropout_rate,
                        weight_decay=weight_decay,
                        warmup_steps=warmup_steps,
                        output_length=output_length,
                        learning_rate=learning_rate,
                        pred_threshold=pred_threshold
                        )

    trainer.fit(model)

    return model