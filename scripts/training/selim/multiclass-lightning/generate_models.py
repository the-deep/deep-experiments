import os
from pytorch_lightning.loggers import TensorBoardLogger
import pytorch_lightning as pl
from transformers import AutoTokenizer

from classes import (
    CustomDataset,
    Transformer
)
from utils import tagname_to_id


def train_on_specific_targets (train_dataset,
                                val_dataset,
                                name_classifier:str,
                                MODEL_NAME:str,
                                TOKENIZER_NAME:str,
                                early_stopping_callback,
                                checkpoint_callback,
                                dropout_rate:float,
                               train_params,
                               val_params,
                                gpu_nb:int,
                               MAX_EPOCHS:int,
                               weight_decay=0.02,
                               warmup_steps=500,
                               output_length=384,
                               max_len=150,
                               weight_classes=None,
                               learning_rate=3e-5,
                               pred_threshold:float=0.5):
    """
    main function used to train model
    """
    # if not os.path.exists(dirpath):
    #    os.makedirs(dirpath)

    train_dataset = train_dataset[['entry_id', 'excerpt', 'target']]
    val_dataset = val_dataset[['entry_id', 'excerpt', 'target']]

    logger = TensorBoardLogger("lightning_logs", name=name_classifier)

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    tagname_to_tagid = tagname_to_id (train_dataset["target"])

    empty_dataset = CustomDataset(None, tagname_to_tagid, tokenizer, max_len)
    
    

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
                        tagname_to_tagid,
                        empty_dataset,
                        train_dataset=train_dataset,
                        val_dataset=val_dataset,
                        train_params=train_params,
                        val_params=val_params,
                        weight_classes = weight_classes,
                        tokenizer=tokenizer,
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