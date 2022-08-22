 ## New model architecture
 ### files

```
├── train_mlflow.py         # general script used for models training
├── loss.py                 # loos function
├── pooling.py        	    # pooling function after transformer output
├── MLPModel.py             # classes for MLP models
├── TransformerModel.py     # classes for backbone models
├── ModelsExplainability.py # classes for models explainability
├── Inference.py            # inference class
├── utils.py                # general util functions for modelling
├── requirements.txt        # requirements to be installed before running the scripts
└── README.md
```
## Data:
- The training data is dta retried from the DEEP platform, annotated by humanitarian experts. We have over 100 tags regroruping primary tags and secondary tags.

## Training structure
- The objective is this new model architecture to adapt to the data drift and tagging differences across analysis frameworks in the DEEP platform.
- The training is done in two steps:
  - First, the backbone finetuning using a multitask learning architecture.
    - we finetune the backbone `nreimers/mMiniLMv2-L6-H384-distilled-from-XLMR-Large` from Hugigngface
    - When training, the embedding and the first layer are freezed
    - the last hidden layer is duplicated in the initialization and finetuned on different tasks, which implies a tree like multitask learning setup
  - Second, we train different two layer MLPs using the embedding model outputs. More specifically, we train one specific MLP for each analysis framework containing enough data and one MLP for the whole data.


<!--<ins>one model architecture</ins>: ![alt text](one_model_architecture.png) -->