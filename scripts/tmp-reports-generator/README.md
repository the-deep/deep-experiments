## reports_generator
This package is intended to generate abstractive reports based on sentences or paragraphs treating the same topic.

### Install

```posh
pip install git+https://github.com/the-deep/reports_generator #not working yet
```

### Usage
#### Basic Usage:
Get a report from raw text.
```
from reports_generator import ReportsGenerator
summarizer = ReportsGenerator()    
generated_report = summarizer(original_text)
```
#### Advanced Usage:
We give an example of another possible usage of the library, using a multilabeled data. 
```
import pandas as pd
from reports_generator import ReportsGenerator
summarizer = ReportsGenerator()

# Example of classified data to import, with two columns, one for the `original_text` and one for the `groundtruth_labels`, with one 
original_df_path = pd.read_csv([csv_path]) 
labels_list = original_df_path.groundtruth_labels.unique()

# Get summary for each label
summarized_data = {}
for one_label in labels_list:
    text_one_label = original_df_path[original_df_path.groundtruth_labels==one_label].original_text.tolist()
    generated_report[one_label] = summarizer(text_one_label)
```

### Documentation
- This package is based on pretrained models from the transformers library. It is restrained to the English language only for now. 
It was first intended to automatically generate reports for the humanitarian world. 
Examples of fully automatically generated reports can be found here. We need to ask premission to project owners and make the generated reports available...
- To keep the most relevant information only, the library is based on doing summarization iterations on the original text. 
- We show pseudocodes of the methodlogy used for reports creation
```
`Method: Main function call`
    `Inputs`: List of sentences or paragraph
    `Output`: summary
    
    summarized text = OneSummarizationIteration(original text)
    While (summarized text too long and maximum number of iterations not reached):
        summarized text = OneSummarizationIteration(summarized text)
        
        
        
`Method: OneSummarizationIteration`
    `Inputs`: text
    `Outputs`: Summarized Text
    
    1 - Split text into sentences
    2 - Get sentence embeddings for each sentence
        i - preprocess sentence (delete stop words, remove punctuation, stem and lemmatize words)
        ii - get sentence embeddings using preprocessed sentences (using pretrained HuggingFace models).
    3 - Cluster sentences into different categories using the HDBscan clustering method
    4 - Generate summary for each cluster (using pretrained HuggingFace models).
    5 - link summaries of each together together and return them.
```