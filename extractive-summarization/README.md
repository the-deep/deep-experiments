# extractive-summarization

# Setup

After setting up the `deep-experiments` repository (see the [README](../README.rst)), create a conda environment for extractive summarization:

```bash
cd extractive-summarization
conda env create -f environment.yml
conda activate extractive-summarization
```

and install the `deep-parser` tool:

```bash
pip install git+https://github.com/the-deep/deepex
```

## Data preparation

The prepared data is available on an S3 Bucket: TODO

Alternatively, you can acquire & prepare the data yourself:

### Data acquisition

Download the PDFs from S3:

```bash
aws s3 cp s3://pdfs-total pdfs-total --recursive
```

and run the following commands:

```bash
python extract_pdfs.py --leads_csv_path ../data/frameworks_data/data_v0.7.1/leads.csv --input_path pdfs-total --output_path pdf_texts # extracts pdfs
python extract_websites.py --leads_csv_path ../data/frameworks_data/data_v0.7.1/leads.csv --output_path website_texts # extracts websites
```

### Data preparation

`prepare_data.py` needs an additional Rust dependency for fuzzily matching excerpts. To install its Python wrapper, first install Rust by following the instructions on https://rustup.rs/, then run the following commands:
 
```bash
cd utils
maturin develop --release
```

Also, some Spacy models are required:

```bash
spacy download en_core_web_sm fr_core_news_sm es_core_news_sm
```

Once you have installed the additional dependencies and extracted the PDF and website texts locally, run:

```bash
python prepare_data.py --excerpts_csv_path ../data/frameworks_data/data_v0.7.1/train_v0.7.1.csv --lead_dirs pdf_texts website_texts --output_path=data.json
```

to create the final `data.json` file.

# Training

To train the latest model, run:

```bash
python train.py configs/best.json
```

By default, this will report progress to Weights and Biases (will require having logged in to `wandb` with `wandb login`). To disable this, add 

```json
"report_to": "none"
```

to the config file.

# Evaluation

TODO