import sys
import os
import argparse
import logging
import json
import pandas as pd

from transformers import MarianMTModel, MarianTokenizer
import timeit

logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument("--batch_size", type=int, default=128)
    
    # Data, model, and output directories
    parser.add_argument("--output-data-dir", type=str, default=os.environ["SM_OUTPUT_DATA_DIR"])
    parser.add_argument("--model_dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--n_gpus", type=str, default=os.environ["SM_NUM_GPUS"])
    parser.add_argument("--training_dir", type=str, default=os.environ["SM_CHANNEL_TRAIN"])
    parser.add_argument("--val_dir", type=str, default=os.environ["SM_CHANNEL_TEST"])
    args, _ = parser.parse_known_args()

    # Set up logging
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        level=logging.getLevelName("INFO"),
        handlers=[logging.StreamHandler(sys.stdout)],
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # load datasets

    ########################################

    def read_merge_data(
        TRAIN_PATH, VAL_PATH, data_format: str = "csv"
    ):

        if data_format == "pickle":
            train_df = pd.read_pickle(f"{TRAIN_PATH}/train.pickle")
            val_df = pd.read_pickle(f"{VAL_PATH}/val.pickle")

        else:
            train_df = pd.read_csv(TRAIN_PATH)
            val_df = pd.read_csv(VAL_PATH)

        all_dataset = pd.concat([train_df, val_df])

        return all_dataset

    all_dataset = read_merge_data(args.training_dir, args.val_dir, data_format="pickle")

    """es_to_fr_model_name = 'Helsinki-NLP/opus-mt-es-fr'
    es_to_fr_tokenizer = MarianTokenizer.from_pretrained(es_to_fr_model_name)
    es_to_fr_model = MarianMTModel.from_pretrained(es_to_fr_model_name).to('cuda:0')

    fr_to_es_model_name = 'Helsinki-NLP/opus-mt-fr-es'
    fr_to_es_tokenizer = MarianTokenizer.from_pretrained(fr_to_es_model_name)
    fr_to_es_model = MarianMTModel.from_pretrained(fr_to_es_model_name).to('cuda:0')"""

    en_to_romance_model_name = 'Helsinki-NLP/opus-mt-en-ROMANCE'
    en_to_romance_tokenizer = MarianTokenizer.from_pretrained(en_to_romance_model_name)
    en_to_romance_model = MarianMTModel.from_pretrained(en_to_romance_model_name).to('cuda:0')

    """romance_to_en_model_name = 'Helsinki-NLP/opus-mt-ROMANCE-en'
    romance_to_en_tokenizer = MarianTokenizer.from_pretrained(romance_to_en_model_name)
    romance_to_en_model = MarianMTModel.from_pretrained(romance_to_en_model_name).to('cuda:0')"""
    

    def translate_lists(texts, model, tokenizer, language="fr"):
        # Prepare the text data into appropriate format for the model
        template = lambda text: f">>{language}<< {text}"
        src_texts = [template(text) for text in texts]

        # Tokenize the texts
        encoded = tokenizer.prepare_seq2seq_batch(src_texts,
                                                return_tensors='pt').to('cuda:0')
        
        # Generate translation using model
        translated = model.generate(**encoded)

        # Convert the generated tokens indices back into text
        translated_texts = tokenizer.batch_decode(translated, skip_special_tokens=True)
        
        return translated_texts

    def generate_new_data (source_df, model, tokenizer, target_language:str):
        list_texts = source_df.excerpt.tolist()
        generated_sentences = []
        for i in range(0, len(list_texts), args.batch_size):
            list_tmp = list_texts[i:i+args.batch_size]
            translated_tmp = translate_lists(list_tmp, model, tokenizer, language=target_language)
            for sentence in translated_tmp:
                generated_sentences.append(sentence)
        generated_df = source_df[['id', 'language']]
        generated_df['language'] = generated_df['language'].apply(lambda x: x + '->' + target_language)
        generated_df['excerpt'] = generated_sentences
        return generated_df

    en_df = all_dataset[all_dataset.language=='en']
    """fr_df = all_dataset[all_dataset.language=='fr']
    es_df = all_dataset[all_dataset.language=='es']"""

    fr_from_en_df = generate_new_data (en_df, en_to_romance_model, en_to_romance_tokenizer, 'fr')
    fr_from_en_df.to_csv(f"{args.output_data_dir}/fr_from_en_df.csv")
    es_from_en_df = generate_new_data (en_df, en_to_romance_model, en_to_romance_tokenizer, 'es')
    """fr_from_es_df = generate_new_data (es_df, es_to_fr_model, es_to_fr_tokenizer, 'fr')
    es_from_fr_df = generate_new_data (fr_df, fr_to_es_model, es_to_fr_tokenizer, 'es')
    en_from_es_df = generate_new_data (es_df, romance_to_en_model, romance_to_en_tokenizer, 'es')
    en_from_fr_df = generate_new_data (fr_df, romance_to_en_model, romance_to_en_tokenizer, 'fr')"""

    tot_generated_df = pd.concat([
        fr_from_en_df,
        es_from_en_df,
        #fr_from_es_df,
        #es_from_fr_df,
        #en_from_es_df,
        #en_from_fr_df
    ])

    tot_generated_df.to_csv(f"{args.output_data_dir}/generated_df_with_translation.csv")
