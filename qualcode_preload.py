import os
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel, PeftConfig
from tqdm.notebook import tqdm


def process_doc_topics(input_csv='doc_topics.csv', output_csv='coded_lemmas_new.csv', model_name = "Likich/falcon-finetune-qualcoding_1000_prompt1_dot"):
    # Set CUDA environment
    with open('token.txt', 'r') as file:
        hf_token = file.read().strip()
        
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # Define BitsAndBytesConfig
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    # Load the PEFT model and tokenizer
    PEFT_MODEL = model_name
    config = PeftConfig.from_pretrained(PEFT_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        config.base_model_name_or_path,
        return_dict=True,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        token = hf_token
    )

    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path, token= hf_token)
    tokenizer.pad_token = tokenizer.eos_token

    # Device and generation configuration
    device = "cuda:0"
    generation_config = model.generation_config
    generation_config.max_new_tokens = 15
    generation_config.temperature = 0.7
    generation_config.top_p = 0.7
    generation_config.num_return_sequences = 1
    generation_config.pad_token_id = tokenizer.eos_token_id
    generation_config.eos_token_id = tokenizer.eos_token_id

    # Load the fine-tuned model
    mod = PeftModel.from_pretrained(model, PEFT_MODEL)

    # Function to process sentences
    def extract_ideas(sentences):
        summaries = []
        for sentence in tqdm(sentences, desc="Extracting ideas"):
            prompt = f"<human>:Summarize the main idea of a citation in less than 5 words.{sentence}\n<assistant>:".strip()
            encoding = tokenizer(prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = mod.generate(input_ids=encoding.input_ids, attention_mask=encoding.attention_mask, generation_config=generation_config)
            parts = tokenizer.decode(outputs[0]).split("<assistant>:")
            summary = parts[1].split("\n")[0].strip()
            summaries.append(summary)
        return summaries

    # Load input CSV
    coded_lemmas = pd.read_csv(input_csv, index_col=False)

    # Extract ideas from the 'paragraphs' column
    coded_lemmas['Extracted codes from lemmas'] = extract_ideas(coded_lemmas['paragraphs'])

    # Clean the DataFrame
    def clean_dataframe(df):
        df = df.replace(to_replace='"', value='', regex=True)
        df = df.replace(to_replace="<|endoftext|>", value='', regex=True)
        df = df.replace(to_replace=r'[\"\'\|\|]+', value='', regex=True)
        return df

    # Clean and save to output CSV
    coded_lemmas_cleaned = clean_dataframe(coded_lemmas)
    coded_lemmas_cleaned.to_csv(output_csv, index=False)
    print(f"Processing complete. Output saved to {output_csv}")

