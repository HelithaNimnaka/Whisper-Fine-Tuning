import os
import pandas as pd
from datasets import Dataset, Audio
from tqdm import tqdm
import soundfile as sf
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from transformers import (
    WhisperProcessor,
    WhisperTokenizer,
    WhisperFeatureExtractor,
)

# === Paths ===
csv_path = "/mnt/sda1/FYP_2024/Helitha/DatasetsZ/asr_sinhala/train.csv"
base_audio_path = "/mnt/sda1/FYP_2024/Helitha/DatasetsZ/"
output_dir = "/mnt/sda1/FYP_2024/Helitha/DatasetsZ/asr_sinhala/output"
os.makedirs(output_dir, exist_ok=True)

# === Load CSV & Fix Audio Paths ===
trainCSV = pd.read_csv(csv_path)


trainCSV["file"] = trainCSV["file"].apply(lambda x: os.path.join(base_audio_path, x))


# === Create Hugging Face Dataset ===
dataset = Dataset.from_pandas(trainCSV)

model_name_or_path = "openai/whisper-large-v2"
task = "transcribe"
language = "Sinhala"


feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name_or_path)

tokenizer = WhisperTokenizer.from_pretrained(model_name_or_path, language=language, task=task)

processor = WhisperProcessor.from_pretrained(model_name_or_path, language=language, task=task)

# === Cast audio column to 16kHz ===
dataset = dataset.cast_column("file", Audio(sampling_rate=16000))

dataset = dataset.remove_columns(["Unnamed: 0", "filename", "x", "full"])


import os
import torch
import torch.multiprocessing as mp
from datasets import Dataset, concatenate_datasets

# ✅ Fix multiprocessing issue with CUDA
mp.set_start_method("spawn", force=True)

# Define output path and ensure directory exists
output_path = output_dir
os.makedirs(output_path, exist_ok=True)

# Check if the output folder already contains processed batches
if os.listdir(output_path):
    print("Output folder is not empty. Loading pre-extracted features...")
    processed_batches = []
    # Load each batch saved as an arrow file
    for filename in os.listdir(output_path):
        if filename.endswith(".arrow"):
            file_path = os.path.join(output_path, filename)
            print(f"Loading {filename}...")
            batch = Dataset.load_from_disk(file_path)
            processed_batches.append(batch)
    # Optionally, concatenate all batches into a single dataset
    print(f"Loaded {len(processed_batches)} batches.")
    processed_dataset = concatenate_datasets(processed_batches[:-10])
    train_dataset = processed_dataset
    eval_dataset = concatenate_datasets(processed_batches[-10:])


    print("Feature extraction loaded from disk.")
else:
    print("Output folder is empty. Extracting features...")

    # Define the function to prepare each dataset sample
    def prepare_dataset(batch, idx):
        # Calculate tokenized length
        tokenized_length = len(tokenizer(batch["sentence"]).input_ids)
        
        # Skip examples with long audio or text
        if batch["file"]["array"].shape[0] > 480000 or tokenized_length > 1024:
            print(f"Skipping example {idx}: Tokenized length ({tokenized_length}) exceeds the limit.")
            return None  # Skip this sample

        # Extract audio array and sampling rate
        audio_array = batch["file"]["array"]
        sampling_rate = batch["file"]["sampling_rate"]

        # ✅ Move audio tensor to GPU (if available) and then back to CPU
        if torch.cuda.is_available():
            audio_array = torch.tensor(audio_array).to("cuda")  # Move to GPU
            audio_array = audio_array.cpu().numpy()  # Convert back to CPU NumPy array

        # Compute log-Mel spectrogram features
        input_features = feature_extractor(audio_array, sampling_rate=sampling_rate).input_features[0]

        # Tokenize text
        labels = tokenizer(batch["sentence"]).input_ids

        return {"input_features": input_features, "labels": labels}

    # Process the dataset in chunks to manage memory
    batch_size = 2000  # Adjust for memory usage
    num_batches = len(dataset) // batch_size + 1

    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(dataset))

        print(f"Processing batch {i+1}/{num_batches}...")

        # Process the current chunk
        processed_batch = dataset.select(range(start_idx, end_idx)).map(
            prepare_dataset,
            with_indices=True,         # Pass the index to the function
            remove_columns=["file", "sentence"],  # Remove original columns
            num_proc=1                 # ✅ Use 1 process for CUDA compatibility
        )

        # Save the processed batch to disk
        batch_filename = os.path.join(output_path, f"processed_batch_{i}.arrow")
        processed_batch.save_to_disk(batch_filename)
        print(f"Saved batch {i+1} to {batch_filename}")

    print("Processing complete. Processed dataset saved to:", output_path)

import torch

from dataclasses import dataclass
from typing import Any, Dict, List, Union


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch


data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)


import evaluate

metric = evaluate.load("wer")

def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    # we do not want to group tokens when computing the metrics
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = 100 * metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}

    
from transformers import WhisperForConditionalGeneration

model = WhisperForConditionalGeneration.from_pretrained(model_name_or_path, load_in_8bit=True, device_map="auto")

# model.hf_device_map - this should be {" ": 0}

model.config.forced_decoder_ids = None
model.config.suppress_tokens = []

# Ensure the latest version of peft is installed
#%pip install --upgrade peft

from peft import prepare_model_for_kbit_training

# Use the correct function for preparing the model
model = prepare_model_for_kbit_training(model)

from peft import LoraConfig, PeftModel, LoraModel, LoraConfig, get_peft_model

config = LoraConfig(r=32, lora_alpha=64, target_modules=["q_proj", "v_proj"], lora_dropout=0.05, bias="none")

model = get_peft_model(model, config)
model.print_trainable_parameters()

from transformers import Seq2SeqTrainingArguments

training_args = Seq2SeqTrainingArguments(
    output_dir="temp",  # change to a repo name of your choice
    per_device_train_batch_size=8,
    gradient_accumulation_steps=1,  # increase by 2x for every 2x decrease in batch size
    learning_rate=1e-3,
    warmup_steps=50,
    num_train_epochs=3,
    evaluation_strategy="epoch",
    fp16=True,
    per_device_eval_batch_size=8,
    generation_max_length=128,
    logging_steps=25,
    remove_unused_columns=False,  # required as the PeftModel forward doesn't have the signature of the wrapped model's forward
    label_names=["labels"],  # same reason as above
)

from transformers import Seq2SeqTrainer

trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
    # compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor,
)
model.config.use_cache = False  # silence the warnings. Please re-enable for inference!

#trainer.train()
trainer.train(resume_from_checkpoint="/mnt/sda1/FYP_2024/Helitha/DatasetsZ/asr_sinhala/temp/checkpoint-15500")
