import os
import torch
from datasets import load_dataset, Audio, concatenate_datasets
from transformers import (
    WhisperProcessor,
    WhisperTokenizer,
    WhisperFeatureExtractor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments
)
from peft import PeftModel, PeftConfig, prepare_model_for_kbit_training
from dataclasses import dataclass
from typing import Any, Dict, List, Union

# === Environment setup ===
os.environ["HF_DATASETS_CACHE"] = "/mnt/sda1/hf_cache"
os.environ["TRANSFORMERS_CACHE"] = "/mnt/sda1/transformers_cache"
os.environ["TMPDIR"] = "/mnt/sda1/tmp"

# === Load Dataset ===
dataset = load_dataset("Lingalingeswaran/asr-sinhala-dataset_json_v1_labeled")["train"]
dataset = dataset.cast_column("audio_path", Audio(sampling_rate=16000))
dataset = dataset.remove_columns(["label"])

# === Processor ===
model_name_or_path = "openai/whisper-large-v2"
language = "Sinhala"
task = "transcribe"

feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name_or_path)
tokenizer = WhisperTokenizer.from_pretrained(model_name_or_path, language=language, task=task)
processor = WhisperProcessor.from_pretrained(model_name_or_path, language=language, task=task)

# === Preprocess ===
def prepare_dataset(batch, idx):
    tokenized_length = len(tokenizer(batch["transcription"]).input_ids)
    if batch["audio_path"]["array"].shape[0] > 480000 or tokenized_length > 1024:
        return None
    audio_array = batch["audio_path"]["array"]
    sampling_rate = batch["audio_path"]["sampling_rate"]
    if torch.cuda.is_available():
        audio_array = torch.tensor(audio_array).to("cuda").cpu().numpy()
    input_features = feature_extractor(audio_array, sampling_rate=sampling_rate).input_features[0]
    labels = tokenizer(batch["transcription"]).input_ids
    return {"input_features": input_features, "labels": labels}

# === Process in Batches ===
processed_batches = []
batch_size = 2000
for i in range(0, len(dataset), batch_size):
    batch = dataset.select(range(i, min(i + batch_size, len(dataset)))).map(
        prepare_dataset,
        with_indices=True,
        remove_columns=["audio_path", "transcription"],
        num_proc=1
    )
    processed_batches.append(batch)

if len(processed_batches) >= 10:
    train_dataset = concatenate_datasets(processed_batches[:-10])
    eval_dataset = concatenate_datasets(processed_batches[-10:])
else:
    split = int(len(processed_batches) * 0.8)
    train_dataset = concatenate_datasets(processed_batches[:split])
    eval_dataset = concatenate_datasets(processed_batches[split:])

# === Data Collator ===
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch

data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

# === Load and prepare model ===
lora_checkpoint_path = "/mnt/sda1/FYP_2024/Helitha/DatasetsZ/asr_sinhala/Whisper-Fine-Tuning/temp/checkpoint-20244"
peft_config = PeftConfig.from_pretrained(lora_checkpoint_path)

# Load base model in 8-bit and prepare BEFORE LoRA
base_model = WhisperForConditionalGeneration.from_pretrained(
    peft_config.base_model_name_or_path,
    load_in_8bit=True,
    device_map="auto"
)
base_model = prepare_model_for_kbit_training(base_model)

# Load LoRA adapter with trainable layers
model = PeftModel.from_pretrained(base_model, lora_checkpoint_path, is_trainable=True)
model.print_trainable_parameters()

# === Training Arguments ===
training_args = Seq2SeqTrainingArguments(
    output_dir="temp2",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=1,
    learning_rate=1e-3,
    warmup_steps=50,
    num_train_epochs=3,
    evaluation_strategy="epoch",
    save_steps=500,
    save_total_limit=3,
    fp16=True,
    generation_max_length=128,
    logging_steps=25,
    remove_unused_columns=False,
    label_names=["labels"],
    report_to="none", 
)

# === Trainer ===
trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
    tokenizer=processor.feature_extractor,
)

# === Train ===
model.config.use_cache = False
trainer.train()
