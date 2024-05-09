import numpy as np
import pandas as pd
import argparse
import nltk
nltk.download('punkt')
import torch
import datasets
from datasets import Dataset
from datasets import load_metric
import transformers
from transformers import AutoTokenizer
import transformers
from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
import gc

gc.collect()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("[INFO] training using {}".format(torch.cuda.get_device_name(0)))

max_input_length = 1024
max_target_length = 128

model_checkpoint ='t5-small'


def load_datasets(dataset_path):
  train_data = pd.read_csv(dataset_path + "train.csv")
  test_data = pd.read_csv(dataset_path + "test.csv")
  validation_data = pd.read_csv(dataset_path + "validation.csv")

  train_data = train_data[['article','highlights']]
  test_data = test_data[['article','highlights']]
  validation_data = validation_data[['article','highlights']]


  train_data = train_data.rename(columns={ 'article':'document', 'highlights':'summary'})
  test_data = test_data.rename(columns={ 'article':'document', 'highlights':'summary'})
  validation_data = validation_data.rename(columns={ 'article':'document', 'highlights':'summary'})

  train_data = train_data.dropna()
  test_data = test_data.dropna()
  validation_data = validation_data.dropna()


  train_data['document']= train_data['document'].apply(lambda x: x.lower())
  train_data['summary'] = train_data['summary'].apply(lambda x: x.lower())


  test_data['document']= test_data['document'].apply(lambda x: x.lower())
  test_data['summary'] = test_data['summary'].apply(lambda x: x.lower())

  validation_data['document']= validation_data['document'].apply(lambda x: x.lower())
  validation_data['summary'] = validation_data['summary'].apply(lambda x: x.lower())


  return train_data, test_data, validation_data



def convert_hf_format(pandas_dataset):
  hf_dataset = Dataset.from_pandas(pandas_dataset)
  return hf_dataset




def load_processed_dataset(dataset_path, tokenizer):

	train_data, test_data, validation_data = load_datasets(dataset_path)

	train = convert_hf_format(train_data)
	valid = convert_hf_format(validation_data)
	test = convert_hf_format(test_data)


	def preprocess_function(examples):
		inputs = ['summarize:' + doc for doc in examples["document"]]
		model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True,padding='max_length')

		# Setup the tokenizer for targets
		with tokenizer.as_target_tokenizer():
			labels = tokenizer(examples["summary"], max_length=max_target_length, truncation=True)

		model_inputs["labels"] = labels["input_ids"]
		return model_inputs


	tokenized_train = train.map(preprocess_function, batched=True)
	tokenized_valid = valid.map(preprocess_function, batched=True)
	tokenized_test = test.map(preprocess_function, batched=True)


	return tokenized_train, tokenized_valid, tokenized_test



def load_tokenizer(model_checkpoint):


  tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

  pad_on_right = tokenizer.padding_side == "right"

  return tokenizer


def load_model_artifacts(model_checkpoint):

  tokenizer = load_tokenizer(model_checkpoint)

  model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
  batch_size = 16
  data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

  model.to(DEVICE)

  return model, tokenizer, batch_size, data_collator



def train_model(model_checkpoint, batch_size, model, tokenized_train, tokenized_valid, data_collator, tokenizer, output_path):
        
        


		def compute_metrics(eval_pred):
			predictions, labels = eval_pred
			decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
			# Replace -100 in the labels as we can't decode them.
			labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
			decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

			# Rouge expects a newline after each sentence
			decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
			decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]

			result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
			# Extract a few results
			result = {key: value.mid.fmeasure * 100 for key, value in result.items()}

			# Add mean generated length
			prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
			result["gen_len"] = np.mean(prediction_lens)

			return {k: round(v, 4) for k, v in result.items()}




		model_name = model_checkpoint.split("/")[-1]
		args = Seq2SeqTrainingArguments(
			f"{model_name}-finetuned-newsarticles",
			evaluation_strategy = "epoch",
			learning_rate=2e-5,
			per_device_train_batch_size=batch_size,
			per_device_eval_batch_size=batch_size,
			weight_decay=0.01,
			save_total_limit=3,
			num_train_epochs=5,
			predict_with_generate=True,
			fp16=True)

		trainer = Seq2SeqTrainer(
			model,
			args,
			train_dataset=tokenized_train,
			eval_dataset=tokenized_valid,
			data_collator=data_collator,
			tokenizer=tokenizer,
			compute_metrics=compute_metrics)

		metric = load_metric("rouge")

		print("Training In Progress")

		trainer.train()

		print("Training Completed")

		print("Saving Model")
		trainer.save_model(output_path)
		print("Model Saved")



def master_function(args):


	dataset_path = str(args.train_file)

	output_path = str(args.output_directory)

	torch.cuda.empty_cache()
	model, tokenizer, batch_size, data_collator = load_model_artifacts(model_checkpoint)

	tokenized_train, tokenized_valid, tokenized_test = load_processed_dataset(dataset_path, tokenizer)

	train_model(model_checkpoint, batch_size, model, tokenized_train, tokenized_valid, data_collator, tokenizer, output_path)


def parse_arguments():
  
  """Parse job arguments."""

  parser = argparse.ArgumentParser()
  # required input arguments
  parser.add_argument(
      '--train_file',
      help='path to training data',
      required=True
  )

  parser.add_argument(
      '--output_directory',
      help='path to output directory',
      required=True
  )
  
  args = parser.parse_args()

  return args



if __name__ == "__main__":

	args = parse_arguments()

	dataset_path = "us-east1-composer-training-50d83041-bucket/data/"
	output_path = "summarization_bucket_2023/model_artifacts/"
	master_function(args)





