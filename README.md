# BERTTraining
All files and models for training BERT on Java source code

## Content
2 trained models: best_checkpoint-1820000_multi, best_checkpoint_110000_own_vocab  
Python files: run_lm_finetuning, model_chain, benchmark

### Models
Trained on 30 training files each containing 10k Java files for 2 epochs, rest on standard configuration  
the own_vocab has a modifed vocabulary file that contains additional Java keywords

### Python files
`run_lm_finetuning.py`: standard training file from the [Huggingface repo](https://github.com/huggingface/transformers)  
`model_chain.py`: multiprocessing of training files, multiple files are allowed, code from [here](https://github.com/EndruK/transformers)  
`benchmark.py`: benchmark for trained model

## How to use it
1. install transformers and the requirements it needs (pytorch & Tensorflow 2.0) ( I used transformers 2.1.1, when using a newer version, the run scripts need to be updated)
2. place model files and run scripts in a folder
3. run one of the scripts
4. wait for the trainig to finish
5. run benchmark

## Run scripts
**Examples I used to train BERT**
Using the normal vocabulary and pretrained model with multiprocessing:
```
python3 ./model_chain.py --data_folder=/path/to/trainfilesdir --output_dir=/path/to/outputdir --model_type=bert --model_name_or_path=bert-base-cased --do_train --do_eval --evaluate_during_training --per_gpu_train_batch_size=1 --per_gpu_eval_batch_size=1 --num_train_epochs=1 --save_steps=100 --save_total_limit=100 --overwrite_output_dir --gpu_ids=0 --pre_process_count=8 --logging_steps=5000 --mlm

```
Using the normal pretrained model and an own vocabulary:
```
python3 ./model_chain.py --data_folder=/path/to/trainfilesdir --output_dir=/path/to/outputdir --cache_dir=/path/to/cachedir --model_type=bert --model_name_or_path=bert-base-cased --do_train --do_eval --evaluate_during_training --per_gpu_train_batch_size=1 --per_gpu_eval_batch_size=1 --num_train_epochs=1 --save_steps=100 --save_total_limit=100 --overwrite_output_dir --gpu_ids=0 --pre_process_count=8 --logging_steps=5000 --mlm

```
The cache folder will contain the cached files the BERT model loads on start with the vocabulary cached file altered to the desired vocabulary. This is a work-around to avoid an error where using an modified vocabulary is not recognized properly, see this [issue](https://github.com/huggingface/transformers/issues/1871)

## Benchmark
1. open the file in an editor of your choice
2. change the path to your trained model
3. alter parameters if needed
4. run 
