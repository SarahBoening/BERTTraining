# BERTTraining
All files and models for training BERT on Java source code

## Content
trained models: Can be found [here](https://www.dropbox.com/sh/3rma84xdvwlnkif/AADSzYlI5BnuSIFaWwO58fpea?dl=0)
Python files for working with this model

### Models
`multi`and `own_vocab` were trained on 30 training files each containing 10k Java files for 2 epochs, rest on standard configuration  
the own_vocab has a modifed vocabulary file that contains additional Java keywords
`bert_full_training` is currently in training. It will be trained on all 212 10k Java files ( approx. 2,1 Mil files) for 1 epoch with standard configurations.
### Python files
`run_lm_finetuning.py`: standard training file from the [Huggingface repo](https://github.com/huggingface/transformers)  
`model_chain.py`: multiprocessing of training files, multiple files are allowed, code from [here](https://github.com/EndruK/transformers)  
`benchmark.py`: benchmark for trained model  
`bert_generation.py` script to test longer text predictions/ generation with BERT  
`finetuned_Bert_Test.py` simple script to test models for BertForMaskedLM  
`java_generation.py` test to try different predictions for BERT (argmax, sampling)  
`load_dataset.py` file to produce subsets of a corpus as .raw files    
## How to use it
1. install transformers and the requirements it needs (pytorch & Tensorflow 2.0) ( I used transformers 2.1.1, when using a newer version, the run-scripts need to be updated)
2. place model files and run-scripts in a folder
3. run one of the scripts
4. wait for the training to finish
5. run benchmark (optionally)

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
Using multiprocessing and a single (small) evaluation file:  
```
python3 ./model_chain.py --data_folder=/home/nilo4793/raid/corpora/Java_split --eval_path=/home/nilo4793/raid/corpora/Java_split/valid_file.java_github_1k.raw --output_dir=/home/nilo4793/raid/output/bert_full_training --model_type=bert --model_name_or_path=bert-base-cased --do_train --do_eval --evaluate_during_training --per_gpu_train_batch_size=1 --per_gpu_eval_batch_size=1 --num_train_epochs=2 --save_steps=100 --save_total_limit=100 --overwrite_output_dir --gpu_ids=6 --pre_process_count=8 --logging_steps=5000 --mlm
```
## Benchmark
1. open the file in an editor of your choice
2. change the path to your trained model & a valid file to use for the benchmark
3. alter parameters if needed
4. run 
