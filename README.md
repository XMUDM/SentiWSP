# SentiELE
## For paper: Multi-granularity Textual Pre-training for Sentiment Classification

## Dependencies
- python>=3.6
- datasets>=1.7.0
- transformers==4.9.2 (Huggingface)
- fastcore>=1.0.20
- fastai<=2.1.10
- wandb>=0.10.4
- hugdatafast>=1.0.0

## Quick Start for Fine-tuning

### Download model and downstream dataset
Our experiments contain sentence-level sentiment classification (e.g. SST-5 / MR / IMDB / Yelp-2 / Yelp-5) and aspect-level sentiment analysis (e.g. Lap14 / Res14). 
You can download the pre-train model in ([Google Drive](https://hub.fastgit.org/)). 
You can download the downstream dataset from [huggingface/datasets](https://github.com/huggingface/datasets) or find download code in SentiELE_fine_tunning_SA.py

### Finetunning  
We show the example of fine-tuning SentiELE on sentence-level sentiment classification IMDB as follows:
```bash
python  SentiELE_fine_tunning_SA.py
	--dataset=imdb 
	--gpu_num=1 
	--loadmodel=True 
	--loadmodelpath=SentiELE 
	--batch_size=8 
	--max_epoch=5 
	--model_size=large 
	--num_class=2
```
the example of fine-tuning SentiELE on aspect-level sentiment analysis Lap14 as follows:
```bash
python  SentiELE_fine_tunning_ASBA.py
	--dataset=laptop 
	--model_name=SentiELE 
	--batch_size=32
	--max_epoch=10 
	--max_len=128 
```
## Pre-training
If you want to conduct pre-training by yourself instead of directly using the checkpoint we provide, this part may help you pre-process the pre-training dataset and run the pre-training scripts.



