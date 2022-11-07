import argparse
import random
import json, os

from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from sklearn.metrics import precision_recall_curve,classification_report

from transformers import ElectraTokenizer, ElectraForSequenceClassification, ElectraConfig, get_linear_schedule_with_warmup,ElectraTokenizerFast
from transformers import AutoTokenizer,  AutoModelForSequenceClassification
import absa_data_utils as data_utils
from transformers import WEIGHTS_NAME, CONFIG_NAME

# 随机数种子
random_seed = 2022
random.seed(random_seed)
torch.manual_seed(random_seed)

# parser config
parser = argparse.ArgumentParser(description='Pre training model configuration')
parser.add_argument('--data_path', default="./jsondata/",help='finetune data path')
parser.add_argument('--dataset', default="rest",help='finetune data name')
parser.add_argument('--inference', default="valid",help='Choose whether to use a test set or a validation set')
parser.add_argument('--model_path', default='./pretrain_model/',help='finetune model path')
parser.add_argument('--model_name', default='ance7word5',help='finetune model name')
parser.add_argument('--tokenizer_path', default="./pretrain_model/electra/large",help='finetune tokenizer path')
parser.add_argument('--single_gpunum', default='0',help='GPU number used in single GPU environment')
parser.add_argument('--save_model', type=bool ,default = False ,help='wheather you want to save the model')
parser.add_argument('--save_model_path', default='./model_result/',help='the save dir of fine-tunning model path')
parser.add_argument('--result_path', default='./result/',help='the save dir of fine-tunning experiment result')
parser.add_argument('--save_model_name', default='ABSA_checkpoint',help='model name')
parser.add_argument('--batch_size',type=int ,default=32,help='the batch size of training process')
parser.add_argument('--max_epoch', type=int,default=10,help='the max epoch of training process')
parser.add_argument('--lr', type=float,default = 1e-5,help='the learning rate of fine-tunning process')
parser.add_argument('--eps', default = 1e-8,help='the eps of fine-tunning process')
parser.add_argument('--max_len', type=int, default = 128,help='the max_length of input sequence')

args = parser.parse_args()

def validation(model,test_loader):
    
    model.eval()
    y_true=[]
    y_pred=[]
    total_eval_loss=0
                      
    for batch in test_loader:

        with torch.no_grad():
            batch = tuple(t.to(device) for t in batch)
            input_ids, segment_ids, input_mask, label_ids = batch
            model_input = {'input_ids':input_ids,
                           'token_type_ids':segment_ids, 
                           'attention_mask':input_mask,
                           'labels':label_ids}
            outputs = model(**model_input)

        loss = outputs.loss
        logits = outputs.logits

        total_eval_loss += loss.item()
        prediction = torch.argmax(logits, 1)
        prediction = prediction.detach().cpu().tolist()
        label_ids = label_ids.to('cpu').tolist()   
        y_pred+=prediction
        y_true+=label_ids
    target_names = ['positive', 'negative', 'neutral']
    print("-------------------------------")
    print(classification_report(y_true, y_pred, target_names=target_names,digits=4))
    file = open(args.result_path + args.model_name+'_'+args.dataset + "_result.txt", "a+")
    file.write(classification_report(y_true, y_pred, target_names=target_names,digits=4) + '\n' + '\n')
    file.close()
    print("Average valid loss: %.4f"%(total_eval_loss/len(test_loader)))
    print("-------------------------------")


def train(model, train_loader, max_epoch, test_loader):
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=args.lr,eps = args.eps)
    total_steps = len(train_loader) * max_epoch
    warmup_steps=0.1*len(train_loader)
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps = warmup_steps, # Default value in run_glue.py
                                                num_training_steps = total_steps)
    model.train()
    
    for epoch in range(max_epoch):
        
        total_train_loss = 0
        
        for iter_num, batch in enumerate(tqdm(train_loader)):
            
            batch = tuple(t.to(device) for t in batch)
            input_ids, segment_ids, input_mask, label_ids = batch
            model_input = {'input_ids':input_ids,
                           'token_type_ids':segment_ids, 
                           'attention_mask':input_mask,
                           'labels':label_ids}
            
            outputs = model(**model_input)
            loss = outputs.loss
            total_train_loss += loss.item()
            loss.backward()
            
            # clip
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm = 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            if iter_num % 20 == 0:
                # print(label_ids)
                # print(torch.argmax(outputs.logits, 1))
                print("epoth: %d, iter_num: %d, loss: %.4f" % (epoch, iter_num, loss.item()))
        
        validation(model, test_loader)
        if args.save_model:
            save_model(model, args.save_model_path + str(epoch+1) + "epoch_" + args.save_model_name)
        print("Epoch: %d, Average training loss: %.4f" %(epoch, total_train_loss/len(train_loader)))

def save_model(model,output_dir):
    if os.path.exists(output_dir) == False:
        os.makedirs(output_dir)
    model_to_save = model.module if hasattr(model, 'module') else model
    output_model_file = os.path.join(output_dir, WEIGHTS_NAME)
    output_config_file = os.path.join(output_dir, CONFIG_NAME)
    torch.save(model_to_save.state_dict(), output_model_file)
    model_to_save.config.to_json_file(output_config_file)
    print("save model in :"+output_dir)

    
if __name__ == "__main__":
    
    # Define the operating environment
    device = torch.device("cuda:" + str(args.single_gpunum) if torch.cuda.is_available() else "cpu")
    data_path=args.data_path+args.dataset
    # load data
    processor = data_utils.AscProcessor()
    label_list = processor.get_labels()
    train_examples = processor.get_train_examples(data_path)
    print("load dataset:",args.dataset)
    # ABSA standard data input format conversion
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    
    train_features = data_utils.convert_examples_to_features(
        train_examples, label_list, args.max_len, tokenizer)
    
    # turn to tensor
    all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
    train_data = TensorDataset(all_input_ids, all_segment_ids, all_input_mask, all_label_ids)
    
    # Random Sampler DataLoader
    train_sampler = RandomSampler(train_data)
    train_loader = DataLoader(train_data, sampler=train_sampler, batch_size=args.batch_size)
    
    # load model
    model_path = args.model_path+args.model_name
    model = AutoModelForSequenceClassification.from_pretrained(save_path, num_labels=len(label_list))
    model.to(device)
    print("load model:", args.model_name)

    # create directory
    if os.path.exists(args.result_path) == False:
        os.makedirs(args.result_path)
    if args.save_model and os.path.exists(args.save_model_path) == False:
        os.makedirs(args.save_model_path)

    # Load valid dataset
    if args.inference == "valid":
        valid_examples = processor.get_test_examples(data_path)
        valid_features = data_utils.convert_examples_to_features(
            valid_examples, label_list, args.max_len, tokenizer)
        valid_all_input_ids = torch.tensor([f.input_ids for f in valid_features], dtype=torch.long)
        valid_all_segment_ids = torch.tensor([f.segment_ids for f in valid_features], dtype=torch.long)
        valid_all_input_mask = torch.tensor([f.input_mask for f in valid_features], dtype=torch.long)
        valid_all_label_ids = torch.tensor([f.label_id for f in valid_features], dtype=torch.long)
        valid_data = TensorDataset(valid_all_input_ids, valid_all_segment_ids, valid_all_input_mask, valid_all_label_ids)
        valid_sampler = SequentialSampler(valid_data)
        valid_loader = DataLoader(valid_data, sampler=valid_sampler, batch_size=args.batch_size)
        # start to train
        train(model, train_loader, args.max_epoch, valid_loader)
    # use test dataset
    elif args.inference == "test":
        test_examples = processor.get_test_examples(data_path)
        test_features = data_utils.convert_examples_to_features(
            test_examples, label_list, args.max_len, tokenizer)
        test_all_input_ids = torch.tensor([f.input_ids for f in test_features], dtype=torch.long)
        test_all_segment_ids = torch.tensor([f.segment_ids for f in test_features], dtype=torch.long)
        test_all_input_mask = torch.tensor([f.input_mask for f in test_features], dtype=torch.long)
        test_all_label_ids = torch.tensor([f.label_id for f in test_features], dtype=torch.long)
        test_data = TensorDataset(test_all_input_ids, test_all_segment_ids, test_all_input_mask, test_all_label_ids)
        test_sampler = SequentialSampler(test_data)
        test_loader = DataLoader(test_data, sampler=test_sampler, batch_size=args.batch_size)
        # start to train
        train(model, train_loader, args.max_epoch, test_loader)
    else:
        print("inference must be test or valid")
        exit(0)
    
