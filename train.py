import argparse
import glob
import os
import json
import time
import logging
import random
import re
from itertools import chain
from string import punctuation
import sys

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl


from transformers import (
    AdamW,
    T5ForConditionalGeneration,
    T5Tokenizer,
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup
)


def set_seed(seed):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

set_seed(42)

class T5FineTuner(pl.LightningModule):
  def __init__(self, hparams):
    super(T5FineTuner, self).__init__()
    self.args = hparams
    
    #self.hparams = hparams
    
    self.model = T5ForConditionalGeneration.from_pretrained(self.args.model_name_or_path)
    self.tokenizer = T5Tokenizer.from_pretrained(self.args.tokenizer_name_or_path)
    
    for n, p in self.model.named_parameters():
      if "shared" in n or "lm_head" in n:
        p.requires_grad = True
        
      else:
        p.requires_grad = False
      
    
    total_params = sum(
	    param.numel() for param in self.model.parameters()
    )
    print(total_params)

    

    
    self.model.resize_token_embeddings(len(self.tokenizer))
    for p in self.model.get_input_embeddings().parameters():
      self.orig_init_emb = p

    for p in self.model.get_output_embeddings().parameters():
      self.orig_init_lin = p

    
    
    
    new_tokens = ["[S*]"]
    
    self.new_tokens = new_tokens
    self.tokenizer.add_tokens(list(new_tokens))

    dummy_tokens = ["[S" + "*"*(j+2)  + "]" for j in range(int(sys.argv[4]))]
    self.tokenizer.add_tokens(list(dummy_tokens))
    
    self.model.resize_token_embeddings(len(self.tokenizer))
    

  def is_logger(self):
    return True
  
  def forward(
      self, input_ids, attention_mask=None, decoder_input_ids=None, decoder_attention_mask=None, lm_labels=None
  ):
    
    return self.model(
        input_ids,
        attention_mask=attention_mask,
        decoder_input_ids=decoder_input_ids,
        decoder_attention_mask=decoder_attention_mask,
        labels=lm_labels,
    )

  def _step(self, batch):
    lm_labels = batch["target_ids"]
    
    lm_labels[lm_labels[:, :] == self.tokenizer.pad_token_id] = -100

    outputs = self(
        input_ids=batch["source_ids"],
        attention_mask=batch["source_mask"],
        lm_labels=lm_labels,
        decoder_attention_mask=batch['target_mask']
    )
    loss = outputs[0]

    return loss

  def training_step(self, batch, batch_idx):
    loss = self._step(batch)

    tensorboard_logs = {"train_loss": loss}
    return {"loss": loss, "log": tensorboard_logs}
  
  def training_epoch_end(self, outputs):
    avg_train_loss = torch.stack([x["loss"] for x in outputs]).mean()
    tensorboard_logs = {"avg_train_loss": avg_train_loss}

  def validation_step(self, batch, batch_idx):
    loss = self._step(batch)
    return {"val_loss": loss}
  
  def validation_epoch_end(self, outputs):
    avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
    tensorboard_logs = {"val_loss": avg_loss}
    self.log("val_loss", avg_loss)
    self.model.save_pretrained(f't5_large_{sys.argv[2]}')
    self.tokenizer.save_pretrained(f't5_large_{sys.argv[2]}')

    return {"avg_val_loss": avg_loss, "log": tensorboard_logs, 'progress_bar': tensorboard_logs}


  def configure_optimizers(self):
    "Prepare optimizer and schedule (linear warmup and decay)"

    model = self.model
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": self.args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate, eps=self.args.adam_epsilon)
    self.opt = optimizer
    return [optimizer]

  def optimizer_step(self,
                    epoch=None, 
                  batch_idx=None, 
                  optimizer=None, 
                  optimizer_idx=None, 
                  optimizer_closure=None, 
                  on_tpu=None, 
                  using_native_amp=None, 
                  using_lbfgs=None
                    ):
    

    if batch_idx == 0:
      self.model.save_pretrained(f't5_large_{sys.argv[2]}')
      self.tokenizer.save_pretrained(f't5_large_{sys.argv[2]}')

    optimizer.step(closure=optimizer_closure)
    optimizer.zero_grad()
    self.lr_scheduler.step()

  
    
  def get_tqdm_dict(self):
    tqdm_dict = {"loss": "{:.3f}".format(self.trainer.avg_loss), "lr": self.lr_scheduler.get_last_lr()[-1]}

    return tqdm_dict

  def train_dataloader(self):
    train_dataset = get_dataset(tokenizer=self.tokenizer, type_path="train", args=self.args)
    
    dataloader = DataLoader(train_dataset, batch_size=self.args.train_batch_size, drop_last=True, shuffle=True, num_workers=4)
    t_total = (
        (len(dataloader.dataset) // (self.args.train_batch_size * max(1, self.args.n_gpu)))
        // self.args.gradient_accumulation_steps
        * float(self.args.num_train_epochs)
    )
    scheduler = get_linear_schedule_with_warmup(
        self.opt, num_warmup_steps=self.args.warmup_steps, num_training_steps=t_total
    )
    self.lr_scheduler = scheduler
    return dataloader

  def val_dataloader(self):
    val_dataset = get_dataset(tokenizer=self.tokenizer, type_path="val", args=self.args)
    return DataLoader(val_dataset, batch_size=self.args.eval_batch_size, num_workers=4)
  

logger = logging.getLogger(__name__)

class LoggingCallback(pl.Callback):
  def on_validation_end(self, trainer, pl_module):
    logger.info("***** Validation results *****")
    if pl_module.is_logger():
      metrics = trainer.callback_metrics
      # Log results
      for key in sorted(metrics):
        if key not in ["log", "progress_bar"]:
          logger.info("{} = {}\n".format(key, str(metrics[key])))

  def on_test_end(self, trainer, pl_module):
    logger.info("***** Test results *****")

    if pl_module.is_logger():
      metrics = trainer.callback_metrics

      # Log and save results to file
      output_test_results_file = os.path.join(pl_module.hparams.output_dir, "test_results.txt")
      with open(output_test_results_file, "w") as writer:
        for key in sorted(metrics):
          if key not in ["log", "progress_bar"]:
            logger.info("{} = {}\n".format(key, str(metrics[key])))
            writer.write("{} = {}\n".format(key, str(metrics[key])))


args_dict = dict(
    data_dir="", # path for data files
    output_dir="", 
    model_name_or_path='laituan245/molt5-large-caption2smiles',
    tokenizer_name_or_path='laituan245/molt5-large-caption2smiles',
    max_seq_length=256,
    learning_rate=3e-1,
    weight_decay=0.0,
    adam_epsilon=1e-8,
    warmup_steps=0,
    train_batch_size=4,
    eval_batch_size=8,
    num_train_epochs=2,
    gradient_accumulation_steps=16,
    n_gpu=1,
    #early_stop_callback=False,
    fp_16=False, # if you want to enable 16-bit training then install apex and set this to true
    opt_level='O1', # you can find out more on optimisation levels here https://nvidia.github.io/apex/amp.html#opt-levels-and-properties
    max_grad_norm=1.0, # if you enable 16-bit training then set this to a sensible value, 0.5 is a good default
    seed=42,
)


class MolDataset(Dataset):
  def __init__(self, tokenizer, data_dir, type_path,  max_len=512):
    self.file_path = os.path.join(data_dir, type_path)
    self.file_path = self.file_path + ".txt"
    
    self.max_len = max_len
    self.tokenizer = tokenizer
    self.inputs = []
    self.targets = []

    self._build()
  
  def __len__(self):
    return len(self.inputs)
  
  def __getitem__(self, index):
    
    source_ids = self.inputs[index]["input_ids"].squeeze()
    target_ids = self.targets[index]["input_ids"].squeeze()
    

    src_mask    = self.inputs[index]["attention_mask"].squeeze()  # might need to squeeze
    target_mask = self.targets[index]["attention_mask"].squeeze()  # might need to squeeze
    
    return {"source_ids": source_ids, "source_mask": src_mask, "target_ids": target_ids, "target_mask": target_mask}
  
  def _build(self):
    self._build_examples_from_files(self.file_path)

  def _build_examples_from_files(self, path):
    df = pd.read_csv(path, sep="\t")
    dataframe = df[["description", "SMILES"]]
    
    for i, line in dataframe.iterrows():

      if i == int(sys.argv[4]):
        break
      
       # tokenize inputs
      print(i, line)
      tokenized_inputs = self.tokenizer.batch_encode_plus(
          [line['description']], max_length=self.max_len, pad_to_max_length=True, return_tensors="pt"
      )
       
      tokenized_inputs['input_ids'][0][8:] = tokenized_inputs['input_ids'][0][7:-1].clone()
      tokenized_inputs['input_ids'][0][7] = 32101 + + i
      tokenized_inputs['attention_mask'][0][8:] = tokenized_inputs['attention_mask'][0][7:-1].clone()
      tokenized_inputs['attention_mask'][0][7] = 1
      
      tokenized_targets = self.tokenizer.batch_encode_plus(
          [line['SMILES']], max_length=self.max_len, pad_to_max_length=True, return_tensors="pt"
      )
      
      self.inputs.append(tokenized_inputs)
      self.targets.append(tokenized_targets)
    self.inputs = self.inputs *5
    self.targets = self.targets *5
    
    
      
    


args_dict.update({'data_dir': sys.argv[1], 'output_dir': sys.argv[2], 'num_train_epochs':int(sys.argv[3])})
args = argparse.Namespace(**args_dict)

checkpoint_callback = pl.callbacks.ModelCheckpoint(
    dirpath=args.output_dir, monitor="val_loss", mode="min", save_top_k=2
)

train_params = dict(
    gpus=args.n_gpu,
    max_epochs=args.num_train_epochs,
    precision= 16 if args.fp_16 else 32,
    gradient_clip_val=args.max_grad_norm,
    callbacks=[LoggingCallback(), checkpoint_callback],
)

def get_dataset(tokenizer, type_path, args):
  return MolDataset(tokenizer=tokenizer, data_dir=args.data_dir, type_path=type_path,  max_len=args.max_seq_length)


model = T5FineTuner(args)
trainer = pl.Trainer(**train_params)
trainer.fit(model)

model.model.save_pretrained(f't5_large_{sys.argv[2]}')
model.tokenizer.save_pretrained(f't5_large_{sys.argv[2]}')
