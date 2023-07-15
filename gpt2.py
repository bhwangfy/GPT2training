import torch  
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config  
from transformers import AdamW, get_linear_schedule_with_warmup  
from datasets import load_dataset  
from transformers import DataCollatorWithPadding
import argparse
from torch.utils.data import DataLoader
# calculate the proper batch size
# available_memory = 38 * (1024 ** 3)  # 38GB in bytes  
# model_memory = 124439808 * 4 * 2  
# input_memory = 1024 * 4  
# optimizer_memory = model_memory  
  
# batch_size = available_memory / (model_memory + input_memory + optimizer_memory)  
# print(batch_size)

# Read the args
parser = argparse.ArgumentParser(description="Training GPT2")  

parser.add_argument('--batch-size', type=int, default=20, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train (default: 100)')
parser.add_argument('--decay-epoch', type=int, default=150, metavar='N',
                    help='number of epochs to train (default: 100)')
parser.add_argument('--lr', type=float, default=5e-4, metavar='LR',
                    help='learning rate (default: 5e-4)')
parser.add_argument('--blocks', type=int, default=18, metavar='BS',
                    help='number of residual blocks in each stage (default: 18)')
parser.add_argument('--precision', type=str, default='FP32', choices=['FP32', 'BFP16', 'FP16'],
                    help='precision choice during training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--save-model', action='store_true', default=False,
                    help='For Saving the current Model')
parser.add_argument('--switch', type=int, default=190, metavar='N',
                    help='number of epochs to switch (default: 190)')
args = parser.parse_args()

# Set the hyperparameterss=
epochs = args.epochs
decay_epoch =args.decay_epoch
batch_size = args.batch_size
learning_rate =  args.lr
log_interval = args.log_interval
switch = args.switch


# Load the dataset, use 1% dataset
dataset_wiki = load_dataset('wikitext', 'wikitext-2-raw-v1')  
train_dataset_wiki = dataset_wiki['train']  
train_dataset_wiki = train_dataset_wiki.train_test_split(test_size=0.99)['train']



dataset_ptb = load_dataset('ptb_text_only', 'penn_treebank')  
train_dataset_ptb = dataset_ptb['train'] 
train_dataset_ptb = train_dataset_ptb.train_test_split(test_size=0.99)['train']

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')  
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

#Define the padding token, tokenizer of gpt2 does not have this
tokenizer.pad_token = tokenizer.eos_token

# Tokenize the dataset  
def tokenize_function(example):  
    text = example.get("text", None)  
    if text is None:  
         text = example["sentence"]  
    return tokenizer(text, return_tensors="pt", truncation=True, max_length=1024, padding='max_length')  
  
tokenized_train_dataset_wiki = train_dataset_wiki.map(tokenize_function, batched=True)  
tokenized_train_dataset_wiki.set_format("torch", columns=["input_ids"]) 

tokenized_train_dataset_ptb = train_dataset_ptb.map(tokenize_function, batched=True)  
tokenized_train_dataset_ptb.set_format("torch", columns=["input_ids"])  

# Load the model
custom_depth = 8  # Replace with your desired depth  
custom_head_number = 12  # Replace with your desired head number  
config = GPT2Config.from_pretrained("gpt2", output_hidden_states=True, n_layer=custom_depth, n_head=custom_head_number)  
model = GPT2LMHeadModel(config=config)  
 

# num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)  
# print(f"Number of trainable parameters: {num_parameters}")  


# Move the model to the GPU if available  
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
model.to(device)  
  
# Training parameters  


total_steps = len(tokenized_train_dataset_wiki) * epochs // batch_size  
optimizer = AdamW(model.parameters(), lr=learning_rate)  

  
# Training loop  
 
for epoch in range(epochs):  
    if epoch >= switch:  
        tokenized_train_dataset = tokenized_train_dataset_ptb  
        train_dataloader = DataLoader(tokenized_train_dataset, batch_size=batch_size, shuffle=True)  
    else:  
        tokenized_train_dataset = tokenized_train_dataset_wiki  
        train_dataloader = DataLoader(tokenized_train_dataset, batch_size=batch_size, shuffle=True) 
    
    # Decay the learning rate
    if epoch == decay_epoch:   
        for param_group in optimizer.param_groups:  
            param_group['lr'] /= 10  
    
    model.train()
    # Initialize the loss and gradient norm
    gradient_epoch = 0.
    loss_epoch = 0.
    print(f"Epoch {epoch + 1}/{epochs}")  
    for i, batch in enumerate(train_dataloader):  
        optimizer.zero_grad()  
        inputs = batch['input_ids'].to(device)  
        labels = inputs.clone()  
        outputs = model(inputs, labels=labels)  
        loss = outputs.loss  
        loss.backward()  
        optimizer.step()  
        # scheduler.step()  
  
        if i % log_interval == 0:  
            print(f"Batch {i}: Loss {loss.item()}")  
    model.eval()
    for i, batch in enumerate(tokenized_train_dataset_wiki):  
        inputs = batch['input_ids'].to(device)  
        labels = inputs.clone()  
        outputs = model(inputs, labels=labels)
        loss = outputs.loss
        # loss.backward() 
        # current_gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm= float('inf'))
        # print(current_gradient_norm)
        # gradient_epoch += current_gradient_norm
        loss_epoch += outputs.loss.item()   
    print(f"Sum Loss {loss_epoch}")  
    # print(f"Sum Gradient Norm: {gradient_epoch}")
# Save the model  
model.save_pretrained("trained_model")  
tokenizer.save_pretrained("trained_model")  
