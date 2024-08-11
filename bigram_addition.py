#--------------------------------------------------------------
#         Title:    bigram.py
#
#   Description:    A simple bigram language model
#
#        Author:    Josh Clark
#
#       Created:    11/08/2024
#
#  Last Updated:    11/08/2024
#
#--------------------------------------------------------------

#--------------------------------------------------------------
# Import relevant packages
#--------------------------------------------------------------
import torch                            # Import PyTorch
import torch.nn as nn                   # Import PyTorch neural network library
from torch.nn import functional as F    # Import functional
import random                           # Import random


#--------------------------------------------------------------
# START OF USER EDIT SECTION
#--------------------------------------------------------------
# Set Hyper Parameters
batch_size = 32                                             # How many examples to process in parallel
block_size = 8                                              # Maximum context length for predictions
max_iters = 200000                                          # Maximum training iterations
eval_interval = 300                                         # Train and Val loss to be evaluated every #eval_interval steps
learning_rate = 1e-3                                        # Learning rate for optimization
device = 'cuda' if torch.cuda.is_available() else 'cpu'     # Run on GPU if available
eval_iters = 200                                            # Number of batches to compute average loss over in estimate_loss()

# Set number of tokens to generate
max_tokens = 100

# Define max and min numbers to sum with
max_int = 1000
min_int = -1000

# Define number of examples
num_examples = 100000

# Set input path for training data
input_path = r"C:\Users\josha\OneDrive\Attachments\Documents\Python\Machine Learning\Andrej Karpathy Lectures\bigram\02. Data\tiny_shakespeare.txt"
#--------------------------------------------------------------
# END OF USER EDIT SECTION
#--------------------------------------------------------------

#--------------------------------------------------------------
# Create Training Data
#--------------------------------------------------------------
# Define #num_examples groups of 2 numbers to be added
nums_to_add = torch.randint(min_int, max_int, (num_examples,2))

# Compute the sum of each row (along dimension 1)
sums = nums_to_add.sum(dim=1, keepdim=True)

# Concatenate the sums as a new column
all_nums = torch.cat((nums_to_add, sums), dim=1)

# Initialize an empty string to hold the final result
final_string = ""

# Iterate through each row in new_tensor
for row in all_nums:
    # Extract the elements
    num1, num2, result = row.tolist()
    
    # Create the string in the format 'num1 + num2 = result\n'
    row_string = f'{int(num1)} + {int(num2)} = {int(result)}\n'
    
    # Append the string to the final string
    final_string += row_string


#--------------------------------------------------------------
# Tokenise the Data
#--------------------------------------------------------------
# Set seed for reproducibility
torch.manual_seed(1337)

# Get list of all unique characters in the dataset
chars = sorted(list(set(final_string)))
vocab_size = len(chars)

# Create a mapping from characters to integers
stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for i,ch in enumerate(chars)}

# Build the encoder and decoder
encode = lambda s: [stoi[c] for c in s]             # Encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l])    # Decoder: take a list of integers, output a string

# Tokenise the dataset
#   Our model is a character-level model
#   hence our tokens are individual characters
data = torch.tensor(encode(final_string), dtype=torch.long)


#--------------------------------------------------------------
# Build the Splits
#--------------------------------------------------------------
# Set split proportions
train_pct = 0.9
val_pct = 0.1
test_pct = 0.0      # We wont define a test set for now

# Calculate split indexes
n1 = int(train_pct * data.size()[0])
n2 = int((train_pct + val_pct) * data.size()[0])

# Create splits
train_data = data[:n1]
val_data = data[n1:n2]
test_data = data[n2:]

#--------------------------------------------------------------
# Define function to generate batches
#--------------------------------------------------------------
def get_batch(split):
    # Define data split to pick from
    if split == 'train':
        data = train_data
    elif split == 'val':
        data = val_data
    elif split == 'test':
        data = test_data
    else:
        raise ValueError(f"Invalid split value: '{split}'. Expected one of: 'train', 'val', 'test'.")
    
    # Generate #batch_size random indexes
    ix = torch.randint(data.size()[0], (batch_size,)) 

    # Gather inputs (x) and targets (y)
    x = torch.stack([data[i, :2] for i in ix])
    y = torch.stack([data[i, -1] for i in ix])
    return x, y


#--------------------------------------------------------------
# Define function to estimate the loss
#--------------------------------------------------------------
# Disable gradient tracking
@torch.no_grad()

# Define function to estimate the loss over multiple batches
def estimate_loss():
    # Initialise output
    out = {}

    # Set model in evaluation mode
    model.eval()

    # Compute avg loss for both train and val splits
    for split in ['train', 'val']:
        # Initialise tensor to store losses
        losses = torch.zeros(eval_iters)

        # Calculate loss for #eval_iters batches (to reduce noise)
        for k in range(eval_iters):
            X, Y = get_batch(split)         # Define inputs and targets
            logits, loss = model(X, Y)      # Calculate loss
            losses[k] = loss.item()         # Add loss of current batch to storage tensor
        
        # Compute the average loss over all batches for current split
        out[split] = losses.mean()
    
    # Reset the model back to training mode
    model.train()

    # Return avg losses for train and val splits
    return out


#--------------------------------------------------------------
# Build Bigram Language Model
#--------------------------------------------------------------
class BigramLanguageModel(nn.Module):

    # Initialise the class
    def __init__(self, vocab_size):
        super().__init__()                                                  # Initialise using parent class nn.Module
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)   # Build embedding table

    # Define forward pass
    #   idx, targets are tensors of size (B,T)
    def forward(self, idx, targets=None):
        # Select a row from embedding table for each index of each batch
        logits = self.token_embedding_table(idx)
    
        # Calculate loss
        if targets is None:
            loss = None
        else:
            # Reshape logits and targets to be compatible with F.cross_entropy
            B, T, C = logits.shape
            logits = logits.view(B*T, C)    # (B,T,C) => (B*T,C)
            targets = targets.view(B*T)     # (B,T)   => (B*T)

            # Evaluate loss using negative log-likelihood
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    # Define generation from the model
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            # Get the predictions from the forward pass
            logits, loss = self(idx)

            # Select last character's embedding for each example in the batch
            logits = logits[:, -1, :]                           # (B,T,C) => (B,C)

            # Apply softmax to get probability distributions
            probs = F.softmax(logits, dim=-1)                   # (B,C)

            # Sample from each distribution 
            idx_next = torch.multinomial(probs, num_samples=1)  # (B,1)

            # Append sampled indexes to running sequence
            idx = torch.cat((idx, idx_next), dim=1)             # (B,T+1)
        return idx


#--------------------------------------------------------------
# Create Instance of the Model
#--------------------------------------------------------------
model = BigramLanguageModel(vocab_size)
m = model.to(device)


#--------------------------------------------------------------
# Training the Model
#--------------------------------------------------------------
# Create a PyTorch Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# Print at start of training
print("-----------------------------------------------------------")
print("MODEL TRAINING")
print("-----------------------------------------------------------")
print("Model Hyper Parameters:")
print(f"     batch_size = {batch_size}")
print(f"     block_size = {block_size}")
print(f"     max_iters = {max_iters}")                                
print(f"     eval_interval = {eval_interval}")
print(f"     learning_rate = {learning_rate}")
print(f"     device = {device}")
print(f"     eval_iters = {eval_iters}")
print("-----------------------------------------------------------")

# Create a simple training loop
for iter in range(max_iters):

    # Every once in a while evaluate the loss of train and val sets
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"Step {iter}: train loss = {losses['train']:.4f}, val loss = {losses['val']:.4f}")

    # Sample a batch of data
    xb, yb = get_batch('train')

    # Evaluate the loss
    logits, loss = m(xb, yb)                # Evaluate the loss
    optimizer.zero_grad(set_to_none=True)   # Zero out gradients from prev step
    loss.backward()                         # Backprop
    optimizer.step()                        # Update parameters

# Display the loss at the end of training
print("-----------------------------------------------------------")
print("-----------------------------------------------------------")
print(f"Loss after {max_iters} training steps: {loss.item()}")
print("-----------------------------------------------------------")
print("-----------------------------------------------------------\n")


#--------------------------------------------------------------
# Generate from the Model
#--------------------------------------------------------------
# Print at start of generation
print("-----------------------------------------------------------")
print("GENERATING FROM THE MODEL")
print("-----------------------------------------------------------")

# Generation
input_idx = torch.zeros((1,1), dtype=torch.long, device=device)             # Start with just idx=0 (new line char)
gen_idxs = m.generate(input_idx, max_new_tokens=max_tokens)[0].tolist()     # Generate next 100 char-indexes
decoded_gen = decode(gen_idxs)                                              # Decode indexes to chars
print(f"Generated text: {decoded_gen}")


