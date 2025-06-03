import torch

def generate_continuation(
    model,
    initial_seq,
    top_k=3,
    device='cpu'
):
    model.eval() #sets model to evaluation mode
    sequence = initial_seq.copy() #creates a copy of the initial sequence
    sequence_tensor = torch.tensor(sequence, dtype=torch.long, device=device).unsqueeze(0)  # shape: [1, seq_len]
    #creates a tensor of the initial sequence, unsqueezes it to add a batch dimension, and moves it to the specified device

    with torch.no_grad():
        logits = model(sequence_tensor)  # shape: [1, seq_len, vocab_size]
        last_token_logits = logits[0, -1]  # selects logits for the last token in the sequence

        probs = torch.softmax(last_token_logits, dim=-1) #converts logits to probabilities
        top_probs, top_indices = torch.topk(probs, top_k) #gets the top k probabilities and indices
        sequences = []
    for i in range(top_k):
        next_token = top_indices[i].item() #gets the next token
        prob = top_probs[i].item() #gets the probability of the next token
        new_seq = sequence + [next_token] #creates a new sequence with the next token
        sequences.append({
            'sequence': new_seq, #new sequence with the next token
            'token': next_token, #next token
            'probability': prob #probability of the next token
        })
    return sequences
