from collections import defaultdict

def build_vocab(remi_sequences):
    token_to_id = {}
    id_to_token = {}

    unique_tokens = set()
    for seq in remi_sequences:
        unique_tokens.update(seq)

    for idx, token in enumerate(sorted(unique_tokens)):
        token_to_id[token] = idx
        id_to_token[idx] = token

    return token_to_id, id_to_token
def remi_to_token_ids(remi_sequence, token_to_id):
    return [token_to_id[token] for token in remi_sequence if token in token_to_id]
def token_ids_to_remi(token_ids, id_to_token):
    return [id_to_token[token_id] for token_id in token_ids if token_id in id_to_token]
