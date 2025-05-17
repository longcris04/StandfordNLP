from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import load_dataset
import torch
import json
import torch.nn.functional as F
import random
import re
from collections import Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load model and tokenizer
model_name = "sshleifer/distilbart-cnn-12-6"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
# load model to gpu for faster inference
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"Using device: {device}")
model.eval()


    # Upload cnn_dm_200.json file

    # Load 200-sample dataset
with open("cnn_dm_200.json", "r") as f:
    dataset = json.load(f)

print("Because the limited computation resource in google collab, I only use first 100 samples from the dataset to run experiment.")
dataset = dataset[:100]

print(f"Loaded {len(dataset)} samples.")
print("\nExample article:")
print(dataset[0]["article"])
print("\nReference summary:")
print(dataset[0]["highlights"])



def greedy_decode(input_ids, max_length=45):
    """
    Perform greedy decoding from the model using logits.

    Args:
        input_ids (torch.Tensor): Tokenized input tensor of shape [1, seq_len].
        max_length (int): Maximum length of the generated sequence.

    Returns:
        str: Decoded summary text.
    """
    # TODO: Implement Greedy Decoding from scratch
    input_ids = input_ids.to(device) # load model to device
    decoder_input_ids = torch.tensor([[model.config.decoder_start_token_id]]).to(device) # take the start token for decoding
    eos_token_id = model.config.eos_token_id # take the EOS token id from the vocab

    for _ in range(max_length):

        with torch.no_grad():
            output = model(input_ids=input_ids, decoder_input_ids=decoder_input_ids)
        proba_output = torch.softmax(output.logits[:,-1,:],axis=-1) # take the last token logits and apply softmax to get the probability of each token
        id = torch.tensor([[torch.argmax(proba_output, axis=-1)]]).to(device) # greedy approach to take the token_id with largest probability
        decoder_input_ids = torch.cat([decoder_input_ids,id],axis=-1) # append the decoded token to the generated token sequence

        if id.item() == eos_token_id: # If the model generate an EOS token, the decoding process should be stop
            break

    summary = tokenizer.decode(decoder_input_ids[0], skip_special_tokens=True) # convert the generated token sequence to string

    return summary


def top_k_decode(input_ids, k=5, max_length=45, temperature=1.0):
    """
    Perform Top-k sampling decoding from the model.

    Args:
        input_ids (torch.Tensor): Tokenized input tensor of shape [1, seq_len].
        k (int): Number of top tokens to sample from.
        max_length (int): Maximum length of the generated sequence.
        temperature (float): Softmax temperature for sampling.

    Returns:
        str: Decoded summary text.
    """
    # TODO: Implement Top-k Sampling Decoding
    input_ids = input_ids.to(device)
    decoder_input_ids = torch.tensor([[model.config.decoder_start_token_id]]).to(device) # take the start token for decoding
    eos_token_id = model.config.eos_token_id # take the EOS token id from the vocab

    for _ in range(max_length):
      with torch.no_grad():
        output = model(input_ids=input_ids, decoder_input_ids=decoder_input_ids)
      output_proba = torch.softmax(output.logits[:,-1,:]/temperature,axis=-1) # convert logits to probability, devide by the temperature to control the randomness and shape of probability distribution
      values, indices = torch.topk(output_proba, k=k, axis=-1) # take top k tokens with largest probability
      values = values/ torch.sum(values,axis=-1,keepdim=True).item() # redistribute probability of selected tokens, made them summing up to 1
      local_index = torch.multinomial(values[0],num_samples=1) # sampling from redistributed probability distribution
      id = indices[:,local_index] # take the token id of the sampled token
      decoder_input_ids = torch.cat([decoder_input_ids,id],axis=-1) # add the sampled token to the generated token sequence

      if id.item() == eos_token_id: # If the model generate an EOS token, the decoding process should be stop
        break

    summary = tokenizer.decode(decoder_input_ids[0], skip_special_tokens=True) # convert the generated token sequence to string
    return summary


def top_p_decode(input_ids, p=0.9, max_length=45, temperature=1.0):
    """
    Perform Top-p (nucleus) sampling decoding from the model.

    Args:
        input_ids (torch.Tensor): Tokenized input tensor of shape [1, seq_len].
        p (float): Cumulative probability threshold for sampling.
        max_length (int): Maximum length of the generated sequence.
        temperature (float): Softmax temperature for sampling.

    Returns:
        str: Decoded summary text.
    """
    # TODO: Implement Top-p Sampling Decoding

    input_ids = input_ids.to(device)
    decoder_input_ids = torch.tensor([[model.config.decoder_start_token_id]]).to(device) # take the start token for decoding
    eos_token_id = model.config.eos_token_id # take the EOS token id from the vocab

    for _ in range(max_length):
      with torch.no_grad():
        output = model(input_ids=input_ids, decoder_input_ids=decoder_input_ids)
      output_proba = torch.softmax(output.logits[:,-1,:]/temperature,axis=-1)[0] # convert logits to probability, devide by the temperature to control the randomness and shape of probability distribution
      sorted_probs, sorted_indices = torch.sort(output_proba, descending=True) # sort the probabilites and their corresponding token ids
      cumulative_probs = torch.cumsum(sorted_probs, dim=0) # calculate the cumulative probability
      nucleus_mask = cumulative_probs <= p # create a mask for the tokens that are below the threshold
      nucleus_mask[max((cumulative_probs > p).nonzero(as_tuple=True)[0][0], 0)] = True # Ensure we include at least one token after passing threshold
      nucleus_probs = sorted_probs[nucleus_mask] # filter the probabilities based on the mask
      nucleus_indices = sorted_indices[nucleus_mask] # filter the token ids based on the mask
      nucleus_probs = nucleus_probs/torch.sum(nucleus_probs) # redistribute the probabilities of the selected tokens
      local_index = torch.multinomial(nucleus_probs,num_samples=1) # sampling from redistributed probability distribution
      id = torch.tensor([[nucleus_indices[local_index]]]).to(device)
      decoder_input_ids = torch.cat([decoder_input_ids,id],axis=-1)

      if id.item() == eos_token_id: # If the model generate an EOS token, the decoding process should be stop
        break


    summary = tokenizer.decode(decoder_input_ids[0], skip_special_tokens=True)
    return summary

def beam_search_decode(input_ids, beam_size=2, max_length=45):
    """
    Perform beam search decoding from the model.

    Args:
        input_ids (torch.Tensor): Tokenized input tensor of shape [1, seq_len].
        beam_size (int): Number of beams to explore.
        max_length (int): Maximum length of the generated sequence.

    Returns:
        str: Decoded summary text.
    """
    # TODO: Implement Beam Search Decoding
    input_ids = input_ids.to(device)
    decoder_input_ids = torch.tensor([[model.config.decoder_start_token_id]]).to(device) # take the start token for decoding
    eos_token_id = model.config.eos_token_id # take the EOS token id from the vocab
    vocab_size = model.config.vocab_size # take the vocab size from the model config

    top_beam = [] # to store top k beams with highest probability at each step
    with torch.no_grad():
      output = model(input_ids=input_ids, decoder_input_ids=decoder_input_ids)

    output_proba = torch.softmax(output.logits[:,-1,:],axis=-1)
    probas, indices = torch.topk(output_proba,k=beam_size,axis=-1) # select top k tokens with largest probability, called beams in the first step
    for index,proba in zip(indices[0],probas[0]):
      top_beam.append((torch.tensor([[decoder_input_ids,index]]),proba.item())) # append top k beams to the list to start the beam search process

    for _ in range(max_length):
      beam_list = [] # beam list to store all the beams, total number of beams in comparison will be beam_size * vocab_size
      for beam in top_beam:
        with torch.no_grad():
          output = model(input_ids=input_ids, decoder_input_ids=beam[0].to(device)).logits[:,-1,:]
        output_proba = beam[1] * torch.softmax(output,axis=-1) # calculate the cumulative probability by multiplying the probability of the current beam with the conditional probability of the next token
        for idx,local_proba in enumerate(list(output_proba[0])):
          beam_list.append((torch.cat([beam[0],torch.tensor([[idx]])],axis=-1), local_proba.item())) # append each candidate to the beam list, the candidate is the current beam with the next token id appended to it

      sorted_beam_list = sorted(beam_list, key=lambda item: item[1], reverse=True) # sort the beam list based on the cumulated probability for later selection
      top_beam = sorted_beam_list[:beam_size] # select top k beams with largest probability

      if top_beam[0][0][0][-1].item() == eos_token_id:
        break

    summary = tokenizer.decode(top_beam[0][0][0], skip_special_tokens=True)
    return summary


def isBlocked(token_seq: list(), no_repeat_ngram_size: int, next_token: int):
  '''
  This function checks if the next token is blocked by the n-gram repetition blocking rule.
  It iteratively check if the next token is in the last n-gram size of the token sequence.
  '''
  if len(token_seq) < no_repeat_ngram_size:
    return False
  candidate_seq = token_seq[-(no_repeat_ngram_size-1):] + [next_token]

  for i in range(len(token_seq)-(no_repeat_ngram_size-1)):
    ref_seq = token_seq[i: i+ no_repeat_ngram_size]
    if ref_seq == candidate_seq:
      return True
  return False

def beam_search_ngram_block(input_ids, beam_size=2, max_length=45, no_repeat_ngram_size=3):
    """
    Perform beam search decoding with n-gram repetition blocking.

    Args:
        input_ids (torch.Tensor): Tokenized input tensor of shape [1, seq_len].
        beam_size (int): Number of beams to explore.
        max_length (int): Maximum length of the generated sequence.
        no_repeat_ngram_size (int): Size of n-gram to prevent from repeating.

    Returns:
        str: Decoded summary text.
    """
    # TODO: Implement Beam Search with n-gram blocking


    input_ids = input_ids.to(device)
    decoder_input_ids = torch.tensor([[model.config.decoder_start_token_id]]).to(device)
    eos_token_id = model.config.eos_token_id # take the EOS token id from the vocab
    vocab_size = model.config.vocab_size # take the vocab size from the model config

    top_beam = []
    with torch.no_grad():
      output = model(input_ids=input_ids, decoder_input_ids=decoder_input_ids).logits[:,-1,:]
    output_proba = torch.softmax(output,axis=-1)
    probas, indices = torch.topk(output_proba,k=beam_size,axis=-1)
    for index,proba in zip(indices[0],probas[0]):
      top_beam.append((torch.tensor([[decoder_input_ids,index]]),proba.item()))

    for _ in range(max_length):
      beam_list = []
      for beam in top_beam:
        with torch.no_grad():
          output = model(input_ids=input_ids, decoder_input_ids=beam[0].to(device))
        output = output.logits[:,-1,:]
        output_proba = beam[1] * torch.softmax(output,axis=-1)
        for idx,local_proba in enumerate(list(output_proba[0])):
          if not isBlocked(token_seq=beam[0][0].tolist(),no_repeat_ngram_size=no_repeat_ngram_size,next_token=idx): # check if the next token is blocked by the n-gram repetition blocking rule before being added to the beam list
            beam_list.append((torch.cat([beam[0],torch.tensor([[idx]])],axis=-1), local_proba.item()))
      sorted_beam_list = sorted(beam_list, key=lambda item: item[1], reverse=True)
      top_beam = sorted_beam_list[:beam_size]

      if top_beam[0][0][0][-1].item() == eos_token_id:
        break

    summary = tokenizer.decode(top_beam[0][0][0], skip_special_tokens=True)
    return summary




def tokenize(text):
    return re.findall(r'\w+', text.lower())

def compute_rouge_n(reference: str, generated: str, n: int = 1) -> dict:
    """
    Compute ROUGE-N score between reference and generated text.

    Args:
        reference (str): The reference summary.
        generated (str): The generated summary.
        n (int): The n-gram size (e.g., 1 for ROUGE-1).

    Returns:
        dict: Dictionary with 'precision', 'recall', and 'f1' scores.
    """
    ref_tokens = tokenize(reference)
    gen_tokens = tokenize(generated)

    ref_ngrams = Counter([tuple(ref_tokens[i:i+n]) for i in range(len(ref_tokens)-n+1)])
    gen_ngrams = Counter([tuple(gen_tokens[i:i+n]) for i in range(len(gen_tokens)-n+1)])


    overlap = sum((ref_ngrams & gen_ngrams).values())

    precision = overlap / sum(gen_ngrams.values()) if gen_ngrams else 0.0
    recall = overlap / sum(ref_ngrams.values()) if ref_ngrams else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {'precision': precision, 'recall': recall, 'f1': f1}

def lcs(X, Y):
    """
    Helper function to compute the length of Longest Common Subsequence (LCS)
    """
    m, n = len(X), len(Y)
    dp = [[0] * (n+1) for _ in range(m+1)]

    for i in range(m):
        for j in range(n):
            if X[i] == Y[j]:
                dp[i+1][j+1] = dp[i][j] + 1
            else:
                dp[i+1][j+1] = max(dp[i][j+1], dp[i+1][j])
    return dp[m][n]

def compute_rouge_l(reference: str, generated: str) -> dict:
    """
    Compute ROUGE-L (Longest Common Subsequence) score.

    Args:
        reference (str): The ground truth summary.
        generated (str): The generated summary by the model.

    Returns:
        dict: Dictionary with precision, recall, and f1 scores.
    """
    ref_tokens = tokenize(reference)
    gen_tokens = tokenize(generated)

    lcs_len = lcs(ref_tokens, gen_tokens)

    precision = lcs_len / len(gen_tokens) if gen_tokens else 0.0
    recall = lcs_len / len(ref_tokens) if ref_tokens else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {'precision': precision, 'recall': recall, 'f1': f1}

def generate_summary_custom(article_text, strategy="greedy", max_length=45, temperature=1.0, k=5, p=0.9, beam_size=2, no_repeat_ngram_size=3):
    input_ids = tokenizer(article_text, return_tensors="pt", truncation=True, max_length=1024).input_ids
    if strategy == "greedy":
        return greedy_decode(input_ids,max_length=max_length)
    elif strategy == "top_k":
        return top_k_decode(input_ids, k=k, max_length=max_length, temperature=temperature)
    elif strategy == "top_p":
        return top_p_decode(input_ids, p=p, max_length=max_length, temperature=temperature)
    elif strategy == "beam":
        return beam_search_decode(input_ids, beam_size=beam_size, max_length=max_length)
    elif strategy == "beam_block":
        return beam_search_ngram_block(input_ids,beam_size=beam_size, max_length=max_length, no_repeat_ngram_size=no_repeat_ngram_size)
    else:
        raise ValueError("Unknown decoding strategy")


def inference():
    
    
    # Decoding strategies
    strategies = ["greedy", "top_k", "top_p", "beam", "beam_block"]

    # Save evalution results
    results = {s: [] for s in strategies}

    # Evaluation loop
    for i, sample in enumerate(dataset):
        article = sample["article"]
        reference = sample["highlights"]

        input_ids = tokenizer(article, return_tensors="pt", truncation=True, max_length=1024).input_ids

        for strategy in strategies:
            try:
                generated = generate_summary_custom(article, strategy=strategy)
                print(f"here: {generated}")
                rouge1 = compute_rouge_n(reference, generated, n=1)["f1"]
                rouge2 = compute_rouge_n(reference, generated, n=2)["f1"]
                rougel = compute_rouge_l(reference, generated)["f1"]

                results[strategy].append({
                    "rouge1": rouge1,
                    "rouge2": rouge2,
                    "rougeL": rougel
                })
            except Exception as e:
                print(f"[{strategy}] Error on sample {i}: {e}")

    # code to save results
    with open("evaluation_results.json", "w") as f:
        json.dump(results, f, indent=4)
    
    strategies = ["greedy", "top_k", "top_p", "beam", "beam_block"]
    results = json.load(open("evaluation_results.json", "r"))
    summary = {}
    for strategy in strategies:
        if results[strategy]:
            rouge1s = [x["rouge1"] for x in results[strategy]]
            rouge2s = [x["rouge2"] for x in results[strategy]]
            rougels = [x["rougeL"] for x in results[strategy]]

            summary[strategy] = {
                "ROUGE-1": np.mean(rouge1s),
                "ROUGE-2": np.mean(rouge2s),
                "ROUGE-L": np.mean(rougels)
            }

    df = pd.DataFrame(summary).T.sort_values("ROUGE-L", ascending=False)
    df.to_csv("evaluation_summary.csv")
    print(f"finished saving evaluation_summary.csv")
    # display(df)
    print(f"This table result follows the following paramater settings: \n p = 0.9, k = 5, beam size = 2, n-gram size = 3, max_length = 45")
    
        
        # create a dataframe to store the results with different beam sizes
    print(f"Start experiment with different beam sizes")
    beam_sizes = [1,2,3,4]
    beam_sizes_results = pd.DataFrame(columns=beam_sizes)
    max_length = 45
    num_samples = 100
    temperature = 1.0
    no_repeat_ngram_size = 3
    k = 5
    p = 0.9
    for beam in beam_sizes:
        print(f"start testing beam size {beam}")
        for i,sample in enumerate(dataset[:num_samples]):
            
            article = sample["article"]
            reference = sample["highlights"]
            generated = generate_summary_custom(article, strategy="beam", max_length=max_length, temperature=temperature,k=k,p=p, beam_size=beam, no_repeat_ngram_size=no_repeat_ngram_size)
            # generated = "Students and faculty members marched Wednesday afternoon chanting"
            print(f"finished generate sample {i}")
            rouge1 = compute_rouge_n(reference, generated, n=1)["f1"]
            rouge2 = compute_rouge_n(reference, generated, n=2)["f1"]
            rougel = compute_rouge_l(reference, generated)["f1"]
            rouge_mean = (rouge1 + rouge2 + rougel) / 3
            beam_sizes_results.loc[i, beam] = rouge_mean
            beam_sizes_results.to_csv("beam_sizes_results.csv", index=False) # save the results to a csv file
            

    # create a dataframe to store the results with different beam sizes
    print(f"Start experiment with different temperatures")
    temperatures = [0.5,1,2,5]
    temperatures_results = pd.DataFrame(columns=temperatures)
    max_length = 45
    num_samples = 100
    beam_size = 1.0 # doesnot matter for top_p sampling
    no_repeat_ngram_size = 3
    k = 5
    p = 0.5
    for temperature in temperatures:
        print(f"start testing temperature {temperature}")
        for i,sample in enumerate(dataset[:num_samples]):
            
            article = sample["article"]
            reference = sample["highlights"]
            generated = generate_summary_custom(article, strategy="top_p", max_length=max_length, temperature=temperature,k=k,p=p, beam_size=beam_size, no_repeat_ngram_size=no_repeat_ngram_size)
            # generated = "Students and faculty members marched Wednesday afternoon chanting"
            print(f"finished generate sample {i}")
            rouge1 = compute_rouge_n(reference, generated, n=1)["f1"]
            rouge2 = compute_rouge_n(reference, generated, n=2)["f1"]
            rougel = compute_rouge_l(reference, generated)["f1"]
            rouge_mean = (rouge1 + rouge2 + rougel) / 3
            temperatures_results.loc[i, temperature] = rouge_mean
            temperatures_results.to_csv("temperatures_results.csv", index=False) # save the results to a csv file
            


    


if __name__ == "__main__":
    inference()
    
    
    
    