# Imports
import numpy as np
from typing import List, Tuple
from numpy.typing import ArrayLike
import random

def sample_seqs(seqs: List[str], labels: List[bool]) -> Tuple[List[str], List[bool]]:
    """
    This function should sample the given sequences to account for class imbalance. 
    Consider this a sampling scheme with replacement.
    
    Args:
        seqs: List[str]
            List of all sequences.
        labels: List[bool]
            List of positive/negative labels

    Returns:
        sampled_seqs: List[str]
            List of sampled sequences which reflect a balanced class size
        sampled_labels: List[bool]
            List of labels for the sampled sequences
    """
    #count number of true and false labels each
    count_true = sum(labels)
    count_false = len(labels) - count_true

    #determine which label is the majority and minority
    if count_true > count_false:
        majority = True
    else:
        majority = False
    minority = not majority 

    minority_seqs = [seq for seq, label in zip(seqs, labels) if label == minority] #get minority sequences after determining minority label
    majority_seqs = [seq for seq, label in zip(seqs, labels) if label == majority]
    
    difference = abs(count_true - count_false) #difference between label counts
    additional_minority_seqs = random.choices(minority_seqs, k=difference) #sampling w replacement from minority

    #combining original seq with added samples
    sampled_seqs = majority_seqs + additional_minority_seqs
    sampled_labels = [majority] * len(majority_seqs) + [minority] * difference

    #shuffle to maintain randomness
    combined = list(zip(sampled_seqs, sampled_labels))
    random.shuffle(combined)

    final_seq, final_label = zip(*combined)

    return final_seq, final_label

def one_hot_encode_seqs(seq_arr: List[str]) -> ArrayLike:
    """
    This function generates a flattened one-hot encoding of a list of DNA sequences
    for use as input into a neural network.

    Args:
        seq_arr: List[str]
            List of sequences to encode.

    Returns:
        encodings: ArrayLike
            Array of encoded sequences, with each encoding 4x as long as the input sequence.
            For example, if we encode:
                A -> [1, 0, 0, 0]
                T -> [0, 1, 0, 0]
                C -> [0, 0, 1, 0]
                G -> [0, 0, 0, 1]
            Then, AGA -> [1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0].
    """
    final_encoded_seqs = [] #list of lists where each list is the one-hot encoding
    A = [1, 0, 0, 0]
    T = [0, 1, 0, 0]
    C = [0, 0, 1, 0]
    G = [0, 0, 0, 1]

    for seq in seq_arr:
        seq_encoding = []
        for base in seq:
            if base=="A":
                seq_encoding.extend(A)
            elif base=="T":
                seq_encoding.extend(T)
            elif base=="C":
                seq_encoding.extend(C)
            elif base=="G":
                seq_encoding.extend(G)
        final_encoded_seqs.append(seq_encoding)
    output = np.array(final_encoded_seqs)
    return output