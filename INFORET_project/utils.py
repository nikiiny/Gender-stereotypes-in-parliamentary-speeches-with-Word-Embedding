import warnings
import math
from numba import jit
from fastdist import fastdist
import numba as nb
import gensim
import numpy as np
import pandas as pd
from six import string_types
from gensim.models import KeyedVectors
from gensim.models import Word2Vec

from .data import gendered_neutral_words


def return_min_length(list1, list2):
    """If the input lists have different lenghts, discard the last
    elements of the longer list to obtain lists with the same length."""

    if len(list1) != len(list2):
        min_len = min(len(list1), len(list2))
        return list1[:min_len], list2[:min_len]

    else:
        return list1, list2


def drop_missing_keys(model, keys):
    """Returns list of words actually contained in the model."""
    return [key for key in keys if key in model.index_to_key]


def cosine_similarities_by_words(model, word, words):
    """Compute cosine similarities between a word and a set of other words."""

    assert isinstance(word, string_types), \
        'The arguemnt `word` should be a string.'
    assert not isinstance(words, string_types), \
        'The argument `words` should not be a string.'

    vec = model[word]
    vecs = [model[w] for w in words]
    return model.cosine_similarities(vec, vecs)



def normalize(v):
    """Normalize a 1-D vector."""
    if v.ndim != 1:
        raise ValueError('v should be 1-D, {}-D was given'.format(
            v.ndim))
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


def calculate_avg_vector(model, list_of_words):
        """Calculate a new vector as the average of the vectors given
        as input."""
        vec = []
        for word in list_of_words:
            try:
                vec.append(model.get_vector(word, norm=True))     
            except:
                pass
        return np.array(vec).mean(axis=0)
    
    
def get_avg_seed_vector(model):
    """get seed when using averaged vectors"""
    female_similar = calculate_avg_vector(model, gendered_neutral_words['female'])
    male_similar = calculate_avg_vector(model, gendered_neutral_words['male'])
    
    seed_vector = normalize(female_similar - male_similar)
    return seed_vector


@jit(nopython=True)
def fast_euclidean_dist(M):
    """Fast implementation with numba of Matrix pairwise euclidian distance."""
    return fastdist.matrix_pairwise_distance(M, fastdist.euclidean, "euclidean", return_matrix=True)



@jit(nopython=True)
def fast_cosine_sim(u,v):
    """Fast implementation with numba of cosine similarity with normalization."""
    u_norm = np.linalg.norm(u)
    v_norm = np.linalg.norm(v)
    similarity = u @ v / (u_norm * v_norm)
    return similarity


def get_seed_vector(seed, model):
    positive_end, negative_end = seed
    seed_vector = normalize(model[positive_end]
                                - model[negative_end])

    return seed_vector, positive_end, negative_end




def similar_to_avg_vector(model, words_list, topn=20, verbose=True):
    """
    Returns the most similar words to a vector created by averaging a group
    of words.
    print plots?
    """   
    avg_vector = calculate_avg_vector(model, words_list)
    most_similar = model.similar_by_vector(avg_vector, topn=20+len(words_list))

    most_similar_return = []
    for word in most_similar:
        if word[0] not in words_list:
            most_similar_return.append(word)

    if verbose:
        display(most_similar_return[:topn])

    return most_similar_return[:topn]



