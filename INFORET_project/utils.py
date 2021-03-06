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
import matplotlib.pyplot as plt
import seaborn as sns

from .data import gendered_neutral_words


YEARS = ['1948_1968', '1968_1985', '1985_2000', '2000_2020']
GENDER = ['male','female']


def load_embed_model(years):
    """Load word2vec model of corresponding time period"""
    if not years in YEARS:
        raise ValueError( "Argument 'years' has an incorrect value: use one among {}".format(YEARS))
        
    return KeyedVectors.load(f'we_models/W2V_by_years_{years}')



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



def plot_stackbar_difference_sim(data):

    fig, ax = plt.subplots(figsize=(15, 8))
    ax.bar(x='words', height='distance_female_pos', data=data,
           label='female as positive', color='#ff4d4d')
    ax.bar(x='words', height='distance_male_pos', data=data, bottom='distance_female_pos',
           label='male as positive', color='#5cd65c')

    ax.set_title(f"Difference in average similarities of words returned by analogies to genders",fontsize=40)
    ax.legend()
    plt.setp(ax.get_legend().get_texts(), fontsize='25')
    ax.set_xticklabels(data['words'], fontsize=18, rotation=45, horizontalalignment='right')
    ax.tick_params(axis="y", labelsize=20)

    plt.show()



def plot_barplot_sim(data):

    sns.set_theme(style="darkgrid")
    fig, axes = plt.subplots(1,2, figsize=(30, 8))
    for i,gender in enumerate(GENDER):
        sns.barplot(ax=axes[i], y=data[f"similarity_{gender}_pos"], x=data['words'], hue=data['ref_gender']
                , palette=['#ff80b3', '#4da6ff'])

        plt.suptitle("Similarity of words returned by analogies to male and female", fontsize=40)
        axes[i].set_title(f"Positive gender: {gender}",fontsize=25)
        axes[i].set_xlabel('')
        axes[i].set_ylabel('')
        axes[i].set_xticklabels(axes[i].get_xticklabels(), fontsize=18,rotation=45, horizontalalignment='right')
        plt.setp(axes[i].get_legend().get_texts(), fontsize='25')
        plt.setp(axes[i].get_legend().get_title(),fontsize=25)
        axes[i].tick_params(axis="y", labelsize=20)
        axes[i].set_ylim(0,0.6)

    plt.show()



def to_list(x):
    """Returns a list if the input is a string"""
    if isinstance(x, str):
        return [x]
    else:
        return x


def generate_one_word_forms(word):
    return [word.lower(), word.upper(), word.title()]


def project_reject_vector(v, u):
    """Projecting and rejecting the vector v onto direction u."""
    projected_vector = project_vector(v, u)
    rejected_vector = v - projected_vector
    return projected_vector, rejected_vector


def project_vector(v, u):
    """Projecting the vector v onto direction u."""
    normalize_u = normalize(u)
    return (v @ normalize_u) * normalize_u


def reject_vector(v, u):
    """Rejecting the vector v onto direction u."""
    return v - project_vector(v, u)


def update_word_vector(model, word, new_vector):
    model.vectors[model.key_to_index[word]] = normalize(new_vector)
