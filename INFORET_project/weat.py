"""
From responsibly package https://github.com/ResponsiblyAI/responsibly/blob/master/responsibly/we/weat.py
Added functionalities:
- Works with gensim >= 4.0
- Doesn't raise error when a word is missing from the dictionary, just ignore it. Since the 
  algorithm requires the 2 neutral words groups to have the same length, if some words are 
  missing from one group it discards the last words of the other groups, so that the 2 lenghts match.
"""

import copy
import random
import warnings
import numpy as np
import pandas as pd
from mlxtend.evaluate import permutation_test

from .utils import cosine_similarities_by_words, drop_missing_keys, return_min_length


FILTER_BY_OPTIONS = ['model', 'data']
RESULTS_DF_COLUMNS = ['Target words', 'Attrib. words',
                      'Nt', 'Na', 's', 'd', 'p']
PVALUE_METHODS = ['exact', 'approximate']
PVALUE_DEFUALT_METHOD = 'exact'
ORIGINAL_DF_COLUMNS = ['original_' + key for key in ['N', 'd', 'p']]
WEAT_WORD_SETS = ['first_target', 'second_target',
                  'first_attribute', 'second_attribute']
PVALUE_EXACT_WARNING_LEN = 10

RANDOM_STATE = 42


def calc_single_weat(model,
                     first_target, second_target,
                     first_attribute, second_attribute,
                     with_pvalue=True, pvalue_kwargs=None):
    """
    Calc the WEAT result of a word embedding.
    :param model: Word embedding model of 'gensim.model.KeyedVectors'
    :param dict first_target: First target words list and its name
    :param dict second_target: Second target words list and its name
    :param dict first_attribute: First attribute words list and its name
    :param dict second_attribute: Second attribute words list and its name
    :param bool with_pvalue: Whether to calculate the p-value of the
                             WEAT score (might be computationally expensive)
    :return: WEAT result (score, size effect, Nt, Na and p-value)
    """

    if pvalue_kwargs is None:
        pvalue_kwargs = {}

    first_target['words'] = drop_missing_keys(model, first_target['words'])
    second_target['words'] = drop_missing_keys(model, second_target['words'])
    first_attribute['words'] = drop_missing_keys(model, first_attribute['words'])
    second_attribute['words'] = drop_missing_keys(model, second_attribute['words'])

    (first_associations,
     second_associations) = _calc_weat_associations(model,
                                                    first_target['words'],
                                                    second_target['words'],
                                                    first_attribute['words'],
                                                    second_attribute['words'])

    if first_associations and second_associations:
        score = sum(first_associations) - sum(second_associations)
        std_dev = np.std(first_associations + second_associations, ddof=0)
        effect_size = ((np.mean(first_associations) - np.mean(second_associations))
                       / std_dev)

        pvalue = None
        if with_pvalue:
            pvalue = _calc_weat_pvalue(first_associations,
                                       second_associations,
                                       **pvalue_kwargs)
    else:
        score, std_dev, effect_size, pvalue = None, None, None, None

    return {'Target words': '{} vs. {}'.format(first_target['name'],
                                               second_target['name']),
            'Attrib. words': '{} vs. {}'.format(first_attribute['name'],
                                                second_attribute['name']),
            's': score,
            'd': effect_size,
            'p': pvalue,
            'Nt': '{}x2'.format(len(first_target['words'])),
            'Na': '{}x2'.format(len(first_attribute['words']))}





def _calc_weat_pvalue(first_associations, second_associations,
                      method=PVALUE_DEFUALT_METHOD):

    if method not in PVALUE_METHODS:
        raise ValueError('method should be one of {}, {} was given'.format(
            PVALUE_METHODS, method))

    pvalue = permutation_test(first_associations, second_associations,
                              func=lambda x, y: sum(x) - sum(y),
                              method=method,
                              seed=RANDOM_STATE)  # if exact - no meaning
    return pvalue





def _calc_weat_associations(model,
                            first_target_words, second_target_words,
                            first_attribute_words, second_attribute_words):
    
    # Since the algorithm requires the 2 neutral words groups to have the same length, if 
    #some words are missing from one group it discards the last words of the other groups, 
    #so that the 2 lenghts match. 
    first_target_words, second_target_words = return_min_length(first_target_words, second_target_words)
    first_attribute_words, second_attribute_words = return_min_length(first_attribute_words, second_attribute_words)

    assert len(first_target_words) == len(second_target_words)
    assert len(first_attribute_words) == len(second_attribute_words)

    first_associations = _calc_association_all_targets_attributes(model,
                                                                    first_target_words,
                                                                    first_attribute_words,
                                                                    second_attribute_words)

    second_associations = _calc_association_all_targets_attributes(model,
                                                                     second_target_words,
                                                                     first_attribute_words,
                                                                     second_attribute_words)
    return first_associations, second_associations



def _calc_association_all_targets_attributes(model, target_words,
                                             first_attribute_words,
                                             second_attribute_words):

    return [_calc_association_target_attributes(model, target_word,
                                                first_attribute_words,
                                                second_attribute_words)
            for target_word in target_words]




def _calc_association_target_attributes(model, target_word,
                                        first_attribute_words,
                                        second_attribute_words):
    # pylint: disable=line-too-long

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', FutureWarning)

        first_mean = (cosine_similarities_by_words(model,
                                                   target_word,
                                                   first_attribute_words)
                      .mean())

        second_mean = (cosine_similarities_by_words(model,
                                                    target_word,
                                                    second_attribute_words)
                       .mean())

    return first_mean - second_mean