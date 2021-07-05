"""
Analogies class based on responsibly package https://github.com/ResponsiblyAI/responsibly/blob/master/responsibly/we/utils.py,
https://github.com/ResponsiblyAI/responsibly/blob/master/responsibly/we/bias.py 
and gensim most_similar_cosmul implementation.

Added functionalities:
- Works with gensim >= 4.0
- Doesn't raise error when a word is missing from the dictionary, just ignore it
- Faster algebraic operations with numba
- Possibility to use average vectors for representing genders
"""


import numpy as np
import pandas as pd
import gensim
import warnings
from gensim import utils, matutils
from gensim.models import KeyedVectors
from gensim.models import Word2Vec
from numbers import Integral
from six import string_types
from numpy import (
    dot, float32 as REAL, double, array, zeros, vstack,
    ndarray, sum as np_sum, prod, argmax, dtype, ascontiguousarray, frombuffer,
)

from .utils import normalize, calculate_avg_vector, get_seed_vector, get_avg_seed_vector
from .utils import fast_euclidean_dist, fast_cosine_sim
from .data import gendered_neutral_words



GENDER = ['male','female']


class Analogies():
    """
    Performs three different types of analogies:
    - Cosadd: a:? = c:d
        https://www.aclweb.org/anthology/N13-1090/
    - Cosmul: a:? = c:d
        https://www.aclweb.org/anthology/W14-1618/
    - Bolukbasi analogy: a:? = c:?
        https://arxiv.org/abs/1607.06520

    The analogies can be either constrained (returns b ≠ d) or unconstrained 
    (might return b = d).
    Instead of using a single word for each gender, it is possible to automatically
    compute a mean vector between a set of gendered words.


    Parameters
    ----------------
    model: Word embedding model of 'gensim.model.KeyedVectors'.

    Methods
    ----------------
    most_similar: cosadd analogy
    most_similar_cosmul: cosmul analogy
    generate_analogies: Bolukbasi analogy

    """
    
    def __init__(self, model):
        self.model = model

    def most_similar(self, positive=None, negative=None, use_avg_gender=False,
                    positive_gender='female', topn=10, restrict_vocab=10000, 
                    unrestricted=True):
        """
        Find the top-N most similar words.

        Positive words contribute positively towards the similarity,
        negative words negatively.

        This function computes cosine similarity between a simple mean
        of the projection weight vectors of the given words and
        the vectors for each word in the model.
        The function corresponds to the `word-analogy` and `distance`
        scripts in the original word2vec implementation.

        Based on Gensim implementation.


        Parameters
        ----------------
        positive (str/list): List of words that contribute positively. 
            If use_avg_gender=True, give neutral words as positive.
        negative (list): List of words that contribute negatively.
        use_avg_gender (bool): Whether to use the average vector of gendered
            words to represent genders.
        positive_gender (str): Gender that contributes positively, by default the opposite
            gender contributes negatively. Possible values are 'female', 'male'.
            Default: 'female'.
        topn (int): Number of top-N similar words to return.
            Default: 10
        restrict_vocab (int): Optional integer which limits the
            range of vectors which are searched for most-similar values.
            For example, restrict_vocab=10000 would only check the first 10000 word vectors
            in the vocabulary order. (This may be meaningful if you've sorted the vocabulary
            by descending frequency. The vocabulary is already sorted by frequency.)
            Default: 10000
        unrestricted (bool): Whether to restricted the most similar words to be not from
            the positive or negative word list.
            Default: True.
            
        Returns:
        ----------------
        Sequence of (word, similarity).
        """
        
        if use_avg_gender and positive_gender not in GENDER:
            raise ValueError( "Argument 'positive_gender' has an incorrect value: use one among {}".format(GENDER))
            
        if topn is not None and topn < 1:
            return []

        if positive is None:
            positive = []
        if negative is None:
            negative = []

        self.model.fill_norms()

        if (isinstance(positive, string_types)
                and not negative):
            # allow calls like most_similar('dog'),
            # as a shorthand for most_similar(['dog'])
            positive = [positive]

        if ((isinstance(positive, string_types) and negative)
                or (isinstance(negative, string_types) and positive)):
            raise ValueError('If positives and negatives are given, '
                             'both should be lists!')

            
        if use_avg_gender:
            # create average vectors
            if positive_gender == 'female':
                positive.append(calculate_avg_vector(self.model, gendered_neutral_words['female']))
                negative.append(calculate_avg_vector(self.model, gendered_neutral_words['male']))
            elif positive_gender == 'male':
                positive.append(calculate_avg_vector(self.model, gendered_neutral_words['male']))
                negative.append(calculate_avg_vector(self.model, gendered_neutral_words['female']))
                
                
        # add weights for each word, if not already present;
        # default to 1.0 for positive and -1.0 for negative words
        positive = [
            (word, 1.0) if isinstance(word, string_types + (np.ndarray,))
            else word
            for word in positive
        ]
        negative = [
            (word, -1.0) if isinstance(word, string_types + (np.ndarray,))
            else word
            for word in negative
        ]

        # compute the weighted average of all words
        all_words, mean = set(), []
        for word, weight in positive + negative:
            if isinstance(word, np.ndarray):
                mean.append(weight * word)
            else:
                try: 
                    mean.append(weight * self.model.get_vector(word, norm=True))
                    if word in self.model.key_to_index.keys():
                        all_words.add(self.model.key_to_index[word])
                except:
                    pass

        if not mean:
            raise ValueError("Cannot compute similarity with no input.")
        mean = gensim.matutils.unitvec(np.array(mean, dtype=np.float32)
                                       .mean(axis=0)) 


        limited = (self.model.get_normed_vectors() if restrict_vocab is None
                   else self.model.get_normed_vectors()[:restrict_vocab])

        dists = fast_cosine_sim(limited,mean)

        if topn is None:
            return dists
        

        gendered_words = []
        # don't return words used to create the averaged gendered vector
        if use_avg_gender:
            gendered_words = [ self.model.key_to_index[w] for w in
                        gendered_neutral_words['female'] + gendered_neutral_words['male']]
        
        best = gensim.matutils.argsort(dists,
                                       topn=topn + len(all_words) + len(gendered_words),
                                       reverse=True)

        # if not unrestricted, then ignore (don't return)
        # words from the input.
        if unrestricted:
            result = [(self.model.index_to_key[sim], float(dists[sim])) for sim in best 
                      if sim not in gendered_words]
        else:
            result = [(self.model.index_to_key[sim], float(dists[sim])) for sim in best 
                      if sim not in all_words + gendered_words]
        
        

        return result[:topn]
                                 
                                 
                             
                                 
    def most_similar_cosmul(self, positive=None, negative=None, use_avg_gender=True,
                        positive_gender='female', topn=10, unrestricted=True):
        """Find the top-N most similar words, using the multiplicative combination objective,
        proposed by `Omer Levy and Yoav Goldberg "Linguistic Regularities in Sparse and Explicit Word Representations"
        http://www.aclweb.org/anthology/W14-1618. Positive words still contribute positively towards the similarity,
        negative words negatively, but with less susceptibility to one large distance dominating the calculation.
        In the common analogy-solving case, of two positive and one negative examples,
        this method is equivalent to the "3CosMul" objective (equation (4)) of Levy and Goldberg.
        Additional positive or negative examples contribute to the numerator or denominator,
        respectively - a potentially sensible but untested extension of the method.
        With a single positive example, rankings will be the same as in the default
        :meth:`~gensim.models.keyedvectors.KeyedVectors.most_similar`.

        Parameters
        ----------
        positive : list of str, optional
            List of words that contribute positively.
        negative : list of str, optional
            List of words that contribute negatively.
        use_avg_gender (bool): Whether to use the average vector of gendered
            words to represent genders.
        positive_gender (str): Gender that contributes positively, by default the opposite
            gender contributes negatively. Possible values are 'female', 'male'.
            Default: 'female'.
        topn : int or None, optional
            Number of top-N similar words to return, when `topn` is int. When `topn` is None,
            then similarities for all words are returned.

        Returns
        -------
        list of (str, float) or numpy.array
            When `topn` is int, a sequence of (word, similarity) is returned.
            When `topn` is None, then similarities for all words are returned as a
            one-dimensional numpy array with the size of the vocabulary.
        """
        # TODO: Update to better match & share code with most_similar()
        
        if use_avg_gender and positive_gender not in GENDER:
            raise ValueError("Argument 'positive_gender' has an incorrect value: use one among {}".format(GENDER))
            
        if topn is not None and topn < 1:
            return []

        if positive is None:
            positive = []
        if negative is None:
            negative = []

        self.model.fill_norms()

        if (isinstance(positive, string_types)
                and not negative):
            # allow calls like most_similar('dog'),
            # as a shorthand for most_similar(['dog'])
            positive = [positive]

            
        if use_avg_gender:
            # create average vectors
            if positive_gender == 'female':
                positive.append(calculate_avg_vector(self.model, gendered_neutral_words['female']))
                negative.append(calculate_avg_vector(self.model, gendered_neutral_words['male']))
        elif positive_gender == 'male':
                positive.append(calculate_avg_vector(self.model, gendered_neutral_words['male']))
                negative.append(calculate_avg_vector(self.model, gendered_neutral_words['female']))
                
    
        all_words = {
            self.model.get_index(word) for word in positive + negative
            if not isinstance(word, ndarray) and word in self.model.key_to_index
            }
        

        positive = [
            self.model.get_vector(word, norm=True) if isinstance(word, str) else word
            for word in positive
        ]
        
        
        negative = [
            self.model.get_vector(word, norm=True) if isinstance(word, str) else word
            for word in negative
        ]

        if not positive:
            raise ValueError("cannot compute similarity with no input")

        # equation (4) of Levy & Goldberg "Linguistic Regularities...",
        # with distances shifted to [0,1] per footnote (7)
        pos_dists = [((1 + fast_cosine_sim(self.model.vectors, term) / self.model.norms) / 2) for term in positive]
        neg_dists = [((1 + fast_cosine_sim(self.model.vectors, term) / self.model.norms) / 2) for term in negative]
        dists = prod(pos_dists, axis=0) / (prod(neg_dists, axis=0) + 0.000001)

        if not topn:
            return dists
        

        gendered_words = []
        # don't return words used to create the averaged gendered vector
        if use_avg_gender:
            gendered_words = [ self.model.key_to_index[w] for w in
                        gendered_neutral_words['female'] + gendered_neutral_words['male']]
        

        best = matutils.argsort(dists, topn=topn + len(all_words) + len(gendered_words), reverse=True)
        # ignore (don't return) words from the input
        
        
        if unrestricted:
            result = [(self.model.index_to_key[sim], float(dists[sim])) for sim in best 
                      if sim not in gendered_words]
        else:
            result = [(self.model.index_to_key[sim], float(dists[sim])) for sim in best 
                      if sim not in all_words + gendered_words]

        return result[:topn]
    

    
    
    def generate_analogies(self, n_analogies=100, seed=None, use_avg_gender=False,
                            multiple=False, delta=1., restrict_vocab=10000,
                            unrestricted=True):
            """
            Generate analogies based on a seed vector.
            x - y ~ seed vector.
            or a:x::b:y when a-b ~ seed vector.
            The seed vector can be defined by two word ends,
            or by the bias direction.
            ``delta`` is used for semantically coherent. Default vale of 1
            corresponds to an angle <= pi/3.
            There is criticism regarding generating analogies
            when used with `unstricted=False` and not ignoring analogies
            with `match` column equal to `False`.
            Tolga's technique of generating analogies, as implemented in this
            method, is limited inherently to analogies with x != y, which may
            be force "fake" bias analogies.
            See:
            - Nissim, M., van Noord, R., van der Goot, R. (2019).
              `Fair is Better than Sensational: Man is to Doctor
              as Woman is to Doctor https://arxiv.org/abs/1905.09866.
            
            Parameters
            ----------------
            seed (tuple): The definition of the seed vector. A tuple of two word ends.
            n_analogies (int): Number of analogies to generate.
                Default: 100
            use_avg_gender (bool): Whether to use the average vector of gendered
            words to represent genders.
                Default: False
            multiple (bool): Whether to allow multiple appearances of a word
                in the analogies.
                Default: False
            delta (float): Threshold for semantic similarity. The maximal distance 
                between x and y.
                Default: 1
            restrict_vocab (int): Restrict the vocabulary to the top most similar words.
            unrestricted (bool): Whether to validate the generated analogies
                with unrestricted `most_similar`.
                                      
            Returns
            ------------
            Data Frame of analogies (x, y), their distances,
                     and their cosine similarity scores
            """
            # pylint: disable=C0301,R0914,E1136

            if not unrestricted:
                warnings.warn('Not Using unrestricted most_similar '
                              'may introduce fake biased analogies.')

            self.model.fill_norms()
            
            if use_avg_gender:
                positive_end = 'female_avg'
                negative_end = 'male_avg'

                seed_vector = get_avg_seed_vector(self.model)


            else:   
                (seed_vector,
                 positive_end,
                 negative_end) = get_seed_vector(seed, self.model)



            # NB: the vectors are already ordered by their frequency in descending order
            restrict_vocab_vectors = self.model.vectors[:restrict_vocab]

            normalized_vectors = (restrict_vocab_vectors
                                  / np.linalg.norm(restrict_vocab_vectors, axis=1)[:, None])
            pairs_distances = fast_euclidean_dist(normalized_vectors)


            # `pairs_distances` must be not-equal to zero
            # otherwise, x-y will be the zero vector, and every cosine similarity
            # will be equal to zero.
            # This cause to the **limitation** of this method which enforce a not-same
            # words for x and y.
            pairs_mask = (pairs_distances < delta) & (pairs_distances != 0)

            pairs_indices = np.array(np.nonzero(pairs_mask)).T
            x_vectors = np.take(normalized_vectors, pairs_indices[:, 0], axis=0)
            y_vectors = np.take(normalized_vectors, pairs_indices[:, 1], axis=0)


            x_minus_y_vectors = x_vectors - y_vectors
            normalized_x_minus_y_vectors = (x_minus_y_vectors
                                            / np.linalg.norm(x_minus_y_vectors, axis=1)[:, None])

            cos_distances = fast_cosine_sim(normalized_x_minus_y_vectors, seed_vector)

            sorted_cos_distances_indices = np.argsort(cos_distances)[::-1]

            sorted_cos_distances_indices_iter = iter(sorted_cos_distances_indices)

            analogies = []
            generated_words_x = set()
            generated_words_y = set()

            while len(analogies) < n_analogies:
                cos_distance_index = next(sorted_cos_distances_indices_iter)
                pairs_index = pairs_indices[cos_distance_index]
                word_x, word_y = [self.model.index_to_key[index]
                                  for index in pairs_index]

                if multiple or (not multiple
                                and (word_x not in generated_words_x
                                     and word_y not in generated_words_y)):

                    analogy = ({positive_end: word_x,
                                negative_end: word_y,
                                'score': cos_distances[cos_distance_index],
                                'distance': pairs_distances[tuple(pairs_index)]})

                    generated_words_x.add(word_x)
                    generated_words_y.add(word_y)

                    if unrestricted:
                        most_x = next(word
                                      for word, _ in self.most_similar(
                                                                  [word_y, positive_end],
                                                                  [negative_end]))
                        most_y = next(word
                                      for word, _ in self.most_similar(
                                                                  [word_x, negative_end],
                                                                  [positive_end]))

                        analogy['most_x'] = most_x
                        analogy['most_y'] = most_y
                        analogy['match'] = ((word_x == most_x)
                                            and (word_y == most_y))

                    analogies.append(analogy)

            df = pd.DataFrame(analogies)

            columns = [positive_end, negative_end, 'distance', 'score']

            if unrestricted:
                columns.extend(['most_x', 'most_y', 'match'])

            df = df[columns]

            return df
