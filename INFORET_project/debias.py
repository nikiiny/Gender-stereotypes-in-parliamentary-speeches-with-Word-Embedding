"""
Debiasing class based on responsibly package 
https://github.com/ResponsiblyAI/responsibly/blob/master/responsibly/we/bias.py

Added functionalities:
- Works with gensim >= 4.0
- Doesn't raise error when a word is missing from the dictionary, just ignore it
- Faster algebraic operations with numba
- When applyng the debiasing algorithm, the gensim object KeyedVectors is returned to allow
	a more flexible use.
"""

import gensim
from tabulate import tabulate 
import copy
import warnings
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from tqdm.auto import tqdm
from itertools import chain

from .utils import (fast_cosine_sim, normalize, generate_one_word_forms, 
	update_word_vector, project_reject_vector, project_vector, reject_vector, 
	drop_missing_keys)
from .data import gendered_neutral_words



DIRECTION_METHODS = ['single', 'sum', 'pca']
FIRST_PC_THRESHOLD = 0.5
MAX_NON_SPECIFIC_EXAMPLES = 1000




class Debias():
	"""
	Measuring and adjusting bias in word embedding by Bolukbasi (2016).
		https://arxiv.org/abs/1607.06520
	
	Parameters
	------------
	model: Word embedding model of 'gensim.model.KeyedVectors'
	definitional_pairs (list): pair of gendered words.
	verbose (bool)
		default: False
	identify_direction (str): how to identify the direction of gender. 
		Possible choices are 'single', 'sum', 'pca'
	"""
	def __init__(self, model, definitional_pairs,
	 	verbose=False, identify_direction='pca', to_normalize=True):

		self.model = model
		self.verbose = verbose
		self.to_normalize = to_normalize

		self._data = copy.deepcopy(gendered_neutral_words['definitional_pairs']) 

		if identify_direction:
			definitional = None
		if identify_direction not in DIRECTION_METHODS:
			raise ValueError( "Argument 'identify_direction' has an incorrect value: use one among {}".format(DIRECTION_METHODS))


		if identify_direction == 'single':
			definitional = definitional_pairs
		elif identify_direction == 'sum':
			definitional = list(zip(*self._data))
		elif identify_direction == 'pca':
			definitional = self._data

		self._identify_direction(definitional_pairs[0], definitional_pairs[1],
                                     definitional,
                                     identify_direction)
       
	def _is_direction_identified(self):
		if self.direction is None:
			raise RuntimeError('The direction was not identified'
                               ' for this {} instance'
                               .format(self.__class__.__name__))


	def hard_debias(self, equality_sets=None):
		"""Debias the word embedding by using hard debiasing method (neutralize + equalize)
			from Bolukbasi.

		Parameters
		------------
		equality_sets (list): List of equality sets in the format [[a1,a2],..],
			for the equalize step. The sets represent the direction.

		Returns
		------------
		Word embedding model of 'gensim.model.KeyedVectors'
        """

        # pylint: disable=W0212
		bias_word_embedding = copy.deepcopy(self)
        
		neutral_words = self.get_neutral_words(equality_sets)

		if self.verbose:
			print('Neutralize...')
		bias_word_embedding._neutralize(neutral_words)

		if self.verbose:
			print('Equalize...')

		assert all(len(equality_set) == 2
			for equality_set in equality_sets)

		equality_sets = self._generate_pair_candidates(equality_sets)

		bias_word_embedding._equalize(equality_sets)

		
		return bias_word_embedding.model



	def get_neutral_words(self, equality_sets):
		"""Generate neutral words as the difference between all the words
		in the model and the equality sets containing the gendered direction"""
		gendered_words = list(chain.from_iterable(equality_sets))
		return [w for w in self.model.index_to_key if w not in gendered_words]


	def _neutralize(self, neutral_words):
		self._is_direction_identified()

		if self.verbose:
			neutral_words_iter = tqdm(neutral_words)
		else:
			neutral_words_iter = iter(neutral_words)

		for word in neutral_words_iter:
			neutralized_vector = reject_vector(self.model[word],
                                               self.direction)
			update_word_vector(self.model, word, neutralized_vector)

		self.model.init_sims(replace=True)




	def _generate_pair_candidates(self, pairs):
        # pylint: disable=line-too-long
		return {(candidate1, candidate2)
				for word1, word2 in pairs
				for candidate1, candidate2 in zip(generate_one_word_forms(word1), 
                                                  generate_one_word_forms(word2))
				if candidate1 in self.model and candidate2 in self.model}




	def _equalize(self, equality_sets):
        # pylint: disable=R0914

		self._is_direction_identified()

		if self.verbose:
			words_data = []

		for equality_set_index, equality_set_words in enumerate(equality_sets):
			try:
				equality_set_vectors = [normalize(self.model[word]) 
	                                    for word in equality_set_words]
				center = np.mean(equality_set_vectors, axis=0)
				(projected_center,
					rejected_center) = project_reject_vector(center, 
	                                                      self.direction)
				scaling = np.sqrt(1 - np.linalg.norm(rejected_center)**2)

				for word, vector in zip(equality_set_words, equality_set_vectors):
					projected_vector = project_vector(vector, self.direction)

					projected_part = normalize(projected_vector - projected_center)

	                # In the code it is different of Bolukbasi
	                # It behaves the same only for equality_sets
	                # with size of 2 (pairs) - not sure!
	                # However, my code is the same as the article
	                # equalized_vector = rejected_center + scaling * self.direction
	                # https://github.com/tolga-b/debiaswe/blob/10277b23e187ee4bd2b6872b507163ef4198686b/debiaswe/debias.py#L36-L37
	                # For pairs, projected_part_vector1 == -projected_part_vector2,
	                # and this is the same as
	                # projected_part_vector1 == self.direction
					equalized_vector = rejected_center + scaling * projected_part

					update_word_vector(self.model, word, equalized_vector) 

					if self.verbose:
						words_data.append({
	                        'equality_set_index': equality_set_index,
	                        'word': word,
	                        'scaling': scaling,
	                        'projected_scalar': vector @ self.direction,
	                        'equalized_projected_scalar': (equalized_vector
	                                                       @ self.direction),
	                    })

			except:
				pass

		if self.verbose:
			print('Equalize Words Data '
                  '(all equal for 1-dim bias space (direction):')
			words_data_df = (pd.DataFrame(words_data)
                             .set_index(['equality_set_index', 'word']))
			print(tabulate(words_data_df, headers='keys'))

		self.model.init_sims(replace=True)



	def _identify_direction(self, positive_end, negative_end,
                            definitional, method='pca'):
		if method not in DIRECTION_METHODS:
			raise ValueError('method should be one of {}, {} was given'.format(
                DIRECTION_METHODS, method))

		if positive_end == negative_end:
			raise ValueError('positive_end and negative_end'
                             'should be different, and not the same "{}"'
                             .format(positive_end))
		if self.verbose:
			print('Identify direction using {} method...'.format(method))

		direction = None

		if method == 'single':
			if self.verbose:
				print('Positive definitional end:', definitional[0])
				print('Negative definitional end:', definitional[1])
			direction = normalize(normalize(self.model[definitional[0]])
                                  - normalize(self.model[definitional[1]]))

		elif method == 'sum':
			group1_sum_vector = np.sum([self.model[word]
                                        for word in drop_missing_keys(self.model, definitional[0])], axis=0)
			group2_sum_vector = np.sum([self.model[word]
                                        for word in drop_missing_keys(self.model, definitional[1])], axis=0)

			diff_vector = (normalize(group1_sum_vector)
                           - normalize(group2_sum_vector))

			direction = normalize(diff_vector)

		elif method == 'pca':
			pca = self._identify_subspace_by_pca(definitional, 2)
			direction = pca.components_[0]

            # if direction is opposite (e.g. we cannot control
            # what the PCA will return)
			ends_diff_projection = fast_cosine_sim((self.model[positive_end]- self.model[negative_end]),
                                                     np.array(direction, dtype=np.float32))
			if ends_diff_projection < 0:
				direction = -direction  # pylint: disable=invalid-unary-operand-type

		self.direction = direction
		self.positive_end = positive_end
		self.negative_end = negative_end




	def _identify_subspace_by_pca(self, definitional_pairs, n_components):
		matrix = []

		for word1, word2 in definitional_pairs:
			vector1 = normalize(self.model[word1])
			vector2 = normalize(self.model[word2])

			center = (vector1 + vector2) / 2

			matrix.append(vector1 - center)
			matrix.append(vector2 - center)

			pca = PCA(n_components=n_components)
			pca.fit(matrix)

		if self.verbose:
			table = enumerate(pca.explained_variance_ratio_, start=1)
			headers = ['Principal Component',
                       'Explained Variance Ratio']
			print(tabulate(table, headers=headers))

		return pca

