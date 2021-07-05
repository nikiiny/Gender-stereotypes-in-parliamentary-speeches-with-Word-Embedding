import numpy as np
import gensim
from gensim.models import KeyedVectors
from gensim.models import Word2Vec
from scipy.stats import spearmanr

from .utils import fast_cosine_sim, calculate_avg_vector


class ECT():

    def __init__(self, model, 
        female_gender_words, 
        male_gender_words):

        self.model = model
        self.female_gender_words = female_gender_words
        self.male_gender_words = male_gender_words

        """Performs Embedding Coherence Test.
        References 'Attenuating Bias in Word Vectors' by S.dev and J.M.Phillips 
        https://arxiv.org/abs/1901.07656
        
        The algorithm takes as input two groups of gendered pairs of words and
        a group of neutral words. A representative vector for each gender is created
        as a mean vector of all the words in each group, then the cosine similarity between each word
        in the neutral group and each of the mean gendered vector is computed. This results in
        two vectors of similarities for each gender. 
        Finally, the Spearman rank-order correlation coefficient is calculated between the 2 similarities
        vectors. The main idea is that the higher is the correlation, the lower is the bias. That's because
        the distance between the gendered words and the neutral words would be similar.


        Parameters
        --------------
        model: Word embedding model of 'gensim.model.KeyedVectors'.
        female_gender_words (list): list of words related to female gender.
        male_gender_words (list): list of words related to male gender.
        """

    def get_bias(self,
        neutral_words, 
        verbose=True):

        """
        Parameters
        --------------
        neutral_words (list): list of neutral words not specifically related to any gender. 
        verbose (bool): if True, prints correlation and p-value.
            Default: True

        Returns
        --------------
        Spearman rank-order correlation.
        """

        # create mean vector for each gender
        female_avg = calculate_avg_vector(self.model, self.female_gender_words)
        male_avg = calculate_avg_vector(self.model, self.male_gender_words)
        
        self.cos_sim_female = []
        self.cos_sim_male = []
        self.keys_found = []

        if isinstance(neutral_words, str):
            neutral_words = [neutral_words]
        
        for word in neutral_words:
            try:
                # calculate cosine similarity between a word and each gendered vector
                #and append to a list
                self.cos_sim_female.append(fast_cosine_sim(female_avg, self.model[word]))
                self.cos_sim_male.append(fast_cosine_sim(male_avg, self.model[word]))
                # store words found in the model
                self.keys_found.append(word)
            except:
                pass

        # calculate spearman rank-order correlation coefficient
        spearman_corr, pval = spearmanr(np.array(self.cos_sim_female), 
                                        np.array(self.cos_sim_male))
        
        if verbose:
            print(f"ECT for words: {self.keys_found}\n")
            print(f"Spearman correlation has value {spearman_corr:.4f} with p-value {pval:.4f}")
            
            if spearman_corr>0.75:
                print("High correlation --> Low bias\n")
            elif spearman_corr<0.3:
                print("Low correlation --> High bias\n")
            else:
                print("Moderate correlation --> Moderate bias\n")
            
        return spearman_corr


    def get_cosine_sim_words(self, verbose=True):
        """
        Orders the words in the neutral group by the difference in the cosine similarities
            between the word and the female group and the word and the male group.

        Parameters
        --------------
        verbose (bool): if True, prints neutral words and their cosine similarity to female 
            and male groups.
            Default: True

        Returns
        --------------
        Sorted list of tuples (neutral word, cosine similarity to female group, cosine similarity to male group)
        """

        # order the words by the distance of the cosine similarities between female and the word and male and
        #the word 
        cos_sim = [ (word, round(self.cos_sim_female[i],4), round(self.cos_sim_male[i],4)) for i,word in enumerate(self.keys_found)]
        cos_sim = sorted(cos_sim, key= lambda x: abs(x[1]-x[2]), reverse=True)

        if verbose:
            for i,word in enumerate([j[0] for j in cos_sim]):
                print(f"Cosine similarity of '{word}' to 'female' is {cos_sim[i][1]:.4f}, to 'male' is {cos_sim[i][2]:.4f}")

        
        return cos_sim
                
                
        
