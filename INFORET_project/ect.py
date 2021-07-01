import numpy as np
import gensim
from gensim.models import KeyedVectors
from gensim.models import Word2Vec
from scipy.stats import spearmanr
from .utils import fast_cosine_sim, calculate_avg_vector



from scipy.stats import spearmanr

def ECT(model, 
    female_gender_words, 
    male_gender_words, 
    neutral_words, 
    verbose=True, 
    return_words_sim=True):

    """Performs Embedding Coherence Test.
    References 'Attenuating Bias in Word Vectors' by S.dev and J.M.Phillips 
    https://arxiv.org/abs/1901.07656.
    
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
    neutral_words (list): list of neutral words not specifically related to any gender. 
    verbose (bool): if True, prints correlation and p-value. If returns_words_sim
        is True prints cosine similarity between each word and each gender.
        Default: True
    return_words_sim (bool): if True and verbose True, prints cosine similarity between 
        each word and each gender.
        Default: True

    Returns
    --------------
    Spearman rank-order correlation and p-value.
    """

    # create mean vector for each gender
    female_avg = calculate_avg_vector(model, female_gender_words)
    male_avg = calculate_avg_vector(model, male_gender_words)
    
    cos_sim_female = []
    cos_sim_male = []
    keys_found = []

    if isinstance(neutral_words, str):
        neutral_words = [neutral_words]
    
    for word in neutral_words:
        try:
            # calculate cosine similarity between a word and each gendered vector
            #and append to a list
            cos_sim_female.append(fast_cosine_sim(female_avg, model[word]))
            cos_sim_male.append(fast_cosine_sim(male_avg, model[word]))
            # store words found in the model
            keys_found.append(word)
        except:
            pass

    # calculate spearman rank-order correlation coefficient
    spearman_corr, pval = spearmanr(np.array(cos_sim_female), 
                                    np.array(cos_sim_male))
    
    if verbose:
        print(f"ECT for words: {keys_found}\n")
        print(f"Spearman correlation has value {spearman_corr:.4f} with p-value {pval:.4f}")
        
        if spearman_corr>0.75:
            print("High correlation --> Low bias\n")
        elif spearman_corr<0.3:
            print("Low correlation --> High bias\n")
        else:
            print("Moderate correlation --> Moderate bias\n")
        
        if return_words_sim:
            for i,word in enumerate(keys_found):
                print(f"Cosine similarity of '{word}' to 'female' is {cos_sim_female[i]:.4f}, to 'male' is {cos_sim_male[i]:.4f}")
            
            
    return spearman_corr, pval
