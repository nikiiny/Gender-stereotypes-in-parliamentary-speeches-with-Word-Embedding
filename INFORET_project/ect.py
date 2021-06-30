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
    """Performs Embedding Coherence Test"""

    female_avg = calculate_avg_vector(model, female_gender_words)
    male_avg = calculate_avg_vector(model, male_gender_words)
    
    cos_sim_female = []
    cos_sim_male = []
    keys_found = []
    
    for word in neutral_words:
        try:
            cos_sim_female.append(fast_cosine_sim(female_avg, model[word]))
            cos_sim_male.append(fast_cosine_sim(male_avg, model[word]))
            keys_found.append(word)
        except:
            pass
    
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