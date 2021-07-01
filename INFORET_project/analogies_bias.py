import numpy as np
import pandas as pd
import gensim
from gensim.models import KeyedVectors
from gensim.models import Word2Vec
from collections import defaultdict

from .utils import calculate_avg_vector, fast_cosine_sim
from .data import gendered_neutral_words
from .analogies_functions import Analogies


GENDER = ['male','female']
ANALOGY_TYPE = ['cosadd', 'cosmul']
WORDS_GROUP = ['adj_appearence', 'adj_positive', 'adj_negative','family', 'career']


class Analogies_Distance_Bias():
    """
    Measures how biased are words returned by analogies where a gender and a neutral
    word contributes positively and the other gender contributes negatively.
    Example: uomo:? = donna:gentile

    The algorithm retrieves a set of words by performing an analogy of the type 
    gender_1:? = gender_2:neutral_word. Then, the average cosine similarities between all the
    retrieved words and each of the gendered words is calculated. 
    If there is a difference between the two averaged similarities (one for each gender), it is 
    probable that the words returned by the analogy are more related to one gender 
    than to the other. The main idea is that if there isn't any stereotype, then the words returned
    by the analogy should ideally be equidistant from both gendered words, since they should be
    gender-neutral.
    The analogies are calculated both by using one gender as positive and the other as negative, 
    and vice versa.

    The analogies are by default unrestricted (they might return b = d). 
    Instead of using a single word for each gender, it is possible to automatically
    compute a mean vector between a set of gendered words.
    Some predefined groups of neutral words are already available within the class.


    Parameters
    ------------------
    model: Word embedding model of 'gensim.model.KeyedVectors'.
    gender_female (str/list): word or words related to female gender.
    gender_male (str/list): word or words related to male gender.
    use_avg_gender (bool): whether to use a mean vector of gendered words.
        Default: False
    type_most_similar (str): type of analogy to use. Possible choices are
        'cosmul' and 'cosadd'.
        Default: 'cosmul'

    Attributes
    --------------------
    dict_analogies: stores the result of analogies, average cosine similarity and 
        difference between average similarities for both genders. 
    """
    
    def __init__(self, 
                 model, #model.wv
                 gender_female=None, 
                 gender_male=None, 
                 use_avg_gender=False, 
                 type_most_similar = 'cosmul'
                ):
        
        self.model = model
        self.gender_female = gender_female
        self.gender_male = gender_male
        self.use_avg_gender = use_avg_gender
        self.type_most_similar = type_most_similar
        
        self.dict_analogies = defaultdict(lambda: defaultdict(dict))
        self.center_vectors = defaultdict(dict)
        
        self.tot_bias = defaultdict(dict)
        self.tot_bias['female'] = []
        self.tot_bias['male'] = []
        
        analogies = Analogies(model)
        
        if type_most_similar == 'cosmul':
            self.most_similar = analogies.most_similar_cosmul
        elif type_most_similar == 'cosadd':
            self.most_similar = analogies.most_similar
        
        
        if self.type_most_similar not in ANALOGY_TYPE:
            raise ValueError( "Argument 'type_most_similar' has an incorrect value: use one among {}".format(ANALOGY_TYPE))
            
        
        if self.use_avg_gender:
            # calculates mean vector for each gender
            self.center_vectors['female'] = calculate_avg_vector(self.model, gendered_neutral_words['female'])
            self.center_vectors['male'] = calculate_avg_vector(self.model, gendered_neutral_words['male'])
        else:
            self.center_vectors['female'] = model.get_vector(self.gender_female, norm=True)
            self.center_vectors['male'] = model.get_vector(self.gender_male, norm=True)
            
    
    
    def get_analogies(self, pos_word, topn=10):
        """Performs analogies. First one gender contributes positively and the other
        negatively, then vice versa. The neutral word always contributes positively.
        The analogies are stored in the attribute dict_analogies.
        
        Instead of using a single word for each gender, it is possible to automatically
        compute a mean vector between a set of gendered words.

        Parameters
        ------------------
        pos_word (str): neutral word which contributes positively.
        topn (int): number of words to be returned from analogy.
            Default: 10
        """
    
        if self.use_avg_gender:
            
            analogies_donna = self.most_similar(positive=pos_word, negative=None,
                                                use_avg_gender=self.use_avg_gender,
                                                positive_gender='female')
                
            analogies_uomo = self.most_similar(positive=pos_word, negative=None,
                                                use_avg_gender=self.use_avg_gender,
                                               positive_gender='male')
            
            
        else:
            
            analogies_donna = self.most_similar(positive= [self.gender_female] + [pos_word],
                                                use_avg_gender=self.use_avg_gender, 
                                                negative=[self.gender_male], topn=topn)
                
            analogies_uomo = self.most_similar(positive= [self.gender_male] + [pos_word],
                                                use_avg_gender=self.use_avg_gender, 
                                                negative=[self.gender_female], topn=topn)    
            
        self.dict_analogies[pos_word]['female']['analogies'] = analogies_donna
        self.dict_analogies[pos_word]['male']['analogies'] = analogies_uomo
    
    
    
    def get_top_bias(self, positive_word=None, pred_positive_word=None, topn=10, verbose=True):
        """
        Perform analogies for the neutral word or words for each gender. Then it 
        calculates the average cosine similarity between all the words returned by the analogy
        and each gendered word, computes the difference between the average cosine similarities of
        the two genders and stores them. 

        Parameters
        ------------
        positive_word (str, list): neutral words that contributes positively to the analogy.
        pred_positive_word (str): set of prefedined neutral words. Possible choices are
            'adj_appearence', 'adj_positive', 'adj_negative','family', 'career'.
            TO BE DEFINED
        topn (int): number of words to be returned from analogy.
            Default: 10
        verbose: if True, prints for each word the average similarity for analogies
            of both genders and the bias, ordered by the value of the bias. 
            Default: True


        Returns
        ------------
        list of (word, differences in average distance) ordered by difference.
        """
        
        if pred_positive_word:
            if pred_positive_word not in WORDS_GROUP:
                raise ValueError( "Argument 'predefined_positive_words' has an incorrect value: use one among {}".format(WORDS_GROUP))
            positive_word = gendered_neutral_words[pred_positive_word]
            
            
        else:
            if isinstance(positive_word, str):
                positive_word = [positive_word]
    
            
        for word in positive_word:
            try:
                # perform analogies
                self.get_analogies(word, topn=topn)
                
                for gender in GENDER:
                    # retrieve the words returned by the analogies of each gender
                    analogies = self.dict_analogies[word][gender]['analogies']
                    analogies = [w[0] for w in analogies]
                    # calculates average similarity between all the words returned by the analogy
                    #and the gendered vector
                    cos_sim_male = np.array( [fast_cosine_sim(self.model[w], self.center_vectors['male'])  for w in analogies] ).mean()
                    cos_sim_female =  np.array( [fast_cosine_sim(self.model[w], self.center_vectors['female'])  for w in analogies] ).mean()
                    # store the average similarity
                    self.dict_analogies[word][gender]['cos_sim_male'] = cos_sim_male
                    self.dict_analogies[word][gender]['cos_sim_female'] = cos_sim_female 

                    # calculates the bias as the difference between the average similarities of both genders
                    difference = abs(cos_sim_male - cos_sim_female)
                    # store the bias
                    self.dict_analogies[word][gender]['difference'] = difference
            
            except:
                pass
            
        # sort positive words by the bias of their analogy
        self.sorted_keys = sorted(self.dict_analogies.keys(), 
                                 key=lambda x: (self.dict_analogies[x]['female']['difference'] + self.dict_analogies[x]['male']['difference']),
                                 reverse=True)  
        # sort the bias according to the sorted_keys
        sorted_difference = [ (self.dict_analogies[w]['male']['difference'], 
                                self.dict_analogies[w]['female']['difference']) for w in self.sorted_keys]      
            
        # print the results sorted by the words with the highest bias
        if verbose:
            for word in self.sorted_keys:
                print(f"\nWord: {word}")
                for gender in GENDER:
                    print(f"Similarity of '{gender}' analogies to 'male': {self.dict_analogies[word][gender]['cos_sim_male']}, to 'female': {self.dict_analogies[word][gender]['cos_sim_female']}")
                    print(f"Bias for '{gender}' analogies: {self.dict_analogies[word][gender]['difference']}")
                
        return [*zip(self.sorted_keys,sorted_difference)]
                
    def print_top_analogies(self, topn=5):
        """
        Prints the words retrieved by the analogies for the most biased positive words.
        """
        if not [self.dict_analogies[word][gender]['difference'] for word in self.dict_analogies.keys() for gender in GENDER]:
            raise Exception("Empty dictionary, call method 'get_bias' before")
            
        for i,word in enumerate(self.sorted_keys):
            if i<topn:
                print(f"\nWord: {word}\n")
                for gender in GENDER:
                    print(f"Positive gender: {gender}")
                    display(self.dict_analogies[word][gender]['analogies'])        
                
