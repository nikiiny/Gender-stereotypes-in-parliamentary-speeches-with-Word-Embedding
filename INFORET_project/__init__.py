from .analogies_functions import Analogies
from .analogies_bias import Analogies_Distance_Bias
from .weat import calc_single_weat as WEAT
from .ect import ECT
from .utils import similar_to_avg_vector, calculate_avg_vector, load_embed_model

YEARS = ['1948_1968', '1968_1985', '1985_2000', '2000_2020']
WORDS_GROUP = ['adj_appearence','family', 'career',
                'rage', 'kindness', 'intelligence', 'dumbness', 'active', 'passive',
                'gendered_words', 'female_stereotypes', 'male_stereotypes']
PAIRS_WORDS_GROUP = [['family', 'career'],
                ['rage', 'kindness'], ['intelligence', 'dumbness'], ['active', 'passive'],
                ['female_stereotypes', 'male_stereotypes']]


__all__ = ['Analogies', 'Analogies_Distance_Bias','calculate_avg_vector','similar_to_avg_vector', 'WEAT', 'ECT','load_embed_model']
