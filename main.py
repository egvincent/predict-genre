import csv
import json
import collections
import util
import math
import random
# from gensim.models import Word2Vec    # deprecated apparently
from gensim.models import KeyedVectors

# Schema of movie metadata file:
#   adult,belongs_to_collection,budget,genres,homepage,id,imdb_id,
#   original_language,original_title,overview,popularity,poster_path,
#   production_companies,production_countries,release_date,revenue,runtime,
#   spoken_languages,status,tagline,title,video,vote_average,vote_count
# Notably:
#   3: genres
#   9: overview
MOVIES_METADATA_PATH = "movies_metadata.csv"

# Dict ID -> genre name for all genres we care about.
# Full list of genres include production companies and medium "TV Movie"
# which we exclude:
#   [(12, u'Adventure'), (14, u'Fantasy'), (16, u'Animation'), (18, u'Drama'),
#   (27, u'Horror'), (28, u'Action'), (35, u'Comedy'), (36, u'History'),
#   (37, u'Western'), (53, u'Thriller'), (80, u'Crime'), (99, u'Documentary'),
#   (878, u'Science Fiction'), (2883, u'Aniplex'), (7759, u'GoHands'),
#   (7760, u'BROSTA TV'), (7761, u'Mardock Scramble Production Committee'),
#   (9648, u'Mystery'), (10402, u'Music'), (10749, u'Romance'),
#   (10751, u'Family'), (10752, u'War'), (10769, u'Foreign'),
#   (10770, u'TV Movie'), (11176, u'Carousel Productions'),
#   (11602, u'Vision View Entertainment'), (17161, u'Odyssey Media'),
#   (18012, u'Pulser Productions'), (18013, u'Rogue State'),
#   (23822, u'The Cartel'), (29812, u'Telescene Film Group Productions'),
#   (33751, u'Sentai Filmworks')]
GENRES_BY_ID = {
    12: "Adventure",
    14: "Fantasy",
    16: "Animation",
    18: "Drama",
    27: "Horror",
    28: "Action",
    35: "Comedy",
    36: "History",
    37: "Western",
    53: "Thriller",
    80: "Crime",
    99: "Documentary",
    878: "Science Fiction",
    9648: "Mystery",
    10402: "Music",
    10749: "Romance",
    10751: "Family",
    10752: "War",
    10769: "Foreign"
}


def learnPredictor(trainExamples, testExamples, genreID, numIters, eta):
    '''
    Given |trainExamples| and |testExamples| (each one is a list of (x,y)
    pairs), a |featureExtractor| to apply to x, and the number of iterations to
    train |numIters|, the step size |eta|, return the weight vector (sparse
    feature vector) learned.

    You should implement stochastic gradient descent.

    Note: only use the trainExamples for training!
    You should call evaluatePredictor() on both trainExamples and testExamples
    to see how you're doing as you learn after each iteration.
    '''

    # trainExamples and testExamples are lists of (x, y) pairs in the format:
    #   x: a feature vector in the form of a defaultdict
    #   y: a tuple of all the genreIDs that this movie is tagged with

    weights = {}  # feature => weight
    # BEGIN_YOUR_CODE (our solution is 12 lines of code, but don't worry if you deviate from this)
    for i in range(numIters):
        for idx, j in enumerate(trainExamples):
            f = j[0]
            grad = 0
            for k in f:
                if k not in weights:
                    weights[k] = 0
                grad += weights[k] * f[k]
            temp = 0  # hacky
            if genreID in j[1]:
                temp = 1
            else:
                temp = -1
            if 1 - grad * temp > 0: # formula used to be 1 - grad * j[1] and j[1] used to be 0 or 1: the tru label.
                for k in f:
                    weights[k] -= eta * f[k] * temp*-1
    print util.evaluatePredictor(
        examples = trainExamples, 
        predictor = lambda x: 1 if util.dotProduct(weights, x) > 0 else -1, 
        genreID = genreID), "train error,", "genre=", genreID
    print util.evaluatePredictor(
        examples = testExamples, 
        predictor = lambda x: 1 if util.dotProduct(weights, x) > 0 else -1, 
        genreID = genreID), "test error,", "genre=", genreID
    # END_YOUR_CODE
    return weights



# returns a list of pairs (x, y) such that:
#   - x is the overview string
#   - y is the corresponding genre ID tuple
def read_data(csv_path):
    examples = []
    with open(csv_path, "rb") as csvfile:
        reader = csv.reader(csvfile)
        # each row is a list of strings
        for row in reader:
            if reader.line_num == 1:
                continue  # skip header/schema line

            x = row[9]

            genres = row[3].replace("'", '"')  # ' -> " for json package
            genre_dict_list = json.loads(genres)
            genre_id_list = []
            for genre in genre_dict_list:
                # genre in format {"id": <int>, "name": <string>}
                genre_id_list.append(genre["id"])
            y = tuple(genre_id_list)

            examples.append( (x, y) )

    return examples

# returns a sparse feature vector (default dict) in the format
#   (n adjacent words) -> (1 if present else 0)
# @param use_counts: 
#   - if true, vector contains # occurences of the n-gram
#   - if false, vector contains 1 if the n-gram is present, 0 otherwise
# @param use_characters:
#   - if true: use character grams instead of word grams (i.e. tuples of n adjacent characters)
def extract_n_gram_features(x, n, use_counts=False, use_characters=False):
    features = collections.defaultdict(float)

    words_or_chars = x
    if not use_characters:
        words_or_chars = words_or_chars.split()

    for i in range(len(words_or_chars) - n + 1):
        n_gram = tuple(words_or_chars[i : i+n])

        if use_counts:
            features[n_gram] += 1
        else: 
            features[n_gram] = 1

    return features


def cosine_similarity(d1, d2):
    vec1 = list(d1.values())
    vec2 = list(d2.values())
    sum11, sum12, sum22 = 0, 0, 0
    for i in range(len(vec1)):
        x = vec1[i];
        y = vec2[i]
        sum11 += x * x
        sum22 += y * y
        sum12 += x * y
    return sum12 / math.sqrt(sum11 * sum22)


#pass in two full example
def jaccard_similarity(a, b):
    d1 = set()
    d2 = set()
    d3 = set()  # copy of d1
    if len(a[0]) == 0 and len(b[0]) == 0:
        return 1
    for i in a[0]:
        d1.add(i)
        d3.add(i)

    for i in b[0]:
        d2.add(i)

    intersection = len(d1.intersection(d2))
    union = (len(d3) + len(d2)) - intersection
    return float(intersection) / union







# example:
raw_examples = read_data(MOVIES_METADATA_PATH)
print "generating features ..."
examples = [(extract_n_gram_features(x, n=1), y) for x,y in raw_examples]
#examples_counts = [(extract_n_gram_features(x, n=1, use_counts=True), y) for x,y in raw_examples]
#examples_counts_5_gram = [(extract_n_gram_features(x, n=5, use_counts=True), y) for x,y in raw_examples]
#examples_counts_5_gram_chars = [(extract_n_gram_features(x, n=1, use_counts=True, use_characters=True), y) for x,y in raw_examples]

training_data = examples[0:int(.8 * len(examples))]
test_data = examples[int(.8 * len(examples)):]

# for genreID in GENRES_BY_ID:
#     # print GENRES_BY_ID[genreID]
#     # learnPredictor(training_data, test_data, genreID, 20, .01)
#     print GENRES_BY_ID[genreID]
#
#     print "n = 1"
#     learnPredictor(examples[0:int(.8 * len(examples))], examples[int(.8 * len(examples)):], genreID, 20, .01)
#
#     print "n = 1, use_counts"
#     learnPredictor(examples_counts[0:int(.8 * len(examples))], examples[int(.8 * len(examples)):], genreID, 20, .01)
#
#     print "n = 5, use_counts"
#     learnPredictor(examples_counts_5_gram[0:int(.8 * len(examples))], examples[int(.8 * len(examples)):], genreID, 20, .01)
#
#     print "n = 5, use_counts, use_characters"
#     learnPredictor(examples_counts_5_gram_chars[0:int(.8 * len(examples))], examples[int(.8 * len(examples)):], genreID, 20,
#                    .01)



# load a ~3.5gb pretrained model file, which takes up ~4.5gb memory to store all the vectors
word2vec_model = KeyedVectors.load_word2vec_format(
    './GoogleNews-vectors-negative300.bin', binary=True)

print "vector for 'test': ", word2vec_model.wv["test"]



def n_top_values(my_list, n):
    return sorted(range(len(my_list)), key=lambda i: my_list[i])[-n:]


def predict_single_example(training_data, i):
    k_nn = []
    for j in training_data:
        k_nn.append(jaccard_similarity(j, i))
    k = 10
    top_k = n_top_values(k_nn, k)
    closest = []
    for j in top_k:
        closest.append(training_data[j])
    genres = set()
    for j in closest:
        for g in j[1]:
            genres.add(g)
    final_pred = []
    threshold = .25
    for g in genres:
        num_containing = 0
        for x in closest:
            if g in x[1]:
                num_containing += 1
        if float(num_containing) / k >= threshold:
            final_pred.append(g)
    return final_pred

def classify_jaccard(training_data, test_data):
    predictions = []
    for i in test_data:
        final_pred = predict_single_example(training_data, i)
        predictions.append(final_pred)
    return predictions


knn_predictions_for_test_data = classify_jaccard(training_data, test_data)

def evaluateJaccard(examples, predictions, genreID):
    '''
    predictor: a function that takes an x and returns a predicted y.
    Given a list of examples (x, y), makes predictions based on |predict| and returns the fraction
    of misclassiied examples.
    '''
    error = 0
    for i in range(examples):
        if genreID not in predictions[i] and genreID in examples[i][1]:
            error += 1
        elif genreID not in predictions[i] and genreID not in examples[i][1]:
            error += 1
    return 1.0 * error / len(examples)

for genreID in GENRES_BY_ID:
    print evaluateJaccard(test_data, knn_predictions_for_test_data, genreID)