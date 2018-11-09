import csv
import json
import collections
import util
import math
import random

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
    print util.evaluatePredictor(trainExamples, lambda x: math.copysign(1, util.dotProduct(weights, x)), genreID), "train:", "genre=", genreID
    print util.evaluatePredictor(testExamples, lambda x: math.copysign(1, util.dotProduct(weights, x)), genreID), "test"
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
def extract_n_gram_features(x, n):
    # TODO
    # currently only doing 1-grams:
    return collections.defaultdict(float, [(word, 1) for word in x.split()])


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



def jaccard_similarity(d1, d2):
    d1 = set(d1.values)
    d2 = set(d2.values)
    intersection = len(d1.intersection(d2))
    union = (len(d1) + len(d2)) - intersection
    return float(intersection) / union







# example:
train_examples = read_data(MOVIES_METADATA_PATH)
n_grams = 1
train = [(extract_n_gram_features(x, n_grams), y) for x,y in train_examples]

training_data = train[0:int(.8 * len(train))]
test_data = train[int(.8 * len(train)):]

for genreID in GENRES_BY_ID:
    # print GENRES_BY_ID[genreID]
    # learnPredictor(training_data, test_data, genreID, 20, .01)
    pass




def n_top_values(my_list, n):
    return sorted(range(len(my_list)), key=lambda i: my_list[i])[-n:]


def classify_jaccard(training_data, test_data):
    predictions = []
    for i in test_data:
        k_nn = []
        for j in training_data:
            k_nn.append(jaccard_similarity(j[0], i[0]))
        k = 10
        top_k = n_top_values(k_nn, k)
        closest = []
        for j in top_k:
            closest.append(training_data[j])
        genres = set()
        for j in closest:
            for g in j[1]:
                genres.append(g)
        final_pred = []
        threshold = .5
        for g in genres:
            num_containing = 0
            for x in closest:
                if g in x[1]:
                    num_containing += 1
            if float(num_containing) / k >= threshold:
                final_pred.append(g)
        predictions.append(final_pred)







classify_jaccard(training_data, test_data)


