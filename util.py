import csv
import json
import collections

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

    # 1-grams only:
    n = 1
    return collections.defaultdict(float, [(word, 1) for word in x.split()])


# example:
train_examples = read_data(MOVIES_METADATA_PATH)
phi = [extract_n_gram_features(x, 1) for x,y in train_examples]

print train_examples[0]
print phi[0]
