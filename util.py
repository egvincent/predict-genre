import csv

# Schema of movie metadata file:
#   adult,belongs_to_collection,budget,genres,homepage,id,imdb_id,
#   original_language,original_title,overview,popularity,poster_path,
#   production_companies,production_countries,release_date,revenue,runtime,
#   spoken_languages,status,tagline,title,video,vote_average,vote_count
# Notably:
#   3: genres
#   9: overview
MOVIES_METADATA_PATH = "movies_metadata.csv"

def read_data(csv_path):
    with open(csv_path, "rb") as csvfile:
        reader = csv.reader(csvfile)
        # each row is a list of strings
        for row in reader:
            if reader.line_num == 1:
                continue  # skip header/schema line
            
            print "overview: {}, genres: {}".format(row[9], row[3])

            if reader.line_num > 10: break  # testing

read_data(MOVIES_METADATA_PATH)