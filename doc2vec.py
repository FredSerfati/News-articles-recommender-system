import sys
import re
import string
import os
import numpy as np
import codecs
import pickle

# From scikit learn that got words from:
# http://ir.dcs.gla.ac.uk/resources/linguistic_utils/stop_words
ENGLISH_STOP_WORDS = frozenset([
    "a", "about", "above", "across", "after", "afterwards", "again", "against",
    "all", "almost", "alone", "along", "already", "also", "although", "always",
    "am", "among", "amongst", "amoungst", "amount", "an", "and", "another",
    "any", "anyhow", "anyone", "anything", "anyway", "anywhere", "are",
    "around", "as", "at", "back", "be", "became", "because", "become",
    "becomes", "becoming", "been", "before", "beforehand", "behind", "being",
    "below", "beside", "besides", "between", "beyond", "bill", "both",
    "bottom", "but", "by", "call", "can", "cannot", "cant", "co", "con",
    "could", "couldnt", "cry", "de", "describe", "detail", "do", "done",
    "down", "due", "during", "each", "eg", "eight", "either", "eleven", "else",
    "elsewhere", "empty", "enough", "etc", "even", "ever", "every", "everyone",
    "everything", "everywhere", "except", "few", "fifteen", "fifty", "fill",
    "find", "fire", "first", "five", "for", "former", "formerly", "forty",
    "found", "four", "from", "front", "full", "further", "get", "give", "go",
    "had", "has", "hasnt", "have", "he", "hence", "her", "here", "hereafter",
    "hereby", "herein", "hereupon", "hers", "herself", "him", "himself", "his",
    "how", "however", "hundred", "i", "ie", "if", "in", "inc", "indeed",
    "interest", "into", "is", "it", "its", "itself", "keep", "last", "latter",
    "latterly", "least", "less", "ltd", "made", "many", "may", "me",
    "meanwhile", "might", "mill", "mine", "more", "moreover", "most", "mostly",
    "move", "much", "must", "my", "myself", "name", "namely", "neither",
    "never", "nevertheless", "next", "nine", "no", "nobody", "none", "noone",
    "nor", "not", "nothing", "now", "nowhere", "of", "off", "often", "on",
    "once", "one", "only", "onto", "or", "other", "others", "otherwise", "our",
    "ours", "ourselves", "out", "over", "own", "part", "per", "perhaps",
    "please", "put", "rather", "re", "same", "see", "seem", "seemed",
    "seeming", "seems", "serious", "several", "she", "should", "show", "side",
    "since", "sincere", "six", "sixty", "so", "some", "somehow", "someone",
    "something", "sometime", "sometimes", "somewhere", "still", "such",
    "system", "take", "ten", "than", "that", "the", "their", "them",
    "themselves", "then", "thence", "there", "thereafter", "thereby",
    "therefore", "therein", "thereupon", "these", "they", "thick", "thin",
    "third", "this", "those", "though", "three", "through", "throughout",
    "thru", "thus", "to", "together", "too", "top", "toward", "towards",
    "twelve", "twenty", "two", "un", "under", "until", "up", "upon", "us",
    "very", "via", "was", "we", "well", "were", "what", "whatever", "when",
    "whence", "whenever", "where", "whereafter", "whereas", "whereby",
    "wherein", "whereupon", "wherever", "whether", "which", "while", "whither",
    "who", "whoever", "whole", "whom", "whose", "why", "will", "with",
    "within", "without", "would", "yet", "you", "your", "yours", "yourself",
    "yourselves"])


def load_glove(filename):
    """
    Read all lines from the indicated file and return a dictionary
    mapping word:vector where vectors are of numpy `array` type.
    GloVe file lines are of the form:

    the 0.418 0.24968 -0.41242 0.1217 ...

    So split each line on spaces into a list; the first element is the word
    and the remaining elements represent factor components.
    The length of the vector does not matter; read vectors of any length.

    ignore stopwords
    """
    with open(filename, 'r') as file:
        word_to_vec_dict = {
            line.split(' ')[0]: np.array(line.split(' ')[1:]).astype(float)
            for line in file.readlines()
            if line.split(' ')[0] not in ENGLISH_STOP_WORDS}
    return word_to_vec_dict


def filelist(root):
    """Return a fully-qualified list of filenames
    for text files under root directory"""
    full_filepaths = []
    for path, _, files in os.walk(root):
        for name in files:
            if name.endswith('.txt'):
                full_filepath = os.path.join(path, name)
                full_filepaths.append(full_filepath)
    return full_filepaths


def full_to_short_filepath(full_filepath):
    """Return the short filepath from the full filepath.

    Input: full_filepath, which looks like
    "/Users/user_name/data/bbc/tech/text1.txt",
    assuming the data has been stored in ~/data.

    Output: short_filepath, containing only the topic
    and the filename, which looks like "tech/text1.txt
    for the previous example.
    """

    idx = full_filepath.find('bbc/')
    short_filepath = full_filepath[idx+4:]
    return short_filepath


def get_text(full_filepath):
    """
    Load and return the text of a text file, assuming latin-1 encoding as that
    is what the BBC corpus uses.  Use codecs.open() function not open().
    """
    with codecs.open(full_filepath, encoding='latin-1', mode='r') as file:
        text = file.read()
    return text


def words(text):
    """
    Given a string, return a list of words normalized as follows.

        1. Lowercase all words
        2. Use re.sub function and string.punctuation + '0-9\\r\\t\\n]'
            to replace all those char with a space character.
        3. Split on space to get word list.
        4. Ignore words < 3 char long.
        5. Remove English stop words
    """
    text = text.lower()
    text = re.sub("[" + string.punctuation + '0-9\\r\\t\\n]', ' ', text)
    words_list = text.split(' ')
    words_list = [word for word in words_list if len(
        word) >= 3 and word not in ENGLISH_STOP_WORDS]
    return words_list


def split_title(text):
    """
    Given text returns title and the rest of the article.

    Split the text by "\n" and assume that the first element is the title
    """
    title = text.split('\n')[0]
    rest_of_article = ','.join(text.split('\n')[1:]).strip(',')
    return title, rest_of_article


def compute_centroid(words_list, gloves):
    """ Given a list of words and the gloves file,
    compute the centroid of the embeddings of the words."""

    word_embeddings_list = [gloves[word]
                            for word in words_list if word in gloves]
    wordvec_centroid_for_article_text = np.mean(word_embeddings_list, axis=0)
    return wordvec_centroid_for_article_text


def load_articles(articles_dirname, gloves):
    """
    Load all .txt files under articles_dirname
    and return a table (list of lists/tuples)
    where each record is a list of:

      [filename, title, article-text-minus-title,
        wordvec-centroid-for-article-text]
      -> replace filename with short_filepath

    We use gloves parameter to compute the word vectors and centroid.

    The filename is fully-qualified name of the text file including
    the path to the root of the corpus passed in on the command line.

    When computing the vector for each document, use just the text,
    not the text and title.
    """

    table = []
    full_filepaths = filelist(articles_dirname)

    for full_filepath in full_filepaths:
        text = get_text(full_filepath)
        title, article_text_minus_title = split_title(text)
        words_list = words(article_text_minus_title)
        wordvec_centroid_for_article_text = compute_centroid(
            words_list, gloves)
        short_filepath = full_to_short_filepath(full_filepath)
        table.append([short_filepath, title, article_text_minus_title,
                     wordvec_centroid_for_article_text])
    return table


def doc2vec(text, gloves):
    """
    Return the word vector centroid for the text. Sum the word vectors
    for each word and then divide by the number of words. Ignore words
    not in gloves.
    """
    words_list = words(text)
    wordvec_centroid_for_text = compute_centroid(words_list, gloves)
    return wordvec_centroid_for_text


def fields(article):
    """ Return a dictionary containing the different fields of a given article.
        This function is not necessary but will improve code's readability.
    Input:
        article: list [short_filepath, title,
                        text-minus-title, wordvec-centroid]
    Output:
        dictionary which keys are short_filepath, topic, title, text
        and wordvec-centroid and values are corresponding fields.
    """
    short_filepath = article[0]
    idx = short_filepath.index('/')
    name = short_filepath[idx+1:]
    topic = short_filepath[:idx]
    title = article[1]
    text = article[2]
    wordvec_centroid = article[3]

    fields_dict = {'name': name, 'topic': topic, 'title': title,
                   'text': text, "wordvec_centroid": wordvec_centroid}
    return fields_dict


def euclidean_distance(vec_1, vec_2):
    """ Return the euclidean distance between two vectors vec_1 and vec_2.
    """
    return np.linalg.norm(vec_1-vec_2)


def distances(article, articles):
    """
    Compute the euclidean distance from article to every other article.

    Inputs:
        article = [filename, title, text-minus-title, wordvec-centroid]
        articles is a list of [filename, title,
        text-minus-title, wordvec-centroid]

    Output:
        list of (distance, a) for a in articles
        where a is a list [filename, title, text-minus-title, wordvec-centroid]
    """
    wordvec_centroid_for_text = fields(article)['wordvec_centroid']
    distances_list = [
        (euclidean_distance(wordvec_centroid_for_text, a[3]), a)
        for a in articles]
    return distances_list


def recommended(article, articles, n_recommendations):
    """ Return top n articles closest to article.

    Inputs:
        article: list [filename, title, text-minus-title, wordvec-centroid]
        articles: list of list [filename, title,
                                text-minus-title, wordvec-centroid]

    Output:
         list of [topic, filename, title]
    """
    # Careful: don't include the article itself from the recommendation
    # (corresponds the the smallest distance: 0)

    distances_list = distances(article, articles)
    distances_list_sorted = sorted(distances_list, key=lambda x: x[0])
    articles_list_sorted = [item[1]
                            for item in distances_list_sorted]  # list of lists

    article_title = fields(article)['title']
    article_text = fields(article)['text']

    recommended_articles = [
        [fields(a)['topic'], fields(a)['name'], fields(a)['title'],
            article_title, article_text]
        for a in articles_list_sorted[1: n_recommendations+1]
    ]
    return recommended_articles


def main():

    # file giving the word embeddings for lots of common words
    glove_filename = os.path.expanduser(sys.argv[1])

    # directory where the articles are
    articles_dirname = os.path.expanduser(sys.argv[2])

    # dictionary giving the word embedding for all words in glove_filename
    gloves = load_glove(glove_filename)

    # list of articles, each article being represented
    # by a list [filename, title, article_text_minus_title, wordvec_centroid]
    articles_for_recommendations = load_articles(articles_dirname, gloves)

    # save the previous list as articles.pkl with
    # the following information about articles
    # [[topic, filename, title, text], ...]

    articles_for_pickle = [
        [fields(a)['topic'], fields(a)['name'],
         fields(a)['title'], fields(a)['text']]
        for a in articles_for_recommendations
    ]

    pickle_articles_path = os.path.expanduser("~/data/articles.pickle")
    with open(pickle_articles_path, 'wb') as file:
        pickle.dump(articles_for_pickle, file)

    # save as recommended.pkl a dictionary with top 5
    # recommendations for each article.
    # given an article, use (topic, filename) as the key
    # the recommendations are a list of [topic, filename, title]
    # for the top 5 closest articles
    # you may want to also add the title and text of the current article

    n_recommendations = 5
    article_to_recommendations = {
        (fields(a)['topic'], fields(a)['name']):
        recommended(a, articles_for_recommendations, n_recommendations)
        for a in articles_for_recommendations
    }

    pickle_recommended_path = os.path.expanduser("~/data/recommended.pickle")
    with open(pickle_recommended_path, 'wb') as file:
        pickle.dump(article_to_recommendations, file)


if __name__ == '__main__':
    main()
