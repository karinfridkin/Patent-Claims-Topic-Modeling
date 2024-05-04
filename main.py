import gensim
from flask import Flask, request, jsonify
import pandas as pd
from gensim import corpora

app = Flask(__name__)

# Load the DataFrame containing the documents
numbered_corpus = pd.read_pickle('num_corpus.pkl')
id2word = pd.read_pickle('id2word.pkl')
word_corpus = pd.read_pickle('corpus.pkl')
data = pd.read_pickle('data_words.pkl')


def train_lda_model(corpus, id2word, num_topics):
    lda_model = gensim.models.LdaModel(corpus=corpus,
                                       id2word=id2word,
                                       num_topics=num_topics,
                                       random_state=100,
                                       update_every=1,
                                       chunksize=1,
                                       passes=10,
                                       alpha="auto")
    return lda_model, lda_model.print_topics()


def get_topic_counts(lda_model, word_corpus):
    topic_terms = lda_model.get_topics()

    n_top_words = 1
    top_words_per_topic = []
    for i, topic in enumerate(topic_terms):
        top_words = [id2word[id] for id in topic.argsort()[-n_top_words:][::-1]]
        top_words_per_topic.append(top_words)

    topic_labels = [' '.join(words) for words in top_words_per_topic]

    print("Topic Labels:")
    for i, label in enumerate(topic_labels):
        print("Topic {}: {}".format(i, label))

    dictionary = corpora.Dictionary(data)
    document_topics = []

    for idx, row in word_corpus.iterrows():
        text_data = row.iloc[0]
        tokens = text_data.split()

        bow = dictionary.doc2bow(tokens)
        topic_dist = lda_model.get_document_topics(bow)

        document_topics.append(topic_dist)

    def get_dominant_topic(topic_dist):
        dominant_topic = max(topic_dist, key=lambda x: x[1])
        dominant_topic_id = dominant_topic[0]
        return dominant_topic_id

    word_corpus['topic_dist'] = document_topics
    word_corpus['topics'] = word_corpus['topic_dist'].apply(get_dominant_topic).map(lambda x: topic_labels[x])

    # Group by topics and count the claims, then reindex with all possible topics
    topic_counts = word_corpus.groupby('topics').size().reindex(topic_labels, fill_value=0).reset_index(name='claims')

    print("topics count  ", topic_counts)
    return topic_labels, topic_counts



@app.route('/', methods=['POST', 'GET'])
def form_example():
    if request.method == 'POST':
        num_topics = int(request.form['num_topics'])
        lda_model, topic_labels = train_lda_model(numbered_corpus, id2word, num_topics)
        topic_labels, topic_counts = get_topic_counts(lda_model, word_corpus)

        # Convert the topic counts to a JSON response
        response = {
            'number of claims': topic_counts.to_dict(orient='records'),
            'topics': topic_labels[:num_topics]
        }
        print(response)
        return jsonify(response)

    return '''<form method="POST"> 
    num_topics <input type="number" name="num_topics">
    <input type="submit">
    </form>'''


if __name__ == '__main__':
    app.run(debug=True)
