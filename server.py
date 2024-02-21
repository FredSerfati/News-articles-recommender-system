# Launch with python server.py

import os

from flask import Flask, render_template
import sys
import pickle


app = Flask(__name__)


@app.route("/")
def articles():
    """Show a list of article titles"""
    return render_template("articles.html", articles=articles)


@app.route("/article/<topic>/<filename>")
def article(topic, filename):
    """
    Show an article with relative path filename. Assumes the BBC structure of
    topic/filename.txt so our URLs follow that.
    """
    return render_template("article.html",
                           article=recommended[(topic, filename)])
    # recommended[(topic, filename) is a dictionary
    # mapping (topic, filename) to a list of recommendations of the form
    # [topic, filename, title, current_title, current_text]


ARTICLES_FILEPATH = os.path.expanduser('~/data/articles.pickle')
with open(ARTICLES_FILEPATH, 'rb') as file:
    articles = pickle.load(file)


RECOMMENDED_FILEPATH = os.path.expanduser('~/data/recommended.pickle')
with open(RECOMMENDED_FILEPATH, 'rb') as file:
    recommended = pickle.load(file)


# for local debug
if __name__ == '__main__':
    app.run(debug=True)
