import json
from flask import Flask, request
from extract import get_keywords
from decode_seq2seq import get_questions
from qa import get_answers


app = Flask(__name__)


@app.route('/questions', methods=['POST'])
def list_questions():
    input = request.get_json()
    passage = input['passage']
    get_keywords(passage)
    get_questions()
    get_answers(passage)

    with open('questions.txt', 'r') as f:
        questions = f.read().splitlines()
    
    return {'qnlist': questions}

@app.route('/answers', methods=['POST'])
def show_answers():
    with open('questions.txt', 'r') as f:
        questions = f.read().splitlines()
    with open('answers.txt', 'r') as f:
        answers = f.read().splitlines()
    return {'qnlist': questions, 'anslist': answers}