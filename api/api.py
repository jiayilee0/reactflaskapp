import json
from flask import Flask, request
from extract import combine
from decode_seq2seq import decode_extractions


app = Flask(__name__)


@app.route('/questions'), methods=['POST'])
def list_questions():
    input = request.get_json()
    combine(input['passage'])
    decode_extractions()
    with open('questions.txt', 'r') as f:
        questions = f.read().splitlines()
    return {'qnlist': questions}
