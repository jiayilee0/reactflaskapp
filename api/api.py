import time
from flask import Flask
from extract import combine
from decode_seq2seq import decode_extractions


app = Flask(__name__)


@app.route('/questions')
def list_questions():
    combine("Hello, get this from the user input! It should be a long and annoying passage.")
    decode_extractions()
    with open('/backendfiles/questions', 'r') as f:
        questions = f.read().splitlines()
    return {'qnlist': questions}
