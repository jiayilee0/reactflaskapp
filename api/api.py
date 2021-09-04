import time
from flask import Flask
from backendfiles/extract import combine
from decode_seq2seq import decode_extractions


app = Flask(__name__)


@app.route('/questions')
def list_questions():
    combine(passage)
    decode_extractions()
    with open('/backendfiles/questions', 'r') as f:
        questions = f.read().splitlines()
    return {'qnlist': questions}
