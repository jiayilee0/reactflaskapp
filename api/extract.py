from allennlp.predictors.predictor import Predictor
import allennlp_models
import allennlp_models.tagging

ner_predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/ner-elmo.2021-02-12.tar.gz")
openie_predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/openie-model.2020.03.26.tar.gz")

import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize

nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))


def get_extraction(passage):
    final_output = []
    s = sent_tokenize(passage)
    for sentence in s:
        final_output.append([])
        output = []
        ner_result = ner_predictor.predict(sentence)
        openie_result = openie_predictor.predict(sentence)
        for i in range(len(ner_result['tags'])):
            if ner_result['tags'][i] != 'O':
                output.append(ner_result['tags'][i])
            else:
                output.append('')
        for verb in openie_result['verbs']:
            for i in range(len(verb['tags'])):
                if 'TMP' in verb['tags'][i] or 'LOC' in verb['tags'][i] or '-V' in verb['tags'][i]:
                    output[i] += verb['tags'][i]
        for i in range(len(output)):
            if len(output[i]) > 0:
                if output[i][0] == 'B' or output[i][0] == 'U':
                    final_output[-1].append(ner_result['words'][i])
                else:
                    final_output[-1][-1] += ' ' + ner_result['words'][i]
    return final_output


def modify_extraction(passage, final_output):
    qg_input = []
    for i in range(len(final_output)):
        qg_input.append([])
        for j in final_output[i]:
            if j not in stop_words:
                qg_input[-1].append(j)
    final_input = []
    for i in range(len(qg_input)): 
        if len(qg_input[i]) != 0:
            keywords = ''
            for j in qg_input[i]:
                keywords += j + '; '
            keywords = keywords[:-2]
            final_input.append(passage[:-1] + ' [SEP] ' + keywords)
    return final_input


def writefile(model_input):
  with open('keywords.txt', 'w') as f:
    for i in model_input:
      f.write(i)
      f.write('\n')
  return None


def get_keywords(passage):
  keywords = get_extraction(passage)
  model_input = modify_extraction(passage, keywords)
  writefile(model_input)
  return None
