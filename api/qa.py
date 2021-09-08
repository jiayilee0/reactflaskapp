from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch

tokenizer = AutoTokenizer.from_pretrained("deepset/bert-large-uncased-whole-word-masking-squad2")
model = AutoModelForQuestionAnswering.from_pretrained("deepset/bert-large-uncased-whole-word-masking-squad2", return_dict=False)


def get_answers(text):
    with open('questions.txt', 'r') as f:
        questions = f.read().splitlines()

    answers = []
    output_qns = []
    for question in questions:
        inputs = tokenizer.encode_plus(question, text, add_special_tokens=True, return_tensors="pt")
        input_ids = inputs["input_ids"].tolist()[0]

        text_tokens = tokenizer.convert_ids_to_tokens(input_ids)
        answer_start_scores, answer_end_scores = model(**inputs)

        answer_start = torch.argmax(answer_start_scores)
        answer_end = torch.argmax(answer_end_scores)

        answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))
        # print("answer: ", answer)
        if answer != '':
            answers.append(answer)
            output_qns.append(question)

    with open('questions.txt', 'w') as f:
        for qn in output_qns:
            f.write(qn)
            f.write('\n')

    with open('answers.txt', 'w') as f:
        for ans in answers:
            f.write(ans)
            f.write('\n')
