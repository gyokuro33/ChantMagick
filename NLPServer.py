# coding:utf-8

import spacy
from flask import Flask
from sklearn import svm
import csv
import numpy as np


app = Flask(__name__)
nlp = spacy.load('ja_ginza')
word_model = None

@app.route('/magick/<word>')
def hello(word=None):
    result = predict_data(word,word_model)
    return result

def read_csv(csv_file):
    line_num = 0
    with open(csv_file) as f:
        reader = csv.reader(f)
        for num, row in enumerate(reader):
            line_num += 1
    data_array = np.zeros((line_num, 100))
    target_array = np.zeros(line_num)
    with open(csv_file) as f:
        reader = csv.reader(f)
        for num, row in enumerate(reader):
            doc = nlp(row[0])
            for sent in doc.sents:
                for token in sent:
                    data_array[num] = token.vector
                    target_array[num] = row[1]
    return data_array, target_array

def learn_svm(data,target):
    clf = svm.SVC(gamma="scale")
    clf.fit(data, target)
    return clf

def predict_data(data, model):
    doc = nlp(data)
    result_dict = {}
    result_str = ""
    for sent in doc.sents:
        for token in sent:
            if token.pos_ == "NOUN" or token.pos_ == "VERB":
                print(token.text)
                print(model.predict(token.vector.reshape(1, -1)))
                result =  model.predict(token.vector.reshape(1, -1))
                result_dict[token.text] = int(result[0])
                if result_str == "":
                    result_str = str(int(result[0]))
                else:
                    result_str = result_str + ","+str(int(result[0]))
    return result_str

if __name__ == "__main__":
    data,target = read_csv("element.csv")
    word_model = learn_svm(data,target)
    app.run(debug=True)
