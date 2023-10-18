import tensorflow as tf
from flask import Flask, render_template, request, jsonify
import re
import pickle
import numpy as np
from nltk.corpus import stopwords
from profanity_filter import ProfanityFilter
from nltk.stem.porter import PorterStemmer

app = Flask(__name__)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/api/predict', methods=['POST'])
def api_predict():
    new_review = request.json.get('new_review', '')
    result = getPredictions(new_review)
    response = {'Review' : new_review, 'Sentiment': result}
    return jsonify(response)

@app.route('/', methods=['POST'])
def result():
    new_review = request.form['new_review']
    result = getPredictions(new_review)
    return render_template('result.html', result=result)

def getPredictions(new_review):
    ps = PorterStemmer()
    cv = pickle.load(open('SentimentAnalysisScaler.sav', 'rb'))
    loaded_model = tf.keras.models.load_model('SentimentAnalysisModel.h5')

    # pf = ProfanityFilter()
    # pf.censor_char = 'X'
    # new_review = pf.censor(new_review)
    # print("Review with removed profanity ", new_review)

    new_review = re.sub('[^a-zA-Z]', ' ', str(new_review))
    print(new_review)

    new_review = new_review.lower()
    new_review = new_review.split()
    all_stopwords = set(stopwords.words('english'))
    all_stopwords.remove('not')
    print(new_review)

    new_review = [ps.stem(word) for word in new_review if not word in set(all_stopwords)]
    new_review = ' '.join(new_review)
    print(new_review)

    new_corpus = [new_review]
    print(new_corpus)

    new_X_test = cv.transform(new_corpus).toarray()
    print(new_X_test)

    new_y_pred = loaded_model.predict(new_X_test).round()
    print('Predicted : ',new_y_pred)

    labels = ['Anger', 'Fear', 'Joy', 'Love', 'Sadness', 'Surprise']
    new_y_pred = np.argmax(new_y_pred)
    new_y_pred = [labels[new_y_pred]][0]
    print('Sentiment : ',new_y_pred)

    if new_y_pred == 'Anger':
        return 'Anger'
    elif new_y_pred == "Fear":
        return 'Fear'
    elif new_y_pred == 'Joy':
        return 'Joy'
    elif new_y_pred == 'Love':
        return 'Love'
    elif new_y_pred == 'Sadness':
        return 'Sadness'
    elif new_y_pred == 'Surprise':
        return 'Surprise'
    else:
        return 'error'

if __name__ == "__main__":
    app.run(debug=True)


# To test this REST API, we can use a tool like curl or a web client(ex- Postman) to send POST requests to your endpoint at http://127.0.0.1:5000/api/predict. 
# We should send a JSON payload with a 'new_review' field to get the prediction result.

# Command example curl -X POST -H "Content-Type: application/json" -d @payload.json http://127.0.0.1:5000/api/predict
# Output= { result = good/bad }
