import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
app = Flask(__name__)
model = pickle.load(open('hasil_class.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/classification',methods=['POST'])
def classification():

    final_features = [x for x in request.form.values()] 
    prediction = model.predict(final_features)

    output = prediction[0]

    return render_template('index.html', prediction_text='Classification Result: {}'.format(output))

@app.route('/results',methods=['POST'])
def results():

    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)