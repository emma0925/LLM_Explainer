from flask import Flask, jsonify, make_response, request, send_file
from flask_cors import CORS
import requests
import os
import time
import requests
import json
import pandas as pd
import numpy as np
import random
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib
from io import BytesIO
import base64
matplotlib.use('Agg')
from io import BytesIO


app = Flask(__name__)
# Apply CORS to all routes, allowing all origins
CORS(app)

# Helper function to load and preprocess the dataset
def load_and_preprocess_data(file_path):
    dataset = pd.read_csv(file_path)
    X = dataset.loc[:, ["baths", "bedrooms"]]
    ksqft = dataset.sqft.apply(lambda x: x/1000)
    X["ksqft"] = ksqft
    y = (dataset["price"]/1000).values

    X_scaler = StandardScaler().fit(X)
    X_scaled = X_scaler.transform(X)
    X_scaled[:, 0] *= -1

    y_scaler = StandardScaler().fit(y.reshape(-1, 1))
    y_scaled = y_scaler.transform(y.reshape(-1, 1))

    return X, y, X_scaled, y_scaled, X_scaler, y_scaler

# Helper function to split the data
def split_data(X, y, X_scaled, y_scaled, test_size=0.2, random_state=0):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled = train_test_split(X_scaled, y_scaled, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled

# Helper function to train the model
def train_model(X_train_scaled, y_train_scaled):
    model = LinearRegression(fit_intercept=False).fit(X_train_scaled, y_train_scaled)
    return model

# Helper function to make predictions
def make_predictions(model, X_train_scaled, X_test_scaled, y_scaler, y_train_scaled):
    yp_train = model.predict(X_train_scaled)
    yp_test = model.predict(X_test_scaled)
    y_train_pred = y_scaler.inverse_transform(yp_train)
    y_train_true = y_scaler.inverse_transform(y_train_scaled.reshape(-1, 1))
    return y_train_pred, y_train_true, yp_test

# Helper function to simulate user
def simulate_user(fit, instances):
    return np.sum(np.array(fit)*instances, axis=1)

# Define Global Variables
file_path = "/Users/emmazhuang/Documents/Codes/CSCD95/nyc_housing_data.csv"
X, y, X_scaled, y_scaled, X_scaler, y_scaler = load_and_preprocess_data(file_path)
X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled = split_data(X, y, X_scaled, y_scaled)
model = train_model(X_train_scaled, y_train_scaled)
y_train_pred, y_train_true, yp_test = make_predictions(model, X_train_scaled, X_test_scaled, y_scaler, y_train_scaled)

# For Displaying the Data
def model_with_AI_Pred(X_train_scaled, yp_test, n=5, seed=0):
    random.seed(seed)
    indices = random.sample(range(len(yp_test)), n)
    instances = X_test_scaled[indices]
    ai_preds = yp_test[indices].squeeze()
    
    json_data = {}
    for i in range(n):
        json_data[f"item_{i+1}"] = {
            "Feature Values": {
                    "Feature 1": round(instances[i, 0],2),
                    "Feature 2": round(instances[i, 1], 2),
                    "Feature 3": round(instances[i, 2],2),
                },
            "AI Predictions": round(ai_preds[i], 2)
        }

    json_string = json.dumps(json_data, indent=4)
    return json_string

def generate_responses(X_train_scaled, user_input, yp_test, n=5, seed=0):
    random.seed(seed)
    indices = random.sample(range(len(yp_test)), n)
    instances = X_test_scaled[indices]
    ai_preds = yp_test[indices].squeeze()
    good_ans = True
    user_ans = user_input.split()
    user_ans_cleaned = []
    for i in user_ans:
        if i.isdigit():
            user_ans_cleaned.append(int(i))
        else:
            good_ans =False
            break
    
    if len(user_ans_cleaned) != 3:
        good_ans= False

    print(user_ans_cleaned)
    if good_ans:
        responses = simulate_user(fit=user_ans_cleaned, instances=instances)
    else:
        responses = simulate_user(fit=[0, 0, 0], instances=instances)

    json_data = {}
    for i in range(n):
        error= responses[i]-ai_preds[i]
        pos_neg = ' higher '
        if error < 0:
            pos_neg = ' lower '
            error = error*-1
        json_data[f"item_{i+1}"] = {
            "Feature Values": {
                    "Feature 1": instances[i, 0],
                    "Feature 2": instances[i, 1],
                    "Feature 3": instances[i, 2],
                },
            "User Estimation Error": "The user estimation is " + str(error) + pos_neg + "compared to AI Prediction"
        }

    json_string = json.dumps(json_data, indent=4)
    return json_string

# For part 3, What do you think the feature weights are?
@app.route('/submit', methods=['POST'])
def submit():
    data = request.get_json()
    user_input = data.get('prompt', 'No prompt provided')

    system_prompt = """You are a data analyst who will help non-technical users of AI systems understand the influence of 
                    data features on the prediction made by a blackbox AI. You will be given a JSON file of instances with
                    the following attributes: a list of data features and their values, and the user's 
                    estimation's error compared to AI's prediction. You will provide helpful, concise, and actionable feedback 
                    for the user to improve their understanding of how the AI makes predictions. 
                 """

    question = """\nExamine the user estimations error and directly provide instructions to the user on how they can improve their understanding of feature importances. Focus on the biggest areas of error, e.g. if the user has errorenous understanding of the direction of correlation or the mangnitude of influence. 
                      Please phrase the output in 2nd person, addressing the user directly. ONLY TWO SHORT SENTENCES. \n\n
                   """+generate_responses(X_train_scaled, user_input, yp_test, n=4, seed=0)

    # Set up the headers and data payload for the OpenAI API call
    headers = {
        'Content-Type': 'application/json',
        'Authorization': 'Bearer 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'
    }
    payload = {
        'model': 'gpt-4o',  # Specify the model you want to use
        'messages': [{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': question}],
        'max_tokens': 150  # You can adjust max_tokens as per your requirements
    }
    
    # Make the POST request to the OpenAI API
    response = requests.post('https://api.openai.com/v1/chat/completions', json=payload, headers=headers)
    print(response)
    # Check if the request to the OpenAI API was successful
    if response.status_code == 200:
        # Parse the response from OpenAI
        gpt_response = response.json()
        # Extract the text from the response
        # gpt_text = gpt_response['choices'][0]['text']
        # Return the GPT-4 generated text as JSON
        print(gpt_response['choices'][0]['message']['content'])
        return jsonify({'response': gpt_response['choices'][0]['message']['content']})
    else:
        # If the request failed, return an error message and the status code
        return jsonify({'error': 'Failed to fetch response from OpenAI', 'status_code': response.status_code})
    
@app.route('/question', methods=['POST'])
def question():
    data = request.get_json()
    user_input = data.get('prompt', 'No prompt provided')

    system_prompt = """You are a data analyst who will help non-technical users of AI systems understand the influence of 
                    data features on the prediction made by a blackbox AI. Here is the information about the blackbox"""+model_with_AI_Pred(X_train_scaled, yp_test, n=4, seed=0)
    # Set up the headers and data payload for the OpenAI API call
    headers = {
        'Content-Type': 'application/json',
        'Authorization': 'Bearer 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'  # Replace YOUR_OPENAI_API_KEY with your actual OpenAI API key
    }
    payload = {
        'model': 'gpt-4o',  # Specify the model you want to use
        'messages': [{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': user_input}],
        'max_tokens': 150  # You can adjust max_tokens as per your requirements
    }
    
    # Make the POST request to the OpenAI API
    response = requests.post('https://api.openai.com/v1/chat/completions', json=payload, headers=headers)
    print(response)
    # Check if the request to the OpenAI API was successful
    if response.status_code == 200:
        gpt_response = response.json()
        print(gpt_response['choices'][0]['message']['content'])
        return jsonify({'response': gpt_response['choices'][0]['message']['content']})
    else:
        # If the request failed, return an error message and the status code
        return jsonify({'error': 'Failed to fetch response from OpenAI', 'status_code': response.status_code})

@app.route('/display', methods=['GET'])
def display():
    json_output = model_with_AI_Pred(X_train_scaled, yp_test, n=4, seed=0)
    return jsonify(json.loads(json_output))

@app.route('/generate_code', methods=['POST'])
def generate_code():
    data = request.json
    question = data['question']
    dataset_path = '/Users/emmazhuang/Documents/Codes/CSCD95/nyc_housing_data.csv'
    dataset = pd.read_csv(dataset_path)
    sample_data = dataset.head().to_dict(orient='records')
    columns = dataset.columns.tolist()
    dataset_info = f"Dataset columns: {columns}\nSample data: {sample_data}\n"

    prompt = f"""
    This is a SIDE QUERY. Answer the user's question about the dataset saved at /Users/emmazhuang/Documents/Codes/CSCD95/nyc_housing_data.csv. For your information, here is how the dataset looks like:

    {dataset_info}
    
    Try to support the user as much as you can by helping them in VERIFICATION of different assumptions about the data and the current output so far.
    
    DO NOT TRY TO SOLVE THE TASK. Just focus on the user's question. This is a SIDE QUERY. Also, use the dataset path directly in your code. 
    
    You can access all the variables, dataframes, and other objects that have been defined in the previous steps. So you can easily use them to answer the user's question.    
    ## Use the following Template:
    [start-code]
    <python_code>
    [end-code]

    Generate Python code that answers this question:
    {question}
    
    Next message should start with [start-code]
    """

    headers = {
        'Content-Type': 'application/json',
        'Authorization': 'Bearer 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'  # Replace YOUR_OPENAI_API_KEY with your actual OpenAI API key
    }
    payload = {
        'model': 'gpt-4o',  
        'messages': [{'role': 'user', 'content': prompt}],
        'max_tokens': 200 
    }
    
    response = requests.post('https://api.openai.com/v1/chat/completions', json=payload, headers=headers)

    if response.status_code != 200:
        return jsonify({'error': 'Failed to get response from OpenAI API', 'status_code': response.status_code}), response.status_code

    try:
        response_data = response.json()
        code = response_data['choices'][0]['message']['content'].strip().replace("[start-code]", "").replace("[end-code]", "").strip()
        code = code.replace('```python', '').replace('```', '').strip()
        code = code.replace('<python_code>', '').replace('</python_code>', '').strip()
    except (KeyError, IndexError) as e:
        return jsonify({'error': 'Failed to parse response from OpenAI API', 'details': str(e)}), 500

    return jsonify({'code': code})

@app.route('/execute_code', methods=['POST'])
def execute_code():
    data = request.json
    code = data['code']

    # Create an isolated environment to execute the generated code
    local_vars = {}
    exec(code, globals(), local_vars)

    # Assuming the code generates a plot and saves it to a buffer
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()

    # Encode the image to base64 to send it as a response
    img_str = base64.b64encode(buf.getvalue()).decode()
    img_url = f"data:image/png;base64,{img_str}"

    return jsonify({'imageUrl': img_url})

if __name__ == '__main__':
    app.run(debug=True, host='localhost', port=5001)

    
