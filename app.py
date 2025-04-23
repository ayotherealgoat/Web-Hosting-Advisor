from flask import Flask, render_template, request
import pandas as pd
import pickle
import os

app = Flask(__name__)

# Load trained model and dataset
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model.pkl')
with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)

# Load hosting providers database
df = pd.read_csv('hosting_providers.csv')

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        # Create DataFrame directly from form data
        data = {
            'cost': [float(request.form['cost'])],
            'uptime': [float(request.form['uptime'])],
            'storage': [float(request.form['storage'])],
            'bandwidth': [float(request.form['bandwidth'])],
            'tech_stack': [float(request.form['tech_stack'])],
            'control_panel': [float(request.form['control_panel'])]
        }
        
        input_df = pd.DataFrame(data).astype('float64')

        # Load providers database
        df = pd.read_csv('hosting_providers.csv')
        
        # Convert unlimited values
        df['bandwidth'] = df['bandwidth'].apply(lambda x: 999999 if isinstance(x, str) and x.lower() == 'unlimited' else float(x))
        df['storage'] = df['storage'].apply(lambda x: 999999 if isinstance(x, str) and x.lower() == 'unlimited' else float(x))

        # Get matching providers
        matching_providers = df[
            (df['cost'] <= float(request.form['cost']) * 1.2) &
            (df['uptime'] >= float(request.form['uptime'])) &
            (df['storage'] >= float(request.form['storage'])) &
            (df['bandwidth'] >= float(request.form['bandwidth'])) &
            (df['recommended'] == 1)
        ].sort_values('cost').head(5)  # Show top 5 matches

        # Predict using model
        prediction = model.predict(input_df)[0]
        
        return render_template('resuults.html',
                             is_recommended=bool(prediction == 1),
                             inputs=data,
                             providers=matching_providers.to_dict('records'))
                             
    except Exception as e:
        print(f"Debug - Error: {str(e)}")  # Add debug print
        return f"An error occurred: {str(e)}", 500

@app.route('/recommend', methods=['GET'])
def recommend_get():
    return "Please submit the form from the home page.", 405

if __name__ == '__main__':
    app.run(debug=True)