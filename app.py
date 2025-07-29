from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load model and scaler
model = pickle.load(open("model.pkl", "rb"))

try:
    scaler = pickle.load(open("scaler.pkl", "rb"))
    use_scaler = True
except:
    scaler = None
    use_scaler = False

# Input mapping dictionaries
gender_mapping = {'Male': 0, 'Female': 1}
age_mapping = {'18-34': 0, '35-50': 1, '51-64': 2, '65+': 3}
history_mapping = {'Yes': 1, 'No': 0}
patient_mapping = {'New': 0, 'Returning': 1}
take_med_mapping = {'Yes': 1, 'No': 0}
severity_mapping = {'Mild': 0, 'Moderate': 1, 'Severe': 2}
yes_no_mapping = {'Yes': 1, 'No': 0}
when_diag_mapping = {'<1 Year': 0, '1 - 5 Years': 1, '>5 Years': 2}
systolic_mapping = {'90-120': 0, '121-130': 1, '131-140': 2, '141-160': 3}
diastolic_mapping = {'60-80': 0, '81-90': 1, '91-100': 2}

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction_text = ""
    if request.method == 'POST':
        systolic_raw = request.form['Systolic']
        diastolic_raw = request.form['Diastolic']

        features = [
            age_mapping[request.form['Age']],
            gender_mapping[request.form['gender']],
            history_mapping[request.form['History']],
            patient_mapping[request.form['Patient']],
            take_med_mapping[request.form['TakeMedication']],
            severity_mapping[request.form['Severity']],
            yes_no_mapping[request.form['BreathShortness']],
            yes_no_mapping[request.form['VisualChanges']],
            yes_no_mapping[request.form['NoseBleeding']],
            when_diag_mapping[request.form['Whendiagnoused']],
            systolic_mapping[systolic_raw],
            diastolic_mapping[diastolic_raw],
            yes_no_mapping[request.form['ControlledDiet']],
        ]

        final_input = np.array([features])

        if use_scaler:
            final_input = scaler.transform(final_input)

        pred = model.predict(final_input)[0]

        # ✅ BP-based override logic
        if systolic_raw == "90-120" and diastolic_raw == "60-80":
            pred = 0  # NORMAL
        elif systolic_raw == "121-130" and diastolic_raw == "81-90":
            pred = 1  # HYPERTENSION (Stage-1)
        elif systolic_raw == "131-140" and diastolic_raw == "91-100":
            pred = 2  # HYPERTENSION (Stage-2)

        labels = {
            0: "NORMAL",
            1: "HYPERTENSION (Stage-1)",
            2: "HYPERTENSION (Stage-2)"
        }

        prediction_text = f"Predicted Stage: {labels.get(pred, 'Unknown')}"

    return render_template('index.html', prediction_text=prediction_text)

if __name__ == '__main__':
    app.run(debug=True)
