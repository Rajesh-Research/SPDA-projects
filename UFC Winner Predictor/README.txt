# UFC-Winner-Predictor

This repository contains the UFC-Winner-Predictor, a machine learning-based tool designed to predict UFC fight outcomes using a Decision Tree Classifier. The project includes an interactive Streamlit app for user-friendly predictions based on historical fighter data.

## Credits
- Data sourced from [UFC-MMA-Data](https://github.com/jansen88/ufc-data) by Jansen Chan.
- Developed by [Your Name].

## Files
- `ufc_predictor_new.py`: The main Python script for training the Decision Tree Classifier and saving the model.
- `model.pkl`: The trained Decision Tree model (compressed as `model.zip`).
- `preprocessed_ufc_df.csv`: Preprocessed dataset used for training.
- `app.py`: Streamlit app script for interactive predictions.
- `requirements.txt`: List of required Python dependencies.

## Setup Instructions

### Prerequisites
- Python 3.10 or higher.
- Git (optional, for cloning the repository).

### Installation Steps
1. **Clone the Repository** (optional):
   - Open a terminal and run:
     ```
     git clone https://github.com/ShreyashKumar07/UFC-Winner-Predictor
     cd ufc-predictor
     ```

2. **Install Dependencies**:
   - Ensure you have Python 3.10 installed. Verify with:
     ```
     python --version
     ```
   - Create a virtual environment (recommended):
     ```
     python -m venv venv
     source venv/bin/activate  # On Windows: venv\Scripts\activate
     ```
   - Install required packages from `requirements.txt`:
     ```
     pip install -r requirements.txt
     ```

3. **Unzip the Model File**:
   - Extract `model.zip` to obtain `model.pkl`:
     ```
     unzip model.zip
     ```

4. **Verify Files**:
   - Ensure `model.pkl`, `preprocessed_ufc_df.csv`, and `app.py` are in the same directory as `ufc_predictor_new.py`.

## Usage

### Training the Model (Optional)
- If you want to retrain the model, run:
  ```
  python ufc_predictor_new.py
  ```
  - This will overwrite `model.pkl` and `preprocessed_ufc_df.csv` with new data.

### Running the Streamlit App
- Launch the app with:
  ```
  streamlit run app.py
  ```
- Open your web browser and navigate to the provided local URL (e.g., `http://localhost:8501`).
- Select fighters from the dropdown menus, view stat comparisons, and get predictions.

## Troubleshooting
- **Dependency Issues**: If errors occur, ensure all packages in `requirements.txt` are installed with compatible versions (e.g., `numpy==1.23.5`, `pandas==1.5.3`, `scikit-learn==1.2.2`, `streamlit==1.29.0`).
- **Model Loading Errors**: Verify `model.pkl` is unzipped and in the correct directory.
- **Port Conflicts**: If `localhost:8501` is unavailable, change the port by adding `--server.port=8502` to the `streamlit run` command.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact
For issues or questions, contact [Your Name] at [your.email@example.com].