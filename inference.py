import pandas as pd
import numpy as np
import joblib
import argparse
from sklearn.preprocessing import LabelEncoder
from feature_extraction import generate_features, LABELS_SSVEP

def run_inference(data_path, output_file):
    """
    Loads trained models and test data, generates predictions, and saves them to a CSV file.
    """
    print("--- Starting Inference ---")

    # --- MI Inference ---
    print("Running MI inference...")
    mi_model = joblib.load("mi_knn_model_72acc.pkl")
    X_test_mi, ids_mi = generate_features(data_path, 'test_metadata.csv', 'MI')
    
    mi_predictions_numeric = mi_model.predict(X_test_mi)
    mi_predictions_labels = ["Left" if pred == 0 else "Right" for pred in mi_predictions_numeric]
    
    mi_results = pd.DataFrame({'id': ids_mi, 'label': mi_predictions_labels})
    print(f"Generated {len(mi_results)} predictions for MI task.")

    # --- SSVEP Inference ---
    print("\nRunning SSVEP inference...")
    ssvep_model = joblib.load("ssvep_random_forest_model_56acc_56f1.pkl")
    X_test_ssvep, ids_ssvep = generate_features(data_path, 'test_metadata.csv', 'SSVEP')
    
    ssvep_predictions_numeric = ssvep_model.predict(X_test_ssvep)
    
    # Inverse transform labels
    label_encoder = LabelEncoder().fit(LABELS_SSVEP)
    ssvep_predictions_labels = label_encoder.inverse_transform(ssvep_predictions_numeric)
    
    ssvep_results = pd.DataFrame({'id': ids_ssvep, 'label': ssvep_predictions_labels})
    print(f"Generated {len(ssvep_results)} predictions for SSVEP task.")

    # --- Combine and Save ---
    final_predictions = pd.concat([mi_results, ssvep_results]).sort_values('id').reset_index(drop=True)
    
    final_predictions.to_csv(output_file, index=False)
    print(f"\nSuccessfully saved {len(final_predictions)} total predictions to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate predictions for EEG test set.")
    parser.add_argument('--data_path', type=str, default='./', help='Path to the directory containing the data and metadata CSVs.')
    parser.add_argument('--output_file', type=str, default='predictions.csv', help='File path to save the final predictions.')
    args = parser.parse_args()

    run_inference(args.data_path, args.output_file)
