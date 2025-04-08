# Preprocess

import pandas as pd
from datasets import Dataset
from pathlib import Path
from tqdm import tqdm

DATA_DIR = "data/"
OUTPUT_DIR = "dataset/"
TRAIN_TEST_SPLIT = 0.2
VAL_TEST_SPLIT = 0.1
RANDOM_SEED = 42

data_dir = Path(DATA_DIR)
csv_files = {
    "allergies": data_dir / "allergies.csv",
    "careplans": data_dir / "careplans.csv",
    "conditions": data_dir / "conditions.csv",
    "medications": data_dir / "medications.csv",
    "observations": data_dir / "observations.csv",
    "encounters": data_dir / "encounters.csv",
}

# Load CSV files into DataFrames
dfs = {key: pd.read_csv(file) for key, file in csv_files.items()}

# Define COVID-19 related codes
covid_codes = {"840539006": "COVID-19", "840544004": "Suspected COVID-19"}

def process_encounter(enc_id, patient_id, dfs):
    """Process data for a single encounter into a text string and label."""
    # Extract encounter-specific data
    enc_conditions = dfs["conditions"][dfs["conditions"]["ENCOUNTER"] == enc_id]
    enc_medications = dfs["medications"][dfs["medications"]["ENCOUNTER"] == enc_id]
    enc_allergies = dfs["allergies"][dfs["allergies"]["ENCOUNTER"] == enc_id]
    enc_observations = dfs["observations"][dfs["observations"]["ENCOUNTER"] == enc_id]
    enc_careplans = dfs["careplans"][(dfs["careplans"]["ENCOUNTER"] == enc_id) & 
                                    (dfs["careplans"]["PATIENT"] == patient_id)]

    # Symptoms from conditions and observations
    symptoms = []
    if not enc_conditions.empty:
        symptoms.extend(enc_conditions["DESCRIPTION"].tolist())
    if not enc_observations.empty:
        obs = enc_observations[["DESCRIPTION", "VALUE", "UNITS"]].dropna()
        symptoms.extend([f"{row['DESCRIPTION']}: {row['VALUE']} {row['UNITS']}" 
                         for _, row in obs.iterrows()])

    # Medications
    meds = enc_medications["DESCRIPTION"].tolist() if not enc_medications.empty else ["none"]

    # Allergies
    allergies = enc_allergies["DESCRIPTION"].tolist() if not enc_allergies.empty else ["none"]

    # Determine label (COVID-19 or not)
    label = "non-COVID"
    if not enc_conditions.empty:
        if enc_conditions["CODE"].isin(covid_codes.keys()).any():
            label = "COVID-19"
    elif not enc_careplans.empty:
        if enc_careplans["REASONCODE"].isin(covid_codes.keys()).any():
            label = "COVID-19"

    # Construct input text
    input_text = (
        f"Symptoms: {', '.join(symptoms) if symptoms else 'none'}; "
        f"Medications: {', '.join(meds)}; "
        f"Allergies: {', '.join(allergies)}"
    )
    
    return {"text": input_text, "label": label}

def main():
    # Process all encounters
    encounter_data = []
    for _, enc in tqdm(dfs["encounters"].iterrows(), total=len(dfs["encounters"]), desc="Processing encounters"):
        result = process_encounter(enc["Id"], enc["PATIENT"], dfs)
        encounter_data.append(result)

    # Convert to Hugging Face Dataset
    dataset = Dataset.from_list(encounter_data)

    # Split into train, validation, and test sets
    train_test = dataset.train_test_split(test_size=TRAIN_TEST_SPLIT, seed=RANDOM_SEED)
    train_val = train_test["train"].train_test_split(test_size=VAL_TEST_SPLIT, seed=RANDOM_SEED)

    final_dataset = {
        "train": train_val["train"],
        "validation": train_val["test"],
        "test": train_test["test"]
    }

    # Save the dataset
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(exist_ok=True)
    for split, ds in final_dataset.items():
        ds.to_csv(output_dir / f"{split}.csv", index=False)

    print("Data preparation complete. Dataset saved to:", output_dir)
    print("Train size:", len(final_dataset["train"]))
    print("Validation size:", len(final_dataset["validation"]))
    print("Test size:", len(final_dataset["test"]))

if __name__ == '__main__':
    main()
