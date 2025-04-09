import pandas as pd
from datasets import Dataset
from pathlib import Path
from tqdm import tqdm

# Constants
TEST_SPLIT = 0.2
VAL_SPLIT = 0.1
RANDOM_SEED = 42
RAW_DATA_DIR = "./raw_data"
PROCESSED_DATA_DIR = "./processed_data"

# Define file paths
data_dir = Path(RAW_DATA_DIR)
csv_files = {
    "allergies": data_dir / "allergies.csv",
    "careplans": data_dir / "careplans.csv",
    "conditions": data_dir / "conditions.csv",
    "medications": data_dir / "medications.csv",
    "observations": data_dir / "observations.csv",
    "encounters": data_dir / "encounters.csv",
}

# Load CSV files into DataFrames with string dtype for codes and set indices
dfs = {
    "allergies": pd.read_csv(csv_files["allergies"], dtype={"CODE": str}).set_index(["PATIENT", "ENCOUNTER"]),
    "careplans": pd.read_csv(csv_files["careplans"], dtype={"CODE": str, "REASONCODE": str}).set_index(["PATIENT", "ENCOUNTER"]),
    "conditions": pd.read_csv(csv_files["conditions"], dtype={"CODE": str}).set_index("ENCOUNTER"),
    "medications": pd.read_csv(csv_files["medications"], dtype={"CODE": str}).set_index("ENCOUNTER"),
    "observations": pd.read_csv(csv_files["observations"], dtype={"CODE": str}).set_index("ENCOUNTER"),
    "encounters": pd.read_csv(csv_files["encounters"]).set_index("Id"),
}

# Define COVID-19 related codes
covid_codes = {"840539006": "COVID-19", "840544004": "Suspected COVID-19"}

def preprocess_data(dfs):
    """Preprocess all data into a single DataFrame per encounter with progress tracking."""
    # Aggregation steps with progress
    print("Aggregating data...")
    aggregations = {
        "conditions": dfs["conditions"].groupby("ENCOUNTER").agg({
            "DESCRIPTION": lambda x: ", ".join(x) if len(x) > 0 else "none",
            "CODE": list
        }).rename(columns={"DESCRIPTION": "symptoms_conditions", "CODE": "condition_codes"}),
        "medications": dfs["medications"].groupby("ENCOUNTER").agg({
            "DESCRIPTION": lambda x: ", ".join(x) if len(x) > 0 else "none"
        }).rename(columns={"DESCRIPTION": "medications"}),
        "allergies": dfs["allergies"].groupby(["PATIENT", "ENCOUNTER"]).agg({
            "DESCRIPTION": lambda x: ", ".join(x) if len(x) > 0 else "none"
        }).rename(columns={"DESCRIPTION": "allergies"}),
        "careplans": dfs["careplans"].groupby(["PATIENT", "ENCOUNTER"]).agg({
            "REASONCODE": list
        }).rename(columns={"REASONCODE": "careplan_codes"})
    }

    # Handle observations separately with tqdm for finer progress
    print("Aggregating observations...")
    obs_groups = dfs["observations"].groupby("ENCOUNTER")
    observations_agg = pd.DataFrame(index=obs_groups.groups.keys(), columns=["symptoms_observations"])
    observations_agg.index.name = "ENCOUNTER"  # Explicitly name the index
    for enc in tqdm(obs_groups.groups.keys(), desc="Processing observations"):
        group = obs_groups.get_group(enc)
        if not group.empty:
            observations_agg.loc[enc, "symptoms_observations"] = "¿ ".join(
                f"{row['DESCRIPTION']}: {row['VALUE']} {row['UNITS']}"
                for _, row in group[["DESCRIPTION", "VALUE", "UNITS"]].dropna().iterrows()
            )
        else:
            observations_agg.loc[enc, "symptoms_observations"] = "none"

    # Merge all into one DataFrame with progress
    print("Merging data...")
    data = dfs["encounters"][["PATIENT"]].reset_index()
    merge_steps = [
        ("conditions", aggregations["conditions"], {"symptoms_conditions": "none"}, ["Id"], ["ENCOUNTER"]),
        ("medications", aggregations["medications"], {"medications": "none"}, ["Id"], ["ENCOUNTER"]),
        ("allergies", aggregations["allergies"], {"allergies": "none"}, ["PATIENT", "Id"], ["PATIENT", "ENCOUNTER"]),
        ("observations", observations_agg, {"symptoms_observations": "none"}, ["Id"], ["ENCOUNTER"]),
        ("careplans", aggregations["careplans"], {}, ["PATIENT", "Id"], ["PATIENT", "ENCOUNTER"]),
    ]

    for name, agg_df, fill_values, left_cols, right_cols in tqdm(merge_steps, desc="Merging steps"):
        agg_df_reset = agg_df.reset_index()
        # Merge and drop the right join column to avoid duplicates
        data = data.merge(agg_df_reset, left_on=left_cols, right_on=right_cols, how="left").fillna(fill_values)
        # Drop the right_cols that aren’t needed post-merge (e.g., ENCOUNTER from right side)
        cols_to_drop = [col for col in right_cols if col not in left_cols and col in data.columns]
        data = data.drop(columns=cols_to_drop)
        # Handle list columns post-merge
        if "condition_codes" in agg_df.columns:
            data["condition_codes"] = data["condition_codes"].apply(lambda x: x if isinstance(x, list) else [])
        if "careplan_codes" in agg_df.columns:
            data["careplan_codes"] = data["careplan_codes"].apply(lambda x: x if isinstance(x, list) else [])

    # Combine symptoms and format text
    print("Formatting final text...")
    data["symptoms"] = data.apply(
        lambda row: f"{row['symptoms_conditions']}{', ' if row['symptoms_conditions'] != 'none' and row['symptoms_observations'] != 'none' else ''}{row['symptoms_observations']}",
        axis=1
    )
    data["label"] = data.apply(
        lambda row: "COVID-19" if (any(code in covid_codes for code in row["condition_codes"]) or 
                                  any(code in covid_codes for code in row["careplan_codes"])) else "non-COVID",
        axis=1
    )
    data["text"] = data.apply(
        lambda row: f"Symptoms: {row['symptoms']}; Medications: {row['medications']}; Allergies: {row['allergies']}",
        axis=1
    )

    return data[["text", "label"]]

def main():
    # Process data
    print("Preprocessing data...")
    processed_df = preprocess_data(dfs)

    # Convert to Hugging Face Dataset
    dataset = Dataset.from_pandas(processed_df.reset_index(drop=True))

    # Split into train, validation, and test sets
    train_test = dataset.train_test_split(test_size=TEST_SPLIT, seed=RANDOM_SEED)
    train_val = train_test["train"].train_test_split(test_size=VAL_SPLIT, seed=RANDOM_SEED)

    final_dataset = {
        "train": train_val["train"],
        "validation": train_val["test"],
        "test": train_test["test"]
    }

    # Save the dataset
    output_dir = Path(PROCESSED_DATA_DIR)
    output_dir.mkdir(exist_ok=True)
    for split, ds in final_dataset.items():
        ds.to_csv(output_dir / f"{split}.csv", index=False)

    # Print stats
    print("Label distribution:", pd.Series(processed_df["label"]).value_counts())
    print("Data preparation complete. Dataset saved to:", output_dir)
    print("Train size:", len(final_dataset["train"]))
    print("Validation size:", len(final_dataset["validation"]))
    print("Test size:", len(final_dataset["test"]))

if __name__ == "__main__":
    main()
