import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import Counter

# Constants
PROCESSED_DATA_DIR = "./processed_data"
CHART_DIR = "./processed_data/charts"
SPLITS = ["train", "validation", "test"]

def load_data():
    """Load the processed dataset splits into a dictionary of DataFrames."""
    data = {}
    for split in SPLITS:
        file_path = Path(PROCESSED_DATA_DIR) / f"{split}.csv"
        if file_path.exists():
            data[split] = pd.read_csv(file_path)
        else:
            print(f"Warning: {file_path} not found.")
    return data

def plot_label_distribution(data):
    """Plot bar chart for COVID vs Non-COVID distribution across splits."""
    # Combine all splits for overall distribution
    all_data = pd.concat([data[split] for split in data], ignore_index=True)
    
    # Bar chart
    plt.figure(figsize=(10, 6))
    sns.countplot(x="label", hue="label", data=all_data, palette="viridis", legend=False)
    plt.title("Distribution of COVID vs Non-COVID Samples (All Splits)")
    plt.xlabel("Label")
    plt.ylabel("Count")
    plt.savefig(Path(CHART_DIR) / "label_distribution_bar.png")
    plt.close()

def plot_symptom_frequency(data):
    """Plot the frequency of top symptoms (excluding diagnoses) across all encounters."""
    all_data = pd.concat([data[split] for split in data], ignore_index=True)
    
    # Extract symptoms from the 'text' column, focusing on symptoms_conditions part
    symptoms = []
    exclude_terms = {"none", "COVID-19", "Suspected COVID-19"}  # Diagnoses to exclude
    for text in all_data["text"]:
        # Get the symptoms part (before first ";") and split into conditions and observations
        symptom_part = text.split(";")[0].replace("Symptoms: ", "")
        # Split on ", " but only take the part before "¿" to focus on conditions
        conditions = symptom_part.split("¿")[0].split(", ")
        for condition in conditions:
            condition = condition.strip()
            # Skip diagnoses, "none", and non-symptom entries
            if condition in exclude_terms or "finding" in condition or "index" in condition:
                continue
            symptoms.append(condition)
    
    # Count symptom occurrences
    symptom_counts = Counter(symptoms)
    top_symptoms = dict(symptom_counts.most_common(10))  # Top 10 symptoms
    
    # Bar chart
    plt.figure(figsize=(12, 6))
    sns.barplot(x=list(top_symptoms.keys()), y=list(top_symptoms.values()), hue=list(top_symptoms.keys()), palette="viridis", legend=False)
    plt.title("Top 10 Most Frequent Symptoms (Excluding Diagnoses)")
    plt.xlabel("Symptom")
    plt.ylabel("Frequency")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(Path(CHART_DIR) / "symptom_frequency.png")
    plt.close()

def plot_medication_frequency(data):
    """Plot the frequency of top medications across all encounters."""
    all_data = pd.concat([data[split] for split in data], ignore_index=True)
    
    # Extract medications from the 'text' column
    medications = []
    for text in all_data["text"]:
        med_part = text.split(";")[1].replace("Medications: ", "").split(", ")
        medications.extend([med.strip() for med in med_part if med.strip() != "none"])
    
    # Count medication occurrences
    med_counts = Counter(medications)
    top_meds = dict(med_counts.most_common(10))  # Top 10 medications
    
    # Bar chart
    plt.figure(figsize=(12, 6))
    sns.barplot(x=list(top_meds.keys()), y=list(top_meds.values()), hue=list(top_meds.keys()), palette="coolwarm", legend=False)
    plt.title("Top 10 Most Frequent Medications")
    plt.xlabel("Medication")
    plt.ylabel("Frequency")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(Path(CHART_DIR) / "medication_frequency.png")
    plt.close()

def plot_text_length_distribution(data):
    """Plot the distribution of text length (in characters) across all samples."""
    all_data = pd.concat([data[split] for split in data], ignore_index=True)
    text_lengths = all_data["text"].str.len()
    
    # Histogram
    plt.figure(figsize=(10, 6))
    sns.histplot(text_lengths, bins=30, kde=True, color="purple")
    plt.title("Distribution of Text Length (Characters)")
    plt.xlabel("Text Length")
    plt.ylabel("Frequency")
    plt.savefig(Path(CHART_DIR) / "text_length_distribution.png")
    plt.close()

def main():
    # Create chart directory if it doesn't exist
    chart_dir = Path(CHART_DIR)
    chart_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("Loading processed dataset...")
    data = load_data()
    
    if not data:
        print("No data found. Please run preprocess.py first.")
        return
    
    # Generate visualizations
    print("Generating label distribution plot...")
    plot_label_distribution(data)
    
    print("Generating symptom frequency plot...")
    plot_symptom_frequency(data)
    
    print("Generating medication frequency plot...")
    plot_medication_frequency(data)
    
    print("Generating text length distribution plot...")
    plot_text_length_distribution(data)
    
    print(f"Visualizations saved to {CHART_DIR}")

if __name__ == "__main__":
    main()
