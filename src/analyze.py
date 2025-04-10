import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from wordcloud import WordCloud

PROCESSED_DATA_DIR = "./processed_data"
CHARTS_DIR = "./processed_data/charts"

def load_data():
	"""Load processed data splits"""
	return {
		"train": pd.read_csv(Path(PROCESSED_DATA_DIR)/"train.csv"),
		"validation": pd.read_csv(Path(PROCESSED_DATA_DIR)/"validation.csv"),
		"test": pd.read_csv(Path(PROCESSED_DATA_DIR)/"test.csv")
	}

def parse_text_components(data):
	"""Parse structured text components for each data split"""
	parsed_data = {}
	
	for split_name, df in data.items():
		def _parse(text):
			components = {"symptoms": [], "medications": [], "allergies": []}
			current_section = None
			
			for part in text.split('; '):
				if part.startswith("Symptoms: "):
					current_section = "symptoms"
					content = part[len("Symptoms: "):]
				elif part.startswith("Medications: "):
					current_section = "medications"
					content = part[len("Medications: "):]
				elif part.startswith("Allergies: "):
					current_section = "allergies"
					content = part[len("Allergies: "):]
				else:
					content = part
				
				if current_section == "symptoms":
					components["symptoms"] = [s.strip() for s in content.split(', ') if s.strip()]
				elif current_section == "medications":
					components["medications"] = [m.strip() for m in content.split(', ') if m.strip()]
				elif current_section == "allergies":
					components["allergies"] = [a.strip() for a in content.split(', ') if a.strip()]
					
			return components
		
		# Parse text column and merge components
		parsed = df["text"].apply(_parse).apply(pd.Series)
		parsed_data[split_name] = pd.concat([df, parsed], axis=1)
	
	return parsed_data

def analyze_data(data):
	"""Generate analysis visualizations"""
	# Create full dataset for some analyses
	full_df = pd.concat([data["train"], data["validation"], data["test"]])
	
	# 1. Training Set Distribution
	plt.figure(figsize=(10, 6))
	ax = sns.countplot(data=data["train"], x="label", order=["COVID-19", "non-COVID"])
	plt.title("Balanced Training Set Class Distribution")
	plt.xlabel("Diagnosis")
	plt.ylabel("Count")
	
	total = len(data["train"])
	for p in ax.patches:
		percentage = f'{100 * p.get_height()/total:.1f}%'
		ax.annotate(percentage, (p.get_x() + p.get_width()/2, p.get_height()),
					ha='center', va='bottom')
	plt.savefig(f"{CHARTS_DIR}/training_distribution.png", bbox_inches='tight')
	plt.close()

	# 2. Overall Distribution
	plt.figure(figsize=(10, 6))
	ax = sns.countplot(data=full_df, x="label", order=["COVID-19", "non-COVID"])
	plt.title("Balanced Overall Dataset Class Distribution")
	plt.xlabel("Diagnosis")
	plt.ylabel("Count")
	
	total = len(full_df)
	for p in ax.patches:
		percentage = f'{100 * p.get_height()/total:.1f}%'
		ax.annotate(percentage, (p.get_x() + p.get_width()/2, p.get_height()),
					ha='center', va='bottom')
	plt.savefig(f"{CHARTS_DIR}/overall_distribution.png", bbox_inches='tight')
	plt.close()

	# 3. Word Clouds
	fig, axes = plt.subplots(1, 2, figsize=(20, 10))
	
	# COVID cases
	covid_text = " ".join(full_df[full_df["label"] == "COVID-19"]["text"])
	wordcloud = WordCloud(width=800, height=400, background_color='white').generate(covid_text)
	axes[0].imshow(wordcloud, interpolation='bilinear')
	axes[0].set_title("COVID-19 Cases Word Cloud")
	axes[0].axis('off')
	
	# Non-COVID cases
	non_covid_text = " ".join(full_df[full_df["label"] == "non-COVID"]["text"])
	wordcloud = WordCloud(width=800, height=400, background_color='white').generate(non_covid_text)
	axes[1].imshow(wordcloud, interpolation='bilinear')
	axes[1].set_title("Non-COVID Cases Word Cloud")
	axes[1].axis('off')
	
	plt.savefig(f"{CHARTS_DIR}/wordclouds.png")
	plt.close()

	# 4. Text Length Analysis
	full_df["text_length"] = full_df["text"].apply(len)
	plt.figure(figsize=(12, 6))
	sns.boxplot(data=full_df, x="label", y="text_length", showfliers=False,
				order=["COVID-19", "non-COVID"])
	plt.title("Text Length Distribution by Diagnosis")
	plt.xlabel("Diagnosis")
	plt.ylabel("Text Length (characters)")
	plt.savefig(f"{CHARTS_DIR}/text_lengths.png")
	plt.close()

def main():
	Path(CHARTS_DIR).mkdir(parents=True, exist_ok=True)
	print("Loading data...")
	raw_data = load_data()
	print("Parsing text components...")
	parsed_data = parse_text_components(raw_data)
	print("Analyzing data and creating charts...")
	analyze_data(parsed_data)
	print(f"Analysis complete! Results saved to {CHARTS_DIR}")

if __name__ == "__main__":
	main()
