import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap

# Load the dataset
file_path = 'data/kennedy/Measuring Hate Speech.csv'  # Ensure this is the correct path to your CSV file
df = pd.read_csv(file_path)

# Ensure the output directory exists
output_dir = 'analysis_results/'
os.makedirs(output_dir, exist_ok=True)

# Function to determine the majority vote for a label
def majority_vote(series):
    return series.mode().sample(1).iloc[0]  # Picks randomly if there's a tie

# Aggregate label by comment_id to get the mean (rounded for simplicity in voting)
aggregated_df = df.groupby('comment_id').agg(
    aggregate_label=('hatespeech', majority_vote)
).reset_index()

# Merge aggregated labels back to the original dataset
df = df.merge(aggregated_df, on='comment_id', how='left')

# Add a column indicating if an annotator's label agrees with the aggregated label
df['agrees_with_aggregate'] = df['hatespeech'] == df['aggregate_label']

# Filter for cases where there is disagreement
disagreement_df = df[df['agrees_with_aggregate'] == False]

## Gender
gender_agreement = df.groupby(['annotator_gender']).agg(
    agreement_count=('agrees_with_aggregate', 'sum'),  # Total matches with aggregate label
    total_annotations=('agrees_with_aggregate', 'count')  # Total annotations per gender
).reset_index()

# Calculate the percentage agreement
gender_agreement['agreement_percentage'] = (gender_agreement['agreement_count'] / gender_agreement['total_annotations']) * 100

# Create the bar graph
plt.figure(figsize=(10, 6))

cmap = get_cmap("tab10")  # Choose a colormap
colors = [cmap(i) for i in range(len(gender_agreement))]

bars = plt.bar(
    gender_agreement['annotator_gender'],
    gender_agreement['agreement_percentage'],
    color=colors,
    edgecolor='black'
)

for bar, percentage in zip(bars, gender_agreement['agreement_percentage']):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        f'{percentage:.1f}%',  # Format as a percentage with 1 decimal
        ha='center',
        va='bottom'
    )

plt.xlabel('Annotator Gender')
plt.ylabel('Agreement Percentage')
plt.title('Agreement Percentage by Annotator Gender')
plt.xticks(rotation=45)
plt.tight_layout()

plot_path = os.path.join(output_dir, 'gender_agreement_bar_graph.png')
plt.savefig(plot_path)

print(f"Bar graph saved to {plot_path}")

# Disagreement Analysis
disagreement_distribution = disagreement_df.groupby(['annotator_gender', 'hatespeech']).size().reset_index(name='count')
total_disagreements_by_gender = disagreement_distribution.groupby('annotator_gender')['count'].transform('sum')
disagreement_distribution['percentage'] = (disagreement_distribution['count'] / total_disagreements_by_gender) * 100

labels = sorted(disagreement_distribution['hatespeech'].unique())
gender_list = disagreement_distribution['annotator_gender'].unique()

x = np.arange(len(labels))  # Base positions for hatespeech labels
width = 0.8 / len(gender_list)  # Divide bar width equally among genders

plt.figure(figsize=(10, 6))
for i, gender in enumerate(gender_list):
    subset = disagreement_distribution[disagreement_distribution['annotator_gender'] == gender]
    plt.bar(
        x + i * width - (width * len(gender_list)) / 2,  # Offset bars for each gender
        subset['percentage'],
        width=width,
        label=gender.capitalize()
    )

plt.xlabel('Hatespeech Label')
plt.ylabel('Percentage')
plt.title('Percentage of Hatespeech Labels in Disagreement Cases by Gender')
plt.xticks(sorted(disagreement_distribution['hatespeech'].unique()))
plt.legend(title='Annotator Gender')
plt.tight_layout()

plot_path = os.path.join(output_dir, 'gender_disagreement_hatespeech_distribution.png')
plt.savefig(plot_path)

print(f"Disagreement bar graph saved to {plot_path}")

# Save Gender Values
result_df = gender_agreement[['annotator_gender', 'agreement_count', 'agreement_percentage']]

output_path = os.path.join(output_dir, 'gender_agreement_analysis.csv')
result_df.to_csv(output_path, sep='\t', index=False)

print(f"Analysis results saved to {output_path}")


## Trans
trans_agreement = df.groupby(['annotator_trans']).agg(
    agreement_count=('agrees_with_aggregate', 'sum'),  # Total matches with aggregate label
    total_annotations=('agrees_with_aggregate', 'count')  # Total annotations per gender
).reset_index()

trans_agreement['agreement_percentage'] = (trans_agreement['agreement_count'] / trans_agreement['total_annotations']) * 100

# Create the bar graph
plt.figure(figsize=(10, 6))

cmap = get_cmap("tab10")  # Choose a colormap
colors = [cmap(i) for i in range(len(trans_agreement))]

bars = plt.bar(
    trans_agreement['annotator_trans'],
    trans_agreement['agreement_percentage'],
    color=colors,
    edgecolor='black'
)

for bar, percentage in zip(bars, trans_agreement['agreement_percentage']):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        f'{percentage:.1f}%',  # Format as a percentage with 1 decimal
        ha='center',
        va='bottom'
    )

plt.xlabel('Annotator Transgender Yes / No')
plt.ylabel('Agreement Percentage')
plt.title('Agreement Percentage by Annotator Transgender')
plt.xticks(rotation=45)
plt.tight_layout()

# Save the plot as a PNG file
plot_path = os.path.join(output_dir, 'transgender_agreement_bar_graph.png')
plt.savefig(plot_path)

print(f"Bar graph saved to {plot_path}")

# Disagreement Analysis
disagreement_distribution = disagreement_df.groupby(['annotator_trans', 'hatespeech']).size().reset_index(name='count')
total_disagreements_by_trans = disagreement_distribution.groupby('annotator_trans')['count'].transform('sum')
disagreement_distribution['percentage'] = (disagreement_distribution['count'] / total_disagreements_by_trans) * 100

labels = sorted(disagreement_distribution['hatespeech'].unique())
trans_list = disagreement_distribution['annotator_trans'].unique()

x = np.arange(len(labels))  # Base positions for hatespeech labels
width = 0.8 / len(trans_list)  # Divide bar width equally among trans

plt.figure(figsize=(10, 6))
for i, trans in enumerate(trans_list):
    subset = disagreement_distribution[disagreement_distribution['annotator_trans'] == trans]
    plt.bar(
        x + i * width - (width * len(trans_list)) / 2,  # Offset bars for each trans
        subset['percentage'],
        width=width,
        label=trans.capitalize()
    )

plt.xlabel('Hatespeech Label')
plt.ylabel('Percentage')
plt.title('Percentage of Hatespeech Labels in Disagreement Cases by Trans')
plt.xticks(sorted(disagreement_distribution['hatespeech'].unique()))
plt.legend(title='Annotator Trans')
plt.tight_layout()

plot_path = os.path.join(output_dir, 'trans_disagreement_hatespeech_distribution.png')
plt.savefig(plot_path)

print(f"Disagreement bar graph saved to {plot_path}")

# Save Transgender Values

result_df = trans_agreement[['annotator_trans', 'agreement_count', 'agreement_percentage']]

output_path = os.path.join(output_dir, 'trans_agreement_analysis.csv')
result_df.to_csv(output_path, sep='\t', index=False)

print(f"Analysis results saved to {output_path}")


## Education
educ_agreement = df.groupby(['annotator_educ']).agg(
    agreement_count=('agrees_with_aggregate', 'sum'),  # Total matches with aggregate label
    total_annotations=('agrees_with_aggregate', 'count')  # Total annotations per gender
).reset_index()

educ_agreement['agreement_percentage'] = (educ_agreement['agreement_count'] / educ_agreement['total_annotations']) * 100

# Create the bar graph
plt.figure(figsize=(10, 6))

cmap = get_cmap("tab10")  # Choose a colormap
colors = [cmap(i) for i in range(len(educ_agreement))]

bars = plt.bar(
    educ_agreement['annotator_educ'],
    educ_agreement['agreement_percentage'],
    color=colors,
    edgecolor='black'
)

for bar, percentage in zip(bars, educ_agreement['agreement_percentage']):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        f'{percentage:.1f}%',  # Format as a percentage with 1 decimal
        ha='center',
        va='bottom'
    )

plt.xlabel('Annotator Education')
plt.ylabel('Agreement Percentage')
plt.title('Agreement Percentage by Annotator Education')
plt.xticks(rotation=45)
plt.tight_layout()

# Save the plot as a PNG file
plot_path = os.path.join(output_dir, 'education_agreement_bar_graph.png')
plt.savefig(plot_path)

print(f"Bar graph saved to {plot_path}")

# Disagreement Analysis
disagreement_distribution = disagreement_df.groupby(['annotator_educ', 'hatespeech']).size().reset_index(name='count')
total_disagreements_by_educ = disagreement_distribution.groupby('annotator_educ')['count'].transform('sum')
disagreement_distribution['percentage'] = (disagreement_distribution['count'] / total_disagreements_by_educ) * 100

labels = sorted(disagreement_distribution['hatespeech'].unique())
educ_list = disagreement_distribution['annotator_educ'].unique()

x = np.arange(len(labels))  # Base positions for hatespeech labels
width = 0.8 / len(educ_list)  # Divide bar width equally among educ

plt.figure(figsize=(10, 6))
for i, educ in enumerate(educ_list):
    subset = disagreement_distribution[disagreement_distribution['annotator_educ'] == educ]
    plt.bar(
        x + i * width - (width * len(educ_list)) / 2,  # Offset bars for each educ
        subset['percentage'],
        width=width,
        label=educ.capitalize()
    )

plt.xlabel('Hatespeech Label')
plt.ylabel('Percentage')
plt.title('Percentage of Hatespeech Labels in Disagreement Cases by educ')
plt.xticks(sorted(disagreement_distribution['hatespeech'].unique()))
plt.legend(title='Annotator educ')
plt.tight_layout()

plot_path = os.path.join(output_dir, 'educ_disagreement_hatespeech_distribution.png')
plt.savefig(plot_path)

print(f"Disagreement bar graph saved to {plot_path}")

# Save Education Values

result_df = educ_agreement[['annotator_educ', 'agreement_count', 'agreement_percentage']]

output_path = os.path.join(output_dir, 'educ_agreement_analysis.csv')
result_df.to_csv(output_path, sep='\t', index=False)

print(f"Analysis results saved to {output_path}")


## Ideology
ideology_agreement = df.groupby(['annotator_ideology']).agg(
    agreement_count=('agrees_with_aggregate', 'sum'),  # Total matches with aggregate label
    total_annotations=('agrees_with_aggregate', 'count')  # Total annotations per gender
).reset_index()

ideology_agreement['agreement_percentage'] = (ideology_agreement['agreement_count'] / ideology_agreement['total_annotations']) * 100

# Create the bar graph
plt.figure(figsize=(10, 6))

cmap = get_cmap("tab10")  # Choose a colormap
colors = [cmap(i) for i in range(len(ideology_agreement))]

bars = plt.bar(
    ideology_agreement['annotator_ideology'],
    ideology_agreement['agreement_percentage'],
    color=colors,
    edgecolor='black'
)

for bar, percentage in zip(bars, ideology_agreement['agreement_percentage']):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        f'{percentage:.1f}%',  # Format as a percentage with 1 decimal
        ha='center',
        va='bottom'
    )

plt.xlabel('Annotator Ideology')
plt.ylabel('Agreement Percentage')
plt.title('Agreement Percentage by Annotator Ideology')
plt.xticks(rotation=45)
plt.tight_layout()

# Save the plot as a PNG file
plot_path = os.path.join(output_dir, 'ideology_agreement_bar_graph.png')
plt.savefig(plot_path)

print(f"Bar graph saved to {plot_path}")

# Disagreement Analysis
disagreement_distribution = disagreement_df.groupby(['annotator_ideology', 'hatespeech']).size().reset_index(name='count')
total_disagreements_by_ideology = disagreement_distribution.groupby('annotator_ideology')['count'].transform('sum')
disagreement_distribution['percentage'] = (disagreement_distribution['count'] / total_disagreements_by_ideology) * 100

labels = sorted(disagreement_distribution['hatespeech'].unique())
ideology_list = disagreement_distribution['annotator_ideology'].unique()

x = np.arange(len(labels))  # Base positions for hatespeech labels
width = 0.8 / len(ideology_list)  # Divide bar width equally among ideology

plt.figure(figsize=(10, 6))
for i, ideology in enumerate(ideology_list):
    subset = disagreement_distribution[disagreement_distribution['annotator_ideology'] == ideology]
    plt.bar(
        x + i * width - (width * len(ideology_list)) / 2,  # Offset bars for each ideology
        subset['percentage'],
        width=width,
        label=ideology.capitalize()
    )

plt.xlabel('Hatespeech Label')
plt.ylabel('Percentage')
plt.title('Percentage of Hatespeech Labels in Disagreement Cases by Ideology')
plt.xticks(sorted(disagreement_distribution['hatespeech'].unique()))
plt.legend(title='Annotator Ideology')
plt.tight_layout()

plot_path = os.path.join(output_dir, 'ideology_disagreement_hatespeech_distribution.png')
plt.savefig(plot_path)

print(f"Disagreement bar graph saved to {plot_path}")

# Save Ideology Values

result_df = ideology_agreement[['annotator_ideology', 'agreement_count', 'agreement_percentage']]

output_path = os.path.join(output_dir, 'ideology_agreement_analysis.csv')
result_df.to_csv(output_path, sep='\t', index=False)

print(f"Analysis results saved to {output_path}")


## Income
income_agreement = df.groupby(['annotator_income']).agg(
    agreement_count=('agrees_with_aggregate', 'sum'),  # Total matches with aggregate label
    total_annotations=('agrees_with_aggregate', 'count')  # Total annotations per gender
).reset_index()

income_agreement['agreement_percentage'] = (income_agreement['agreement_count'] / income_agreement['total_annotations']) * 100

# Create the bar graph
plt.figure(figsize=(10, 6))

cmap = get_cmap("tab10")  # Choose a colormap
colors = [cmap(i) for i in range(len(income_agreement))]

bars = plt.bar(
    income_agreement['annotator_income'],
    income_agreement['agreement_percentage'],
    color=colors,
    edgecolor='black'
)

for bar, percentage in zip(bars, income_agreement['agreement_percentage']):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        f'{percentage:.1f}%',  # Format as a percentage with 1 decimal
        ha='center',
        va='bottom'
    )

plt.xlabel('Annotator income')
plt.ylabel('Agreement Percentage')
plt.title('Agreement Percentage by Annotator income')
plt.xticks(rotation=45)
plt.tight_layout()

# Save the plot as a PNG file
plot_path = os.path.join(output_dir, 'income_agreement_bar_graph.png')
plt.savefig(plot_path)

print(f"Bar graph saved to {plot_path}")

# Disagreement Analysis
disagreement_distribution = disagreement_df.groupby(['annotator_income', 'hatespeech']).size().reset_index(name='count')
total_disagreements_by_income = disagreement_distribution.groupby('annotator_income')['count'].transform('sum')
disagreement_distribution['percentage'] = (disagreement_distribution['count'] / total_disagreements_by_income) * 100

labels = sorted(disagreement_distribution['hatespeech'].unique())
income_list = disagreement_distribution['annotator_income'].unique()

x = np.arange(len(labels))  # Base positions for hatespeech labels
width = 0.8 / len(income_list)  # Divide bar width equally among income

plt.figure(figsize=(10, 6))
for i, income in enumerate(income_list):
    subset = disagreement_distribution[disagreement_distribution['annotator_income'] == income]
    plt.bar(
        x + i * width - (width * len(income_list)) / 2,  # Offset bars for each income
        subset['percentage'],
        width=width,
        label=income.capitalize()
    )

plt.xlabel('Hatespeech Label')
plt.ylabel('Percentage')
plt.title('Percentage of Hatespeech Labels in Disagreement Cases by income')
plt.xticks(sorted(disagreement_distribution['hatespeech'].unique()))
plt.legend(title='Annotator income')
plt.tight_layout()

plot_path = os.path.join(output_dir, 'income_disagreement_hatespeech_distribution.png')
plt.savefig(plot_path)

print(f"Disagreement bar graph saved to {plot_path}")

# Save income Values

result_df = income_agreement[['annotator_income', 'agreement_count', 'agreement_percentage']]

output_path = os.path.join(output_dir, 'income_agreement_analysis.csv')
result_df.to_csv(output_path, sep='\t', index=False)

print(f"Analysis results saved to {output_path}")
