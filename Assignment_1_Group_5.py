import pandas as pd
import requests
import zipfile
import io
import os
import subprocess
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from datetime import datetime

# Get the current directory and create the path
current_dir = os.path.dirname(os.path.abspath(__file__))

if not os.path.exists(current_dir + "/Assignment_1_Group_5"):
    os.makedirs(current_dir + "/Assignment_1_Group_5")
subfolder_dir = "/Assignment_1_Group_5/"
dashboard_path = os.path.join(current_dir, "learning_analytics_dashboard.py")

# PART 1: Obtaining a Transactional Dataset
# URL to the ZIP file
zip_url = "https://archive.ics.uci.edu/static/public/349/open+university+learning+analytics+dataset.zip"

# PART 2: Data Preprocessing
# Step 1: Download the ZIP file
response = requests.get(zip_url)
if response.status_code == 200:
    print("✓ ZIP file downloaded successfully.")

    # Step 2: Extract ZIP content from memory
    with zipfile.ZipFile(io.BytesIO(response.content)) as z:
        z.extractall(current_dir + "/oulad_data")  # Extracts into folder "oulad_data"
        print("✓ Files extracted to 'oulad_data/'")
else:
    print(f"✗ Failed to download file. Status code: {response.status_code}")

# Load the datasets
vle = pd.read_csv(os.path.join(current_dir, "oulad_data", "vle.csv"))  # Maps activity IDs to resource types
student_vle = pd.read_csv(os.path.join(current_dir, "oulad_data", "studentVle.csv"))  # Logs of what students interacted with

# Merge to get meaningful content types
merged = pd.merge(student_vle, vle, on='id_site', how='left')

# Create one-hot encoded DataFrame
# First, get unique students and activity types
transactions_df = pd.crosstab(
    merged['id_student'], 
    merged['activity_type']
)

# Convert to boolean (0/1) values
transactions_df = transactions_df.astype(bool)

# Save the activity types for later reference
activity_types = list(transactions_df.columns)

# PART 3: Apriori Algorithm Implementation
print("\nImplementing Apriori Algorithm...")

# Step 1: Find frequent itemsets
# Using min_support = 0.1 (10% of transactions)
frequent_itemsets = apriori(transactions_df, min_support=0.1, use_colnames=True)
print("\nFrequent Itemsets:")
print(frequent_itemsets)

# Step 2: Generate association rules
# Using min_threshold = 0.7 for confidence
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.7)
print("\nAssociation Rules:")
print(rules)

# Step 3: Sort rules by confidence and lift
rules = rules.sort_values(['confidence', 'lift'], ascending=[False, False])
print("\nTop 10 Rules by Confidence and Lift:")
print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head(10))

# Step 4: Filter rules with lift > 1
strong_rules = rules[rules['lift'] > 1]
print("\nStrong Rules (lift > 1):")
print(strong_rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])

# Step 5: Save results to CSV
frequent_itemsets.to_csv(current_dir + subfolder_dir + 'frequent_itemsets.csv', index=False)
rules.to_csv(current_dir + subfolder_dir + 'association_rules.csv', index=False)
strong_rules.to_csv(current_dir + subfolder_dir + 'strong_rules.csv', index=False)
print("\nResults saved to CSV files:")
print("- frequent_itemsets.csv")
print("- association_rules.csv")
print("- strong_rules.csv")

# PART 4: Analysis and Interpretation
print("\nAnalyzing Results...")

# 1. Basic Statistics
print("\n1. Basic Statistics:")
print(f"Total number of rules: {len(rules)}")
print(f"Number of strong rules (lift > 1): {len(strong_rules)}")
print(f"Average confidence: {rules['confidence'].mean():.2f}")
print(f"Average lift: {rules['lift'].mean():.2f}")

# 2. Top Activity Types by Support
activity_support = transactions_df.mean().sort_values(ascending=False)
print("\n2. Top 5 Most Common Activity Types:")
print(activity_support.head())

# 3. Visualize Rule Metrics
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.histplot(rules['confidence'], bins=20)
plt.title('Distribution of Confidence')
plt.subplot(1, 2, 2)
sns.histplot(rules['lift'], bins=20)
plt.title('Distribution of Lift')
plt.tight_layout()
plt.savefig(current_dir + subfolder_dir + 'rule_metrics.png')
print("\nRule metrics visualization saved as 'rule_metrics.png'")

# 4. Analyze Strong Rules
print("\n4. Analysis of Strong Rules:")
# Group by consequents to see what activities are commonly predicted
consequent_analysis = strong_rules.groupby('consequents').agg({
    'confidence': 'mean',
    'lift': 'mean',
    'support': 'mean'
}).sort_values('confidence', ascending=False)

print("\nMost Predictable Activities (by confidence):")
print(consequent_analysis.head())

# 5. Generate Recommendations
print("\n5. Activity Recommendations:")
# Create a function to get recommendations based on antecedents
def get_recommendations(activity):
    relevant_rules = strong_rules[
        strong_rules['antecedents'].apply(lambda x: activity in x)
    ].sort_values('lift', ascending=False)
    
    if len(relevant_rules) > 0:
        print(f"\nIf a student uses {activity}, recommend:")
        for _, rule in relevant_rules.head(3).iterrows():
            consequent = list(rule['consequents'])[0]
            print(f"- {consequent} (confidence: {rule['confidence']:.2f}, lift: {rule['lift']:.2f})")

# Get recommendations for top 3 most common activities
for activity in activity_support.head(3).index:
    get_recommendations(activity)

# 6. Business Implications
print("\n6. Business Implications:")
print("Based on the analysis, here are potential recommendations:")
print("1. Course Design:")
print("   - Bundle activities that frequently co-occur in course design")
print("   - Create learning paths that follow natural student behavior patterns")
print("\n2. Student Support:")
print("   - Use activity patterns to identify students who might need additional support")
print("   - Recommend complementary activities based on student engagement patterns")
print("\n3. Resource Allocation:")
print("   - Focus resources on developing content for activities with high support and lift")
print("   - Identify and address gaps in activity sequences that students commonly follow")

# PART 5: Comprehensive Summary and Visualizations
print("\nGenerating Comprehensive Summary...")

# 1. Create Network Graph of Strong Rules
plt.figure(figsize=(15, 10))
G = nx.DiGraph()

# Add nodes and edges
for _, rule in strong_rules.iterrows():
    antecedent = list(rule['antecedents'])[0]
    consequent = list(rule['consequents'])[0]
    G.add_edge(antecedent, consequent, weight=rule['lift'])

# Draw the network
pos = nx.spring_layout(G, k=1, iterations=50)
nx.draw(G, pos, with_labels=True, node_color='lightblue', 
        node_size=2000, font_size=10, font_weight='bold',
        edge_color='gray', width=[G[u][v]['weight'] for u, v in G.edges()])
plt.title('Activity Association Network (Edge Width = Lift)')
plt.savefig(current_dir + subfolder_dir + 'activity_network.png')
print("\nNetwork graph saved as 'activity_network.png'")

# 2. Create Bar Plot of Top Rules
plt.figure(figsize=(12, 6))
top_rules = strong_rules.head(10)
rules_labels = [f"{list(r['antecedents'])[0]} → {list(r['consequents'])[0]}" 
                for _, r in top_rules.iterrows()]
plt.barh(rules_labels, top_rules['lift'])
plt.xlabel('Lift')
plt.title('Top 10 Rules by Lift')
plt.tight_layout()
plt.savefig(current_dir + subfolder_dir + 'top_rules_lift.png')
print("\nTop rules visualization saved as 'top_rules_lift.png'")

# 3. Generate Summary Report
with open(current_dir + subfolder_dir + 'analysis_summary.txt', 'w', encoding='utf-8') as f:
    f.write("Market Basket Analysis of Learning Activities\n")
    f.write("===========================================\n\n")
    
    f.write("1. Dataset Overview\n")
    f.write("------------------\n")
    f.write(f"Total students: {len(transactions_df)}\n")
    f.write(f"Total activity types: {len(activity_types)}\n")
    f.write(f"Date of analysis: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    
    f.write("2. Preprocessing Steps\n")
    f.write("----------------------\n")
    f.write("- Merged student activity logs with activity type information\n")
    f.write("- Created one-hot encoded matrix of student-activity interactions\n")
    f.write("- Filtered for activities with minimum support of 10%\n")
    f.write("- Generated rules with minimum confidence of 70%\n\n")
    
    f.write("3. Top 10 Association Rules\n")
    f.write("--------------------------\n")
    f.write("Antecedent -> Consequent | Support | Confidence | Lift\n")
    f.write("-" * 80 + "\n")
    for _, rule in top_rules.iterrows():
        antecedent = list(rule['antecedents'])[0]
        consequent = list(rule['consequents'])[0]
        f.write(f"{antecedent} -> {consequent} | {rule['support']:.3f} | {rule['confidence']:.3f} | {rule['lift']:.3f}\n")
    f.write("\n")
    
    f.write("4. Key Findings\n")
    f.write("--------------\n")
    f.write("a) Most Common Activity Patterns:\n")
    for activity in activity_support.head(5).index:
        f.write(f"- {activity}: {activity_support[activity]:.2%} of students\n")
    f.write("\n")
    
    f.write("b) Strongest Associations:\n")
    for _, rule in top_rules.head(5).iterrows():
        antecedent = list(rule['antecedents'])[0]
        consequent = list(rule['consequents'])[0]
        f.write(f"- {antecedent} -> {consequent} (lift: {rule['lift']:.2f})\n")
    f.write("\n")
    
    f.write("5. Recommendations\n")
    f.write("-----------------\n")
    f.write("a) Course Design:\n")
    f.write("- Bundle frequently co-occurring activities in course modules\n")
    f.write("- Create learning paths that follow natural student behavior patterns\n")
    f.write("- Design complementary activities that support each other\n\n")
    
    f.write("b) Student Support:\n")
    f.write("- Use activity patterns to identify at-risk students\n")
    f.write("- Implement personalized activity recommendations\n")
    f.write("- Develop intervention strategies based on activity sequences\n\n")
    
    f.write("c) Resource Allocation:\n")
    f.write("- Focus development efforts on high-impact activities\n")
    f.write("- Optimize content delivery based on activity patterns\n")
    f.write("- Allocate support resources to critical activity sequences\n")

print("\nComprehensive analysis summary saved as 'analysis_summary.txt'")
print("Visualizations saved as 'activity_network.png' and 'top_rules_lift.png'")


# Launch the Streamlit app
subprocess.run(["streamlit", "run", dashboard_path])
