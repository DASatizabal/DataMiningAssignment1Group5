import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import networkx as nx
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
import matplotlib.pyplot as plt
import io
import subprocess
import os
# Get the current directory and create the path
current_dir = os.path.dirname(os.path.abspath(__file__))
subfolder_dir = "/Assignment_1_Group_5/"

# Set page config
st.set_page_config(
    page_title="Learning Analytics Dashboard",
    page_icon="📚",
    layout="wide"
)

# Title and description
st.title("📚 Learning Activity Association Explorer")
st.markdown("""
This dashboard helps you explore patterns in student learning activities. 
Discover which activities students tend to use together and get insights for course design.
""")

# Sidebar for controls
st.sidebar.header("Analysis Parameters")

# Load data
@st.cache_data
def load_data():
    # Load the datasets
    vle = pd.read_csv(current_dir + '/oulad_data/vle.csv')
    student_vle = pd.read_csv(current_dir + '/oulad_data/studentVle.csv')
    
    # Merge and preprocess
    merged = pd.merge(student_vle, vle, on='id_site', how='left')
    transactions_df = pd.crosstab(merged['id_student'], merged['activity_type'])
    transactions_df = transactions_df.astype(bool)
    return transactions_df

# Get user input for parameters
min_support = st.sidebar.slider(
    "Minimum Support (%)",
    min_value=1,
    max_value=50,
    value=10,
    help="Minimum percentage of transactions that must contain an itemset"
) / 100

min_confidence = st.sidebar.slider(
    "Minimum Confidence (%)",
    min_value=1,
    max_value=100,
    value=70,
    help="Minimum confidence for association rules"
) / 100

# Load and process data
transactions_df = load_data()

# Generate rules
@st.cache_data
def generate_rules(min_support, min_confidence):
    frequent_itemsets = apriori(transactions_df, min_support=min_support, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
    return rules

rules = generate_rules(min_support, min_confidence)

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Activity Association Network")
    
    # Create network graph
    G = nx.DiGraph()
    for _, rule in rules.iterrows():
        antecedent = list(rule['antecedents'])[0]
        consequent = list(rule['consequents'])[0]
        G.add_edge(antecedent, consequent, weight=rule['lift'])
    
    # Create Plotly figure
    pos = nx.spring_layout(G, k=1, iterations=50)
    
    # Create edge traces
    edge_traces = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        weight = G[edge[0]][edge[1]]['weight']
        
        # Normalize weight for color (between 0 and 1)
        normalized_weight = (weight - min(rules['lift'])) / (max(rules['lift']) - min(rules['lift']))
        
        edge_trace = go.Scatter(
            x=[x0, x1, None],
            y=[y0, y1, None],
            line=dict(width=2, color=f'rgba(136, 136, 136, {0.2 + normalized_weight * 0.8})'),
            hoverinfo='text',
            text=f"Lift: {weight:.2f}",
            mode='lines'
        )
        edge_traces.append(edge_trace)
    
    # Create node trace
    node_trace = go.Scatter(
        x=[],
        y=[],
        mode='markers+text',
        hoverinfo='text',
        text=[],
        textposition="bottom center",
        marker=dict(
            color='lightblue',
            size=20,
            line=dict(color='black', width=2)
        ))
    
    for node in G.nodes():
        x, y = pos[node]
        node_trace['x'] += tuple([x])
        node_trace['y'] += tuple([y])
        node_trace['text'] += tuple([node])
    
    # Create figure
    fig = go.Figure(data=edge_traces + [node_trace],
                    layout=go.Layout(
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20, l=5, r=5, t=40),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)))
    
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Top Rules")
    
    # Display top rules in a table
    top_rules = rules.sort_values('lift', ascending=False).head(10)
    top_rules_display = top_rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].copy()
    top_rules_display['Rule'] = top_rules_display.apply(
        lambda x: f"{list(x['antecedents'])[0]} → {list(x['consequents'])[0]}", axis=1)
    top_rules_display = top_rules_display[['Rule', 'support', 'confidence', 'lift']]
    
    st.dataframe(
        top_rules_display.style.format({
            'support': '{:.2%}',
            'confidence': '{:.2%}',
            'lift': '{:.2f}'
        }),
        use_container_width=True
    )

# Additional insights
st.subheader("Activity Insights")

# Activity frequency
activity_freq = transactions_df.mean().sort_values(ascending=False)
fig = px.bar(
    x=activity_freq.index[:10],
    y=activity_freq.values[:10],
    title="Top 10 Most Common Activities",
    labels={'x': 'Activity Type', 'y': 'Percentage of Students'},
    color=activity_freq.values[:10]
)
st.plotly_chart(fig, use_container_width=True)

# Rule metrics distribution
col1, col2 = st.columns(2)
with col1:
    fig = px.histogram(rules, x='confidence', title='Confidence Distribution')
    st.plotly_chart(fig, use_container_width=True)

with col2:
    fig = px.histogram(rules, x='lift', title='Lift Distribution')
    st.plotly_chart(fig, use_container_width=True)

# Recommendations section
st.subheader("Recommendations")
st.markdown("""
Based on the current analysis:

1. **Course Design**
   - Consider bundling activities that frequently co-occur
   - Design learning paths that follow natural student behavior patterns
   - Create complementary activity pairs

2. **Student Support**
   - Use activity patterns to identify students who might need additional support
   - Implement personalized activity recommendations
   - Develop intervention strategies based on activity sequences

3. **Resource Allocation**
   - Focus development efforts on high-impact activities
   - Optimize content delivery based on activity patterns
   - Allocate support resources to critical activity sequences
""")

# Add download button for rules
@st.cache_data
def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

csv = convert_df_to_csv(rules)
st.download_button(
    label="Download Association Rules",
    data=csv,
    file_name='association_rules.csv',
    mime='text/csv',
) 
