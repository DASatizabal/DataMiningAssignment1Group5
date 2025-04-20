# Learning Analytics Dashboard

This project consists of two main components:
1. A data mining script (`Assignment_1_Group_5.py`) that analyzes student learning activities
2. A Streamlit dashboard (`learning_analytics_dashboard.py`) that visualizes the results

## Project Structure

```
project_root/
├── Assignment_1_Group_5.py      # Main data mining script
├── learning_analytics_dashboard.py  # Streamlit dashboard
├── oulad_data/                  # Downloaded dataset
│   ├── vle.csv
│   └── studentVle.csv
└── Assignment_1_Group_5/        # Output directory
    ├── analysis_summary.txt     # Text summary of analysis
    ├── activity_network.png     # Network visualization
    ├── top_rules_lift.png      # Top rules visualization
    ├── rule_metrics.png        # Rule metrics visualization
    ├── frequent_itemsets.csv   # Frequent itemsets data
    ├── association_rules.csv   # All association rules
    └── strong_rules.csv        # Strong rules (lift > 1)
```

## Output Files

All output files are saved in the `Assignment_1_Group_5/` subdirectory:

1. **CSV Files**:
   - `frequent_itemsets.csv`: Contains all frequent itemsets found by the Apriori algorithm
   - `association_rules.csv`: Contains all generated association rules
   - `strong_rules.csv`: Contains only rules with lift > 1

2. **Visualizations**:
   - `activity_network.png`: Network graph showing relationships between activities
   - `top_rules_lift.png`: Bar chart of top 10 rules by lift
   - `rule_metrics.png`: Distribution plots of confidence and lift metrics

3. **Analysis Summary**:
   - `analysis_summary.txt`: Comprehensive text report of the analysis

## How to Use

### 1. Running the Data Mining Script

Run the main script to perform the analysis and generate all output files:
```bash
python Assignment_1_Group_5.py
```

This will:
- Download and process the dataset
- Generate association rules
- Create visualizations
- Save all output files to the `Assignment_1_Group_5/` directory
- Launch the Streamlit dashboard automatically

### 2. Using the Streamlit Dashboard

The dashboard can be accessed in two ways:

1. **Automatically**: The dashboard will launch automatically after running the data mining script
2. **Manually**: Run the following command:
   ```bash
   streamlit run learning_analytics_dashboard.py
   ```

The dashboard provides:
- Interactive network visualization of activity associations
- Table of top association rules
- Activity frequency analysis
- Distribution plots of rule metrics
- Recommendations based on the analysis

### Dashboard Features

1. **Activity Association Network**:
   - Interactive network graph
   - Hover over edges to see lift values
   - Nodes represent different activities
   - Edge opacity indicates relationship strength

2. **Top Rules Table**:
   - Shows the 10 strongest associations
   - Displays support, confidence, and lift metrics
   - Sortable and filterable

3. **Activity Insights**:
   - Bar chart of most common activities
   - Distribution plots of confidence and lift
   - Interactive filtering options

4. **Recommendations**:
   - Course design suggestions
   - Student support strategies
   - Resource allocation guidance

## Requirements

- Python 3.x
- Required packages:
  - pandas
  - requests
  - mlxtend
  - matplotlib
  - seaborn
  - networkx
  - streamlit
  - plotly

Install requirements:
```bash
pip install pandas requests mlxtend matplotlib seaborn networkx streamlit plotly
``` 