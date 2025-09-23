# Used Car Price Prediction Analysis
### INF601 - Advanced Programming in Python
### Samuel Amoateng
### Mini Project 2

## Project Title
Used Car Price Prediction Analysis

## Description

This project analyzes a used car dataset to identify factors that most significantly impact used car prices. The analysis includes comprehensive data cleaning, exploratory data analysis, correlation studies, and data visualization to understand the relationships between various car features and their sale prices.

**Key Research Question:** "What factors most significantly impact used car prices?"

The project demonstrates proficiency in:
- Data manipulation and cleaning using Pandas
- Statistical analysis and correlation studies
- Data visualization using Matplotlib and Seaborn
- Handling missing values and categorical data encoding
- Creating meaningful insights from real-world datasets

## Getting Started

### Dependencies

* Python 3.7+
* Windows 10/11 or compatible OS
* Required Python packages:
  - pandas - Data manipulation and analysis
  - numpy - Numerical computing
  - matplotlib - Data visualization
  - scikit-learn - Machine learning algorithms
  - seaborn - Statistical data visualization

### Installing

1. Clone or download the project files
2. Install required packages:
```bash
pip install -r requirements.txt
```
3. Ensure the dataset file `Used_Car_Price_Prediction.csv` is in the project root directory

### Executing program

#### Option 1: Run the Python script
```bash
python main.py
```
This will load the dataset and display the first few rows.

#### Option 2: Run the Jupyter Notebook (Recommended)
```bash
jupyter notebook
```
Then open `Used_Car_Price_Prediction.ipynb` to see the complete analysis including:
- Data cleaning and null value handling
- Correlation analysis (numerical and ordinal features)
- Data visualization and insights
- Statistical summaries

#### Generated Visualizations
The analysis automatically generates the following charts in the `charts/` directory:
- Correlation heatmaps (numerical and ordinal features)
- Box plots for key features
- Scatter plots for price relationships
- Bar plots for correlation rankings

## Project Structure

```
miniproject2SamuelAmoateng/
├── main.py                          # Basic data loading script
├── Used_Car_Price_Prediction.ipynb  # Comprehensive analysis notebook
├── Used_Car_Price_Prediction.csv    # Dataset
├── requirements.txt                 # Python dependencies
├── README.md                        # This file
├── .gitignore                       # Excludes charts/ directory
└── charts/                          # Generated visualizations (auto-created)
    ├── numerical_corr_barplot.png
    ├── ordinal_corr_barplot.png
    ├── boxplot_*.png
    └── scatter_*.png
```

## Data Analysis Features

### Data Cleaning
- Handles missing values in categorical and numerical features
- Uses appropriate imputation methods (mode for categorical, median for numerical)
- Converts data types as needed for analysis

### Correlation Analysis
- **Numerical Features:** Pearson correlation with sale price
- **Ordinal Features:** Spearman correlation with sale price
- Identifies top factors influencing car prices

### Visualizations
- Heatmaps showing feature correlations
- Box plots for categorical feature distributions
- Scatter plots for price relationships
- Bar plots ranking feature importance

## Help

### Common Issues
- **Missing dataset:** Ensure `Used_Car_Price_Prediction.csv` is in the project root
- **Missing packages:** Run `pip install -r requirements.txt`
- **Charts not generating:** Check that matplotlib and seaborn are properly installed

### Getting Dataset
The dataset used in this project can be obtained from various sources including:
- Kaggle datasets repository
- Data.gov
- Other public data sources

## Authors

Samuel Amoateng
- Course: INF601 - Advanced Programming in Python
- Project: Mini Project 2

## Version History

* 1.0
    * Initial release with complete analysis
    * Data cleaning and correlation analysis
    * Multiple visualization types
    * Comprehensive documentation

## License

This project is licensed under the MIT License - see the LICENSE.md file for details

## Acknowledgments

* Course materials and guidance from INF601 - Advanced Programming in Python
* Dataset sources and contributors
* Python data science community for tools and libraries
* Inspiration from various data analysis tutorials and examples
