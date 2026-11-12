# Data Analytics MVP

A Streamlit app for exploring CSV and Excel data and comparing regression models. Upload a table to inspect it, summarize results by category, and try what-if predictions for a numeric target.

## What to try

- **Explore** previews the table, summarizes its shape, and plots numeric correlations. You can drop rows with missing values and download the resulting CSV.
- **Aggregate** groups a numeric measure by a text category, shows the top 15 groups, and exports the summary. Choose sum, average, count, maximum, or minimum.
- **Model** compares linear regression, a decision tree, and a random forest using R², RMSE, and MAE. Adjust feature sliders to see how the model's prediction changes.

## Run locally

From the repository root, install the dependencies and start the app.

```sh
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Server-side PNG downloads use Kaleido with Chrome or Chromium. If the server cannot export images, use the download action in the interactive chart toolbar. [Plotly image export setup](https://plotly.com/python/static-image-export/)

Save this example as `sample.csv`, upload it, and select `Sales` as the target with `AdSpend` and `Visits` as features.

```csv
Region,AdSpend,Visits,Sales
North,90,125,14
South,120,180,19
East,140,200,23
North,105,156,17
South,155,220,28
East,130,208,24
North,165,240,31
South,180,255,34
East,145,217,27
North,200,290,39
South,170,249,33
East,220,310,43
```

## Scope

The app expects at least two numeric columns. The grouping tab also needs a text category. The example CSV is synthetic. Model comparison needs at least six complete, finite rows for the selected columns. Scores use a single random 80/20 split, so small samples can produce unstable scores.
