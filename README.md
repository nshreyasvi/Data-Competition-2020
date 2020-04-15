# ml_2020
## Updates
- Bugs with the dataset
    - Removed lines which convert numeric to factor and then again to numeric (e.g. dataset$age=as.numeric(as.factor(dataset$age)))
    - Removed na strings to NaN (NA) and then to 0
    - Changed days_elapsed_old (-1 to 0)

- Added EDA+PCA  -> elastic net in progress (variable selection)

## To Do
- Adding RMarkDown for first report (draft).
- Try to show the data by groups/ellipse in the PCA analysis
- ELASTIC NET
- List of variables as well as their transforms (square root, square etc) that give better results