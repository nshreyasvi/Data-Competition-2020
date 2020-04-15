# ml_2020
## Updates
- Bugs with the dataset
    - Removed lines which convert numeric to factor and then again to numeric (e.g. dataset$age=as.numeric(as.factor(dataset$age)))
    - Removed na strings to NaN (NA) and then to 0
    - Changed days_elapsed_old (-1 to 0)
- Added rf.R file which is random forest getting around 86.63 percent accuracy. However prediction accuracy is not known, this is just a filtered version of the train.csv file used for training.
(Could you do the analysis on the updated r markdown file which is uploaded here, I removed some errors which existed previously along with some data cleaning, let me know if you have any questions.)
- Added EDA+PCA  -> elastic net in progress (variable selection)

## To Do
- Adding RMarkDown for first report (draft).
- Try to show the data by groups/ellipse in the PCA analysis
- ELASTIC NET
- List of variables as well as their transforms (square root, square etc) that give better results