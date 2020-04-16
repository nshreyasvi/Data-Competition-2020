# ml_2020
## Updates
- Bugs with the dataset
    - Removed lines which convert numeric to factor and then again to numeric (e.g. dataset$age=as.numeric(as.factor(dataset$age)))
    - Removed na strings to NaN (NA) and then to 0
    - Changed days_elapsed_old (-1 to 0)
    - Merged 'failure' and 'other' field and converted all of them to NaN
    - Merged 'divorced' into 'single' and made it into 2 columns (single/divorced and married)

- **Changing and merging the columns which are similar i.e. in the outcome_old field, I changed its 4 levels (success, failure, na, other) into 2 factor levels (success and failure, replacing the failure and other values into all NaN (increased accuracy from 86.3% to 86.63%) and in marital field, I changed single, divorced and married I merged the single and divorced based on the fact that they are both sort of single) ---> increased accuracy 86.63% to 86.82% in training**

- **Please try to find a possible way to reduce the job field which will be the step that could help us achieve more than 88% accuracy.**

- Added rf.R file which is random forest getting around 86.82 percent accuracy. However prediction accuracy is not known, this is just a filtered version of the train.csv file used for training.
(Could you do the analysis on the updated r markdown file which is uploaded here, I removed some errors which existed previously along with some data cleaning, let me know if you have any questions.)
- Added EDA+PCA  -> elastic net in progress (variable selection)

## To Do
- Adding RMarkDown for first report (draft). **Please check for pre-processing the csv file before doing analysis**
- Try to show the data by groups/ellipse in the PCA analysis
- ELASTIC NET
- List of variables as well as their transforms (square root, square etc) that give better results

**We need more than 86.63 percent on the prediction set, please check if there is a way to compress the data into a small and meaningfull set**