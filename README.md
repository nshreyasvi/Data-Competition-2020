# ml_2020
## Updates
**-Added report.Rmd file test_code/report.Rmd**


-Finished all analysis (check visual.R code for all analysis stuff)
-Please find best model and do predictions (Try to submit one prediction using elasticnet ---------> let me know before you send it so i verify it, we can't loose tries at this point)
- Detailed stuff is available at test_code/test.R (including pruning trees, other functions and shit) <---------------Use this for adding things, I made code for all things done in class

-Big problem of 8000 missing values-> resolved by implementing 3 different algor(not taking binary into account or scaling them?? IDK) (take all variables just convert them into other formats)

- **Changing and merging the columns which are similar i.e. in the outcome_old field, I changed its 4 levels (success, failure, na, other) into 2 factor levels (success and failure, replacing the failure and other values into all NaN (increased accuracy from 86.3% to 86.63%) and in marital field, I changed single, divorced and married I merged the single and divorced based on the fact that they are both sort of single) ---> increased accuracy 86.63% to 86.82% in training**

- Trying to check if the days_elapsed_old can be changed to categorical for better results (same might be possible for other things for better fitting)

- **Please try to find a possible way to reduce the job field which will be the step that could help us achieve more than 88% accuracy.**

- Added rf.R file which is random forest getting around 86.82 percent accuracy. However prediction accuracy is not known, this is just a filtered version of the train.csv file used for training.
(Could you do the analysis on the updated r markdown file which is uploaded here, I removed some errors which existed previously along with some data cleaning, let me know if you have any questions.)

## To Do
- Adding RMarkDown for first report (draft). **Please check for pre-processing the csv file before doing analysis**
- Try to show the data by groups/ellipse in the PCA analysis
- List of variables as well as their transforms (square root, square etc) that give better results

**We need more than 86.63 percent on the prediction set, please check if there is a way to compress the data into a small and meaningfull set**
