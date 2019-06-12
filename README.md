# Dota2-Game-Predictor
This project uses machine learning (Logistic regression algorithm) to calcuate the winning chance for Dota2 matches. The workflow is as follows:

1. The `matchfiner.py` must be used first to retrieve all of the matches for the player. At this stage, player's Dota2 account ID is hard-coded in this script. It uses [Open Dota API](https://www.opendota.com/). As we are using the free version of the Open Dota, due to the API call limit, this script may take some hours to retrieve all of the matches. The final result will be a large JSON text file named `matches.txt`.

2. The `featurize.py` then should run to open the  `matches.txt` and extract the required features of the matches. For this project, the following features of matches are used:
    * Radiant score - Dire score
    * Radiant average total gold - Dire average total gold
    * Radiant average total XP - Dire average total XP
    
    The positive value of the above features indicates the Radiant's team dominance. The result of this code is the `dataset.txt` which has four columns, three of them are the aforementioned features and the last one is the result of each match. __1__ indicates Radiant win and __0__ means Dire win.
3. Then for the last step, we have run `logisticReg.m` which is an Octave code to run our Logistic Regression algorithm and find the best fit for the provided dataset. At this time, the outputs of this code are:
    * A plot that shows the learning curve of the algorithm.
    * A plot that illustrates the dataset using x for Radiant win case and o for Dire win matches. Because there are three features, in order to plot a 2-D diagram, the score difference was used as the x-axis and average of gold and XP difference was used for the y-axis. Also, this plot includes a simple contour for seperating the win/lose case based on the parameter theta found by the logistic regression.
    * The accuracy of the algorithm for training data and test data.

