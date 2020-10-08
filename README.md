# Optimizing business decisions from online purchasing behavior

### Introduction

Analytics data such as web pages visited, types of web page visited (e.g. related products) exit rates, and bounce rates of users for e-commerce websites provides a picture of the factors surrounding purchase decisions and the chance to inform business decisions such as inventory stocking in order to maximize profits. The <a href="https://archive.ics.uci.edu/ml/datasets/Online+Shoppers+Purchasing+Intention+Dataset">browsing session data of 12,330 sessions for an E-commerce website based in Turkey</a> hosted in a public dataset at the <a href="https://archive.ics.uci.edu/ml/index.php">UCI Machine Learning Repository</a> allows the opportunity to explore online purchasing behavior in terms of these factors. What drives online purchasing behavior and how can that be used to inform business decisions, such as optimal inventory stocking? 

This project has the following goals:

1. Find a model that leads to the most profitable predictions
2. Determine feature importance, i.e. what drives the purchasing behavior?

### The Data

Data source citation: Sakar, C.O., Polat, S.O., Katircioglu, M. et al. Neural Comput & Applic (2018)

Description:

- 12,330 rows (browsing sessions) 
  - 18 features (10 numerical; 8 categorical)
  - Related to web page type (Informational, Product related, etc) and time duration on that page type
  - Related to time (Month, weekend, closeness to holidays, etc.)
  - Analytics (Exit rates, bounce rates, pages visited)
  - Categorical features like browser type, operating system, region, etc.
- Target variable: ‘Revenue’, whether a purchase was made (0 or 1)
  - Class ratio is 84:16 (majority is no purchase)
  
### Modelling: initial considerations

How much money is earned per sale, and what are the costs associated with false positive and false negative predictions? Consider:

**True positive**: Browsing session correctly predicted to result in sale. Money spent to acquire product; revenue gained from sale without additional cost.<br>
**False positive**: Browsing session incorrectly predicted to result in sale. Money spent to acquire product; but because the sale is not immediately made, the product must be stored, incurring carrying cost prior to a possible sale later<br>
**True negative**: Browsing session correctly predicted not to result in sale. No money spent to acquire product; no revenue.<br>
**False negative**: Browsing session incorrectly predicted not to result in sale. Profit is lost, because we don't have the product to make the sale when the customer would've made a purchase. <br>

What about the false positive? If a sale is incorrectly predicted to be made, now we have inventory that must be stored, incurring cost to carry the inventory prior to an eventual sale or (worse) a disposal of the product. Let's assume the product eventually sells. How long is it stored before the sale? If stored long enough, the "profit" could turn out to be a net loss!

Using a 10% profit margin, a cost/benefit matrix could be assembled in terms of revenue R and carrying cost C per sale:

| |Actual positive | Actual negative|  
| --- | --- | --- | 
|Predicted positive |0.1R | 0.1R - C| 
| Predicted negative|-0.1R |0 | 

For simplicity, we could just use R = $1 (that is, every sale earns a buck of revenue). There are other possible costs associated with some of the cost/benefit values, such as a customer going to other businesses for future purchases if they couldn't buy a product they intended to buy from us. However, for simplicity's sake, we'll stick with the above values for this example. Also, to reiterate, we're assuming that products bought that were not sold immediately will sell eventually; there's just a question of how long and how much the carrying cost will dig into the profit. 

To explore possible profits as well as establish a baseline model, we'll split the data into a 80:20 train/test split, train an untuned Random Forest Classifier on the data, and compute profit curves at the range of decision thresholds for different costs or benefits for false positives and see what the maximum possible profits are. We'll use -0.05 (where C renders a negative profit), 0 (break-even), and 0.05 (small enough C to leave a positive profit):

<table>
  <tr>
    <td><img src="img/fp_curve_-0.05.svg"></td><td><img src="img/fp_curve_0.0.svg"></td><td><img src="img/fp_curve_0.05.svg"></td>
  </tr>
  </table>
  

From here we could just see what the maximum profits and associated decision thresholds are from the whole range of -0.1 to 0.1:

  <p align="center">
  <img src="img/prof_range.svg" width=600><br>
  </p>


The "average profit margin per session" is the total profit divided by the number of browsing sessions.

It turns out that from about FP cost/benefit = 0 onwards, max profit occurs at the threshold of nearly zero, meaning that the best strategy here in terms of maximizing profit is to invest in having a product on hand for every browsing session regardless of whether a purchase is made. For a FP cost of -0.1, the max profit (grimly) becomes zero. The range -0.1 to 0 is where the number of products to invest in is some fraction of the number of browsing sessions, and this is where a machine learning model could be tuned to predict the right amount of product to stock. 

### Modelling the FP cost = -0.05 case: model evaluation

We could choose FP = -0.05 as a basis of modelling because it falls in the middle of the range above between where the Random Forest Classifier predicts that no profit is possible and where the max profit is associated with a sale for all sessions (FP = -0.1 to 0), but in principle any value could be chosen, e.g. based on historical averages. 

We'll use the same 80:20 split from before and use the training fold to find a model that might lead to more profitable predictions than the baseline above. We'd like to try and get a better average profit per session than 0.00410 on the test set. Because false negatives are more expensive than false positives in the scenario we're working under, we might consider looking for a model that maximizes recall. I evaluated XGBoost models with these possible parameters:

| Parameter | Value | 
| --- | --- |
| max depth | 2, 5, 7, 9  |
| min samples split | 1, 2, 3, 4 |
| n estimators | 25, 50, 100, 200, 500 |
| learning rate | 0.05, 0.1, 0.2, 0.3, 0.4 |
| SMOTE oversampling | True, False |
| No. of features | 10-18 | 

The features were a randomly chosen set based on the number used for number of features, and I used 3x5-fold cross-validation (that is, scores are the average from 15 random 80:20 train/validation splits). 

I evaluated models with random combinations of the above parameters and scored them on MCC, precision, and recall and came up with the following results:

<table>
  <tr>
    <td><img src="img/eval1.svg"></td><td><img src="img/eval2.svg"></td>
  </tr>
  </table>

It's interesting how the results show two pretty distinct clusters for each metric. There may be something specific that has a marked effect on model performance, to be investigated in a below section.

Besides the conventional classification metrics, why not also just find the maximum profit of each model? I computed the average maximum possible profit from each model and stored the corresponding average decision threshold. The distribution of profits is also shown above, also with the two distinct clusters of models.

### Modelling the FP cost = -0.05 case: performance on test set

Next, the models were trained on the whole train set and evaluated on the test set. The following shows results using the model performing best on each metric (MCC, precision, recall) in the evaluation step vs. the baseline default Random Forest Classifier:

  <p align="center">
  <img src="img/results_bar.svg" width=600><br>
  </p>

The tuned models perform better than the baseline (expectedly) except for precision, likely because not enough parameter combinations were sampled in the training stage.

However, what we're really interested in is the most profitable model. The model performing best on each of the 'conventional' metrics, in addition to the one leading to the best profit, were explored in terms of profit curves:

<table>
  <tr>
    <td><img src="img/pc_MCC.svg"></td><td><img src="img/pc_recall.svg"></td>
  </tr>
    <tr>
    <td><img src="img/pc_precision.svg"></td><td><img src="img/pc_profit.svg"></td>
  </tr>
  </table>
  


The "actual max profit" is the highest possible profit that could have been obtained, and the "max predicted profit" is the profit using the threshold that computed the maximum profit at the model evaluation stage. Fittingly, the model optimized for profit makes the most profitable predictions on the test set -- an average predicted profit margin of 0.00466, vs the max possible 0.00438 from the baseline, which is about a 12% increase.

Here is the confusion matrix from the profit-optimized model:

  <p align="center">
  <img src="img/cfm.svg" width=600><br>
  </p>

The number of positive (true and false) values correspond to the number of items stocked ahead of time. Here, stock was purchased for 603/2466 (24.45%) of the sessions. 

The average profit value could be used to predict an actual profit using different values of R (revenue) per item as well as the number of predicted positives. The size of the test set contained 2466 sessions, which for R = $1 would grant a total profit of $11.5 (2466 * 0.00466 * $1), or if R were $100 the profit would be $1150, etc.

To show the profit calculation more explicitly together with the key initial conditions/assumptions restated, 

- Revenue per sale: $1<br>
- Cost per true positive: $0.90 (stocking cost: 90% of revenue or 10% profit margin)<br>
- Cost per false positive: $1.05 (stocking cost plus carrying cost)<br>
- The false negative's cost and revenue values are the opposite of the true positive, in that the "cost" is the revenue lost, and the "revenue" is the money saved not stocking the product. 

| Result | Number (a) | Cost (b) | Revenue (c) | Profit (a*\[c-b\]) |
| --- | --- |  --- | --- | --- | 
| True positive | 307 | 0.9  | 1 | 30.7 | 
| False positive | 296 | 1.05  | 1 | -14.8 | 
| True negative | 1819 |  0 | 0 | 0 | 
| False negative | 44 |  1 | 0.9 | -4.4 | 

Total profit: $11.50



### What drives sales?

Or at least is associated with them?

A feature importances plot from the profit-optimized model from the previous section gives:

  <p align="center">
  <img src="img/imp1.svg" width=600><br>
  </p>
  
Right away, it's clear that 'PageValues' is important, i.e. the number of web pages visited before making a purchase.

This can be corroborated with SHAP, which is an industry standard for finding feature contributions to model predictions. The following is a summary plot showing the feature importances and corresponding feature value:

  <p align="center">
  <img src="img/shap1.svg" width=600><br>
  </p>
  
 Again, 'PageValues' is at the top, as well as some features that appeared in the previous feature importances plot, like the November and May months. 
 
 One more thing we could do is go back to the model evaluation scores. Remember how the models formed distinct clusters in terms of performance for each metric? What if we labeled which ones had 'PageValues' included in the features?

<table>
  <tr>
    <td><img src="img/pv1.svg"></td><td><img src="img/pv2.svg"></td>
  </tr>
  </table>

The models with 'PageValues' clearly perform better, showing again the importance of this feature

## Conclusions

An XGBClassifier model was tuned to help inform stocking decisions in the event that false positives (false predictions of sales) incur a loss due to a carrying cost exceeding the normal profit margin. If the profit margin is 10% per product sold, and false positives lead to a -5% profit margin, then a model optimized for profitable predictions predicts that a product should be stocked for 24.45% of anticipated browsing sessions of the website. The profitability of the tuned XGBClassifier model leads to 6% more profitable predictions than the baseline untuned Random Forest Classifier model.

In terms of what drives sales, it is abundantly clear that 'PageValues' (the number of web pages visited prior to a sale in a browsing session) is an important factor. The e-commerce website should be optimized to encourage website exploration, e.g. by making the site as visually appealing as possible, easy to browse, and the product recommendation engine optimized to encourage the visitor to explore recommended products.
