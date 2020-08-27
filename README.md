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

True positive: Browsing session correctly predicted to result in sale. Money spent to acquire product; profit gained from sale without additional cost.
False positive: Browsing session incorrectly predicted to result in sale. Money spent to acquire product; but because the sale is not immediately made, the product must be stored, incurring carrying cost prior to a possible sale later
True negative: Browsing session correctly predicted not to result in sale. No money spent to acquire product; no profit.
False negative: Browsing session incorrectly predicted not to result in sale. Profit is lost, because we don't have the product to make the sale when the customer would've made a purchase. 

What about the false positive? If a sale is incorrectly predicted to be made, now we have inventory that must be stored, incurring cost to carry the inventory prior to an eventual sale or (worse) a disposal of the product. Let's assume the product eventually sells. How long is it stored before the sale? If stored long enough, the "profit" could turn out to be a net loss!

Using a 10% profit margin, a confusion matrix could be assembled in terms of revenue R and carrying cost C per sale:

| |Actual positive | Actual negative|  |
| --- | --- | --- | ---| 
|Predicted positive |0.1R | 0.1R - C| 
| Predicted negative|-0.1R |0 | 

For simplicity, we'll just use R = $1. There are other possible costs associated with some of the values, such as a customer going to other businesses for future purchases if they couldn't buy a product they intended to buy from us. However, for simplicity's sake, we'll stick with the above values for this example. Also, to reiterate, we're assuming that products bought that were not sold immediately will sell eventually; there's just a question of how long and how much the carrying cost will dig into the profit. 

To explore possible profits as well as establish a baseline model, we'll split the data into a 80:20 train:test split, train a Random Forest Classifier on the data, and compute profit curves at the range of decision thresholds for different false positive values and see what the maximum possible profits are. We'll use -0.05 (where C renders a negative profit), 0 (break-even), and 0.05 (small enough C to leave a positive profit):
