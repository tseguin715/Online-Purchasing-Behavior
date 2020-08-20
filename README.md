# Optimizing business decisions from online purchasing behavior

### Introduction

Analytics data such as web pages visited, types of web page visited (e.g. related products) exit rates, and bounce rates of users for e-commerce websites provides a picture of the factors surrounding purchase decisions and the chance to inform business decisions such as inventory stocking in order to maximize profits. The <a href="https://archive.ics.uci.edu/ml/datasets/Online+Shoppers+Purchasing+Intention+Dataset">browsing session data of 12,330 sessions for an E-commerce website based in Turkey</a> hosted in a public dataset at the <a href="https://archive.ics.uci.edu/ml/index.php">UCI Machine Learning Repository</a> allows the opportunity to explore online purchasing behavior in terms of these factors. What drives online purchasing behavior and how can that be used to inform business decisions, such as optimal inventory stocking? 

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
  
### Strategy and quick look

This project has the following goals:

1. Establish a confusion matrix in terms of profit margin
2. Find a model that leads to the most profitable predictions
3. Determine feature importance, i.e. what drives the purchasing behavior?

How much money is earned per sale, and what are the costs associated with false positive and false negative predictions? Using 0.1 as a profit margin, we can rationalize the following confusion matrix:

True positive: Browsing session correctly predicted to result in sale. Money spent to acquire product, profit gained from sale.
False positive: Browsing session incorrectly predicted to result in sale. Money spent to acquire product, but because the sale is not immediately made, the product must be stored, incurring carrying cost prior to a possible sale later
True negative: Browsing session correctly predicted not to result in sale. No money spent to acquire product, no profit.
False negative: Browsing session incorrectly predicted not to result in sale. Profit is lost, because we don't have the product to make the sale when the customer would've made a purchase. 

We can assemble a confusion matrix in terms of the profit margin. A typical profit margin is 0.1. This gives 0.1, 0, and -0.1 for true positive, true negative, and false negative, respectively.

However, what about the false positive? If a sale is incorrectly predicted to be made, now we have inventory that must be stored, incurring cost to carry the inventory prior to an eventual sale or (worse) a disposal of the product. Let's assume the product eventually sells. How long is it stored before the sale? If stored long enough, the "profit" could turn out to be a net loss thanks to the carrying cost!

