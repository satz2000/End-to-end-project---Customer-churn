# Customer-Churn-Prediction:
![Churn](https://user-images.githubusercontent.com/90024661/135493461-457a32f2-c03a-4dfa-a9e7-1d1a362dd5f1.png)

  Churn prediction means detecting which customers are likely to cancel a subscription to a service based on how they use the service. It is a critical prediction for many businesses because acquiring new clients often costs more than retaining existing ones. Once you can identify those customers that are at risk of cancelling, you should know exactly what marketing action to take for each individual customer to maximise the chances that the customer will remain.
#### **Why is it so important?**
  Customer churn is a common problem across businesses in many sectors. If you want to grow as a company, you have to invest in acquiring new clients. Every time a client leaves, it represents a significant investment lost. Both time and effort need to be channelled into replacing them. Being able to predict when a client is likely to leave, and offer them incentives to stay, can offer huge savings to a business.
#### **About This Project:**
  * In our dataset, Total amount of Monthly charges are around 16,056,169$ from that 18% of amount loss around 2862927% Due to the customer churn.        
  * Total number of customer around 7043 but 27% of people to be churn which around 1869 customer from the overall customer, 
  * So we need to predict the person who are all wants to be churn.Its very important to that company because they want new customer as well as retain the previous customer to stay in there company.
#### Steps involved in Model Deployment:
  * Data Analysis (EDA)
  * Data Preprocessing.
  * Feature Engineering. 
  * Feature Selection (SelectKBest)
  * Fit into Algorithm (ML Algorithm)
  * Hyper Parameter Tunning (RandomSearchCV)
  * Dump model (Pickle)
  * Creating Flask API (To deploy model)
#### Packages Used:
This project requires **Python** and the following packages are in below:
  * [Numpy](https://numpy.org/)
  * [Pandas](https://pandas.pydata.org/)
  * [Matplotlib](https://matplotlib.org/)
  * [Seaborn](https://seaborn.pydata.org/)
  * [Scikit-learn](https://scikit-learn.org/stable/)
  * [Scipy](https://www.scipy.org/)
  * [Imblearn](https://imbalanced-learn.org/stable/)
  * [Counter](https://docs.python.org/3/library/collections.html)
  * [Flask](https://flask.palletsprojects.com/en/2.0.x/)
#### How To Run:
  In this project, First you need to download dataset [Telco-Customer-churn.csv](https://github.com/satz2000/End-to-end-project---Customer-churn/blob/main/Telco-Customer-Churn.csv) Then open your commant prompt and run this code [pip install jupyterlab](https://jupyterlab.readthedocs.io/en/stable/getting_started/installation.html). After [pip install requirements.txt](https://github.com/satz2000/End-to-end-project---Customer-churn/blob/main/requirements.txt) all packages are needed in this project are automatically installed on your machine. After Download [app.py](https://github.com/satz2000/End-to-end-project---Customer-churn/blob/main/app.py) files and run [TelecomCustomerChurn.ipynb](https://github.com/satz2000/End-to-end-project---Customer-churn/blob/main/TelecomCustomerChurn.ipynb) files  into your machine And some inputs to check our model and Its accuracy of prediction
