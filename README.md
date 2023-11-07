# What variables should be considered for an online loan applicant
Machine learning project to analyze online peer-to-peer(p2p) lending data and with the objective of predict what borrowers are eligible (with high chances to pay off their debts) for getting a loan.

## Bussines Issued Description
Online peer-to-peer (P2P) lending has made the practice of borrowing and lending easy. In this form of lending, there is no in-person interview and a borrower can simply fill out an online form and get a loan approved. The information provided solely by a borrower is prone to exaggeration and distortion, especially when it comes to income. Every P2P lending company relies on a well-designed procedure that rejects borrowers with a high chance of not paying off their debts.

Rejecting anyone without a verified source of income is a relevant policy that lending platforms can roll out to help reduce the rate of bad loans. It is natural to suspect that if a person's income source cannot be verified, then they might default on the loan. However, from the perspective of a borrower, the verification process can be cumbersome and time-consuming and they may just switch to another platform due to this inconvenience.

## Analitical context

The data is downloaded from [LendingClub (LC) Statistics](https://www.lendingclub.com/info/download-data.action) and it contains all loans issued from 2007 - 2012 along with their current loan status (fully paid or charged off). There are ~50 variables that describe the borrowers and the loans; for the sake of reducing complexity, the company has already performed a pre-screening of these variables based on existing analyses from LendingClub to select nine relevant variables, such as annual income, LendingClub credit grade, home ownership, etc. Considering the main target to be able to predict if a lender is going to pay off his adquired debt or not according to the historical charateristics of previous lenders, we make three models to compete (logistic model, random forest and Xgboos) so that we can use it to create a Web service which is latter isolated in a docker container and used in production, namely, to make the decision of either lending or not to new potential borrowers. Used variables are:

|      annual_inc     |                                                 The self-reported annual income provided by the borrower during registration.                                                |
|:-------------------:|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
| verification_status |                                          Indicates if income was verified by LC or not verified                                         |
|      emp_length     |                       Employment length in years. Possible values are between 0 and 10 where 0 means less than one year and 10 means ten or more years.                      |
|    home_ownership   |             The home ownership status provided by the borrower during registration or obtained from the credit report. Our values are: RENT, OWN, MORTGAGE, OTHER            |
|       int_rate      |                                                                           Interest Rate on the loan                                                                          |
|      loan_amnt      | The listed amount of the loan applied for by the borrower. If at some point in time, the credit department reduces the loan amount, then it will be reflected in this value. |
|       purpose       |                                                           A category provided by the borrower for the loan request.                                                          |
|         term        |                                             The number of payments on the loan. Values are in months and can be either 36 or 60.                                             |
|        grade        |                                                                            LC assigned loan grade                                                                            |

## How to use this repository

### Pre-requesits

```python
python
git
docker
```

### Cloning repository

First you should clone the repository to your local machine

```python
git https://github.com/ColombiaMRP/What-variables-should-be-consider-for-an-online-loan-applicant-.git
```

### Virtual enviroment and Dependencies (Direct python running)

If you are going to directly run the jupyter notebook files or web service, you first have to set up the dedicated virtual enviroment and install dependencies. For doing so you should first go to the root of the project (the folder you have just cloned) and:

  1. Ensure that ```pipenv``` is installed. You can install it running:

      ```python
      pip install pipenv
      ```
     
  2. Set up the virtual enviroment and install dependencies. Including "-e ." makes the packages available within all the files in an editable mode:

      ```python
      pipenv install 
      ```
     You can see the list of installed packages using:

      ```python
      pipenv run pip freeze
      ```
     
  4. Now you can activate the VE and run python files for training the model and using it for predicting:

     ```python
      pipenv shell
      python train.py
      python predict.py
      ```
     Take into account that, for testing our model and making predictions, we use an example of a consumer with these chracteristics:

     ```python
     customer ={
      'annual_inc':-0.4,
      'verification_status':'Verified',
      'emp_length':'6 years',
      'home_ownership':'MORTGAGE',
      'int_rate':1,
      'loan_amnt':1.5,
      'purpose':'home_improvement',
      'term':'36 months',
      'grade':'C'
       }

### Web Service Deployment

To host the service locally using the virtual environment, you need to start the flask aplication running:

```python
waitress-serve --listen=0.0.0.0:9696 WS_train:app
```
Now the Flask application is running, you can make HTTP requests to port 9696 running the file ```WS_predict.py``` where the consumer whose characteristics were defined above. Namely, you should run:

```python
pipenv WS_predict.py
```

### Docker Deployment

In Case you want to deploy the web service using and specific and dedicated docker container for that, the steps you might follow are:

  1. download the python image you need for containerizing the providing files using docker. You can do this running:

     ```python
      docker run python python:3.11-slim
      ```
  2. Build an image using the dockerfile provided in the repository:

     ```python
     docker build -t "image_name" .
      ```
  3. To list all the Docker images on your system and verify that the image is there, use:

     ```python
     docker images
      ```
  4. Now the image has been created, you can run a container from it using (as shown in the image below):

      docker run -it --rm -p 9696:9696 "image_name"

![Running built immage](https://github.com/ColombiaMRP/What-variables-should-be-consider-for-an-online-loan-applicant-/blob/main/docker_train.png)

With the Flask application running inside Docker you can make HTTP requests to port 9696 running the file ```WS_predict.py``` where the consumer whose characteristics were defined above. Namely, you should run:

```python
pipenv WS_predict.py
```
![Results from image run](https://github.com/ColombiaMRP/What-variables-should-be-consider-for-an-online-loan-applicant-/blob/main/docker_predict.png)

### Final considerations

You can change the parameters to test out different scenarios by changing values or parameters in files ```predict.py``` (python direct running)  or ```WS_predict.py```(python web service/docker container running). 

**Allowed parameters for features (with restrictions):**

* annual_inc, int_rate and loan_amnt: Numerical values normalized  from -1.96 till 1.96
* purpose: ['debt_consolidation', 'credit_card', 'other', 'home_improvement', 'major_purchase', 'small_business', 'car', 'wedding', 'medical', 'moving', 'house', 'vacation', 'educational', 'renewable_energy']
* verification_status  ['Not Verified', 'Verified', 'Source Verified']
* emp_length = ['10+ years', '< 1 year', '2 years', '3 years', '4 years', '5 years', '1 year', '6 years', '7 years', '8 years', '9 years']
* home_ownership = ['RENT', 'MORTGAGE', 'OWN', 'OTHER']
* term = [' 36 months', ' 60 months']
* grade = ['B', 'A', 'C', 'D', 'E', 'F', 'G']

Once you are happy with the ressults you can run again these files in either case for making predictions in a new console window inside the project folder. 

I would like to give a special "thank-you" to all my colleagues, as I recieved a warm collaboration from them all during the procces of building my project. I also thank for your future and sincere feedback to keep improving my habilities at this beatifull work.
