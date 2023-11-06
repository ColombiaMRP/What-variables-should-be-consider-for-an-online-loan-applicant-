import requests


url = "http://localhost:9696/WS_train"

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

response = requests.post(url, json=customer).json()

print(response)