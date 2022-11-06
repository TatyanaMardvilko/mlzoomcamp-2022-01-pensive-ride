import requests

url = "http://localhost:9696/predict"

student = {
    "gender": "female",
    "race_ethnicity": "group_C",
    "parental_level_of_education": "bachelor's_degree",
    "lunch": "standard",
    "test_preparation_course": "completed"
}

response = requests.post(url, json=student).json()

print(response)
