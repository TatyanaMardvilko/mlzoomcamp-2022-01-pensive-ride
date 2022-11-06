FROM python:3.8.12-slim

RUN pip install pipenv

WORKDIR /usr/app

COPY ["Pipfile", "Pipfile.lock", "./"]

RUN pipenv install --system --deploy

COPY ["src/predict.py",  "./src/predict.py"]
COPY ["models/RandomForestModel.bin", "./models/RandomForestModel.bin"]

EXPOSE 9696

ENTRYPOINT ["python", "src/predict.py"]