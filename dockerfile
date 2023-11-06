FROM python:3.11-slim
RUN pip install pipenv
WORKDIR /app
COPY ["Pipfile","Pipfile.lock","./"]
RUN pipenv install --system --deploy
COPY ["WS_train.py", "model_xgboost.bin","./"]
EXPOSE 9696
ENTRYPOINT [ "waitress-serve", "--listen=0.0.0.0:9696", "WS_train:app"]