FROM python:3.6

#update
RUN apt-get update

#install requirements
COPY src/requirements.txt /tmp/requirements.txt
WORKDIR /tmp
RUN pip3 install -r requirements.txt

COPY src /api
COPY data/apiKey.txt /data/apiKey.txt
COPY data/ml-20m/links.csv /data/ml-20m/links.csv
COPY data/ml-20m/movies.csv /data/ml-20m/movies.csv
COPY responses /responses
WORKDIR /api/recommender_algo
RUN rm editable_svd.cpython-37m-x86_64-linux-gnu.so
RUN python setup.py build_ext --inplace

WORKDIR /api

CMD ["gunicorn", "-w", "1", "-b", ":5000", "-t", "500", "--reload", "wsgi:app"]