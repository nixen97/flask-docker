FROM python:3.6.8
# FROM python:3.6.8-alpine3.8

WORKDIR /usr/src

# RUN apt update
RUN apt install -y libc-dev gcc libatlas-dev libatlas3-base

RUN pip install --upgrade pip

# pip
RUN pip install --no-cache-dir sklearn
RUN pip install --no-cache-dir flask gunicorn dash dash_core_components dash_html_components plotly

COPY ./server /usr/src

CMD ["gunicorn", "dashb:server", "-b 0.0.0.0:5000"]
# CMD ["ls"]