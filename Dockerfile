    #from python:3.6.8
FROM python:3.6.8-alpine3.8

WORKDIR /usr/src

# Gets purged
RUN apk add --no-cache --virtual .build-deps musl-dev postgresql-dev libc-dev gcc

# Doesn't get purged
RUN apk add --no-cache postgresql-libs libxslt-dev

RUN pip install --upgrade pip

# pip
RUN pip install --no-cache-dir flask gunicorn dash dash_core_components dash_html_components plotly sklearn

RUN apk --purge del .build-deps

COPY ./server /usr/src

RUN python -c 'import RSA; RSA.BuildKey()'

CMD ["gunicorn", "-w 4", "-b 0.0.0.0:5000", "app:dashb"]
