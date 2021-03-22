FROM python:3.8
COPY . /trending-sentiments
WORKDIR /trending-sentiments
RUN apt-get update \
    && apt-get install -y --no-install-recommends python3-dev \
    && rm -rf /var/lib/apt/lists/* \
    && pip install -r requirements.txt
RUN python setup.py
EXPOSE 8501
ENTRYPOINT ["streamlit","run"]
CMD ["src/app.py"]