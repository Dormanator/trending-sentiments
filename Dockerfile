FROM python:3.9
COPY . /trending-sentiments
WORKDIR /trending-sentiments
RUN pip install -r requirements.txt
EXPOSE 8501
ENTRYPOINT ["streamlit","run"]
CMD ["app/app.py"]