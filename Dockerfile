FROM python:3.8
COPY . /trending-sentiments
WORKDIR /trending-sentiments
RUN pip install -r requirements.txt
EXPOSE 8501
ENTRYPOINT ["streamlit","run"]
CMD ["src/app.py"]