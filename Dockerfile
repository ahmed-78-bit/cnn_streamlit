FROM python:3.9-slim

COPY ./ ./

# pip-install
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r "requirements.txt"

EXPOSE 8501

ENTRYPOINT [ "streamlit", "run"]
CMD ["app.py"]