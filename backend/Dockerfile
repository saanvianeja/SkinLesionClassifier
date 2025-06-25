FROM python:3.11

WORKDIR /code

COPY . .

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Optional: Install system dependencies for image processing
# RUN apt-get update && apt-get install -y libgl1 libglib2.0-0

EXPOSE 5000

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "main:app"] 