FROM python:latest

COPY . .
RUN python -m pip install -r requirements.txt
RUN pip install ipywidgets && jupyter nbextension enable --py widgetsnbextension
CMD ["jupyter", "notebook", "--port=8888", "--no-browser", "--ip=0.0.0.0", "--allow-root"]