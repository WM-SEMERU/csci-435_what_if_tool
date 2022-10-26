# For more information, please refer to https://aka.ms/vscode-docker-python
FROM jupyter/scipy-notebook

# Install pip requirements
COPY requirements.txt .
RUN python -m pip install -r requirements.txt

