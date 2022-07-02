# For more information, please refer to https://aka.ms/vscode-docker-python
FROM python:3.9
# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE 1
# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED 1
# Install pip requirements
ADD requirements.txt .
RUN python -m pip install -r requirements.txt
#RUN pip3 install torch==1.11.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html

WORKDIR /app
ADD . /app
# Expose PORT
EXPOSE 8501

# Switching to a non-root user, please refer to https://aka.ms/vscode-docker-python-user-rights
RUN useradd appuser && chown -R appuser /app 
USER appuser

WORKDIR /home/appuser
RUN chown -R appuser /home/appuser 

WORKDIR /app

RUN wget https://api.wandb.ai/files/azm630/MultiTask_NLU/4z51yotp/checkpoint-13160/pytorch_model.bin?_gl=1*y1qd4z*_ga*MTI3MTUyNDgzLjE2NTM4MTM1OTQ.*_ga_JH1SJHJQXJ*MTY1NjU3NDkxMS41NS4xLjE2NTY1NzQ5MzAuNDE. -O best_checkpoint.bin

ENTRYPOINT ["streamlit","run"]
# During debugging, this entry point will be overridden. For more information, please refer to https://aka.ms/vscode-docker-python-debug
CMD ["app.py"]