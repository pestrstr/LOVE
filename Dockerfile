FROM python:3.9

WORKDIR /LOVE

COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt

COPY . . 

CMD ["python", "evaluate.py" , "--pretrain_embed_path=output/love.emb", "--emb_dim=300"]