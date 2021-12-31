# Download the Model State
```
> mkdir assets
> python bin/download_model_state.py
```

# Serve Bert Model from Bash Script
```
> chmod +x bin/start_server
> bin/start_server
```

# Deloy Bert Model using Docker

1. Build docker image
```
> docker build -t my_bert_image .
```

2. Start docker container
```
> docker run -d -p 8000:8000 --name sentiment_cls my_bert_image
```

# Test Model Online Inference
Either run below in terminal (replace 'hello' with any text you want to test) or use Postman
```
> curl -H "Content-Type: application/json" -d '{"text":"hello"}' -X POST http://127.0.0.1:8000/predict
```