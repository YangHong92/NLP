# Deloying Bert Model

1. Download the parameter for the model
```
> mkdir assets
> python bin/download_model_state.py
```

2. Start the server using bash script
```
> chmod +x bin/start_server
> bin/start_server
```

3. In a new terminal, run below to test the model (replace 'hello' to any text you want to test)
```
> curl -H "Content-Type: application/json" -d '{"text":"hello"}' -X POST http://127.0.0.1:8000/predict
```