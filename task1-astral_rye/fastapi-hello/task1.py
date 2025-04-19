from fastapi import FastAPI 

app= FastAPI()

@app.get("/")
def read_hello():
	return "Hello World!"
