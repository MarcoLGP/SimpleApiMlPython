from flask import Flask, jsonify, request
from ml import ClientBank

app = Flask(__name__)

# Instanciando a classe do modelo de ml
ml_client_bank_instance = ClientBank()

@app.route("/")
def getIndex():
    return jsonify("Welcome to my api, by Marco Luca")

@app.route("/predicao", methods=["POST"])
def postPredicao():
    request_data = request.get_json()
    idade = request_data["idade"]
    conta_corrente = request_data["conta_corrente"]
    return jsonify(ml_client_bank_instance.predictClient(age= idade, balance= conta_corrente))

@app.route("/score", methods= ["GET"])
def getScore():
    return jsonify(ml_client_bank_instance.getMlScore())

if __name__ == '__main__':
    app.run(debug=True)
