import requests
import json

# Orion Context Broker configuration
orion_host = "10.7.99.170"
orion_port = "1026"
orion_url = f"http://{orion_host}:{orion_port}/v2/subscriptions"

# Subscription payload
payload = {
    "description": "Subscription to changes in Daily_COVID_Cases_In_City_Geolocation entity",
    "subject": {
        "entities": [
            {
                "type": "Daily_COVID_Cases_In_City_Geolocation",
                "idPattern": ".*",
            }
        ],
        "condition": {"attrs": ["data"]},
        "expression": {"q": "codigo_cidade_IBGE==3550308"},
    },
    "notification": {
        "http": {"url": "http://10.3.225.205:5000/notifyv2"},
        "attrs": ["ultimo_confirmados_disponivel", "codigo_cidade_IBGE"],
    },
    "expires": "2024-12-25T14:00:00.00Z",
}

# Headers
headers = {
    "Content-Type": "application/json",
    "X-Auth-token": "Bearer 2090385d09d6884f9a189664101f87d18d207b1e",
}


response = requests.post(orion_url, headers=headers, data=json.dumps(payload))


try:
    print(response.status_code)
    print(response.json())
except Exception as e:
    print(f"Error decoding JSON response: {e} {response}")

from flask import Flask, request, jsonify

app = Flask(__name__)

my_list = []


def append_to_list(value):
    if len(my_list) >= 5:
        my_list.pop(0)
    my_list.append(value)
    print(f"List: {my_list}")


@app.route("/notifyv2", methods=["POST"])
def notify():
    notification = request.get_json()
    data = notification.get("data", [{}])[0]
    ultimo_confirmados_disponivel = data.get("ultimo_confirmados_disponivel", None)
    codigoIbge = data.get("codigo_cidade_IBGE", None)
    if(codigoIbge=='3550308'):
        print(f"Received notification: {ultimo_confirmados_disponivel}")
        append_to_list(ultimo_confirmados_disponivel)

    return jsonify({"status": "received"}), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
