import requests
import json
import numpy as np
from tensorflow.keras.models import Sequential, load_model
import uuid


# Orion Context Broker configuration
orion_host = "10.7.99.170"
orion_port = "1026"
orion_url = f"http://{orion_host}:{orion_port}/v2/subscriptions"
orion_url_entities = f'http://{orion_host}:{orion_port}/v2/entities'
orion_url_subscribe = f"http://{orion_host}:{orion_port}/v2/subscriptions"


def get_all_subscriptions():
    # Get all subscriptions
    response = requests.get(orion_url)
    print(response.json())
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Failed to retrieve subscriptions. Status code: {response.status_code}")
        print(response.json())
        return []

def delete_subscription(subscription_id):
    # Delete a specific subscription
    delete_url = f'{orion_url}/{subscription_id}'
    response = requests.delete(delete_url)
    if response.status_code == 204:
        print(f"Successfully deleted subscription ID: {subscription_id}")
    else:
        print(f"Failed to delete subscription ID: {subscription_id}. Status code: {response.status_code}")
        print(response.json())

def delete_existing_subscriptions():
    # Get all subscriptions
    subscriptions = get_all_subscriptions()
    
    # Delete each subscription
    for subscription in subscriptions:
        subscription_id = subscription['id']
        if subscription['notification']['http']['url'] == "http://10.3.225.205:5000/notifyv2":
            delete_subscription(subscription_id)
    print("All subscriptions deleted")

delete_existing_subscriptions()
model_carregado = load_model('covid_lstm.h5')
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


def create_subscription():
    # Create subscription
    response = requests.post(orion_url, headers=headers, data=json.dumps(payload))

    try:
        print(response.status_code)
        print(response.json())
    except Exception as e:
        print(f"Error decoding JSON response: {e} {response}")
        
    print(response.status_code)
    if response.status_code == 201:
        print("Subscription created successfully")
    

create_subscription()


from flask import Flask, request, jsonify
import datetime

app = Flask(__name__)

my_list = []

def run_model(array):
  xtest = np.array(array)
  xtest_reshaped = np.reshape(xtest, (1, xtest.shape[0], 1))
  y_pred_carregado = model_carregado.predict(xtest_reshaped)
  print(f"Prediction: {y_pred_carregado}")
  publish_to_orion(y_pred_carregado.tolist())

def publish_to_orion(payload):
  entity = {
    "type": "predict_covid_cases",
    "id": f"predict_covid_cases_by_lstm{uuid.uuid4()}",
    "cidade" : {
      "type" : "Text",
      "value" : "São Paulo"
    },
    "dia1": {
      "type": "Real",
      "value": payload[0][0]
    },
    "dia2": {
      "type": "Real",
      "value": payload[0][1]
    },
    "dia3": {
      "type": "Real",
      "value": payload[0][2]
    },
    "dia4": {
      "type": "Real",
      "value": payload[0][3]
    },
    "dia5": {
      "type": "Real",
      "value": payload[0][4]
    },
    "timestamp":{
      "type": "Date",
      "value": datetime.datetime.now().isoformat()
    }
  }
  print(entity)
  response = requests.post(orion_url_entities, headers={"Content-Type": "application/json"}, data=json.dumps(entity))
  print(response.status_code)
  



def append_to_list(value):
    if len(my_list) >= 10:
        my_list.pop(0)
    my_list.append(value)
    print(f"List: {my_list}")
    
    if len(my_list) == 10:
        run_model(my_list)


@app.route("/notifyv2", methods=["POST"])
def notify():
    notification = request.get_json()
    data = notification.get("data", [{}])[0]
    ultimo_confirmados_disponivel = data.get("ultimo_confirmados_disponivel", None)
    codigoIbge = data.get("codigo_cidade_IBGE", None)
    print(f"Received notification: {codigoIbge}")
    print("SÃO PAULO")
    print(f"Received notification: {ultimo_confirmados_disponivel}")
    append_to_list(ultimo_confirmados_disponivel.get("value", 0))
    return jsonify({"status": "received"}), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
