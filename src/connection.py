import requests
import json

# Orion Context Broker configuration
orion_host = '10.7.99.170'
orion_port = '3010'
orion_url = f'http://{orion_host}:{orion_port}/v2/subscriptions'

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
        "condition": {
            "attrs": [
                "data"
            ]
        }
    },
    "notification": {
        "http": {
            "url": "http://10.7.99.170:5000/notify"
        },
        "attrs": [
            "ultimo_confirmados_disponivel"
        ]
    },
    "expires": "2024-12-25T14:00:00.00Z"
}

# Headers
headers = {
    'Content-Type': 'application/json',
    'Authorization': 'Bearer ee7ff0524bd473d4c8ffcb0ec8ee2d3a2af14fa5'
}

# Make the request
response = requests.post(orion_url, headers=headers, data=json.dumps(payload))

# Print the response
try:
  print(response.status_code)
  print(response.json())
except Exception as e:
  print(f"Error decoding JSON response: {e} {response}")

from flask import Flask, request, jsonify

app = Flask(__name__)

items = []

@app.route('/notify', methods=['POST'])
def notify():
    notification = request.get_json()
    print(f"Received notification: {notification}")
    items.push(notification.ultimo_confirmados_disponivel)
    return jsonify({'status': 'received'}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)