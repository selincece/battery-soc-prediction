#!/usr/bin/env python3
import json
import os
from typing import Any, Dict

import paho.mqtt.client as mqtt
import requests

BROKER_HOST = os.getenv("MQTT_BROKER_HOST", "localhost")
BROKER_PORT = int(os.getenv("MQTT_BROKER_PORT", "1883"))
TOPIC_IN = os.getenv("MQTT_TOPIC_IN", "battery/B0006")
TOPIC_OUT = os.getenv("MQTT_TOPIC_OUT", "predictions/B0006")
API_URL = os.getenv("API_URL", "http://localhost:8000")


def on_connect(client, userdata, flags, reason_code, properties=None):
    print("Connected to MQTT", reason_code)
    client.subscribe(TOPIC_IN)


def on_message(client, userdata, msg):
    try:
        payload = json.loads(msg.payload.decode("utf-8"))
        # Expecting payload with keys: voltage_v, current_a, temperature_c, soc
        features = [
            float(payload["voltage_v"]),
            float(payload["current_a"]),
            float(payload["temperature_c"]),
            float(payload["soc"]),
        ]
        resp = requests.post(f"{API_URL}/predict", json={"features": features}, timeout=5)
        resp.raise_for_status()
        pred = resp.json().get("soc_pred")
        out = {"soc_pred": pred, **payload}
        client.publish(TOPIC_OUT, json.dumps(out))
        print("Published prediction", out)
    except Exception as e:
        print("Error handling message:", e)


def main():
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(BROKER_HOST, BROKER_PORT, keepalive=30)
    client.loop_forever()


if __name__ == "__main__":
    main()
