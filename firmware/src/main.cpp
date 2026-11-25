#include "DHT.h"
#include <WiFi.h>
#include <PubSubClient.h>
#include "secrets.h"

#define DHTPIN 13
#define DHTTYPE DHT11

const char* TOPIC = "room/lab1/telemetry";
const unsigned long PUBLISH_INTERVAL_MS = 10000;

DHT dht(DHTPIN, DHTTYPE);
WiFiClient wiFiClient;
PubSubClient mqtt(wiFiClient);

unsigned long lastPublishMs = 0;

void connectWifi() {
  Serial.print("Connecting to WiFi");
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\nWIFI OK");
}

void connectMqtt() {
  char clientId[48];
  snprintf(clientId, sizeof(clientId), "esp32-lab1-%06llX", ESP.getEfuseMac() & 0xFFFFFFULL);

  while (!mqtt.connected()) {
    Serial.print("MQTT Connecting...");
    if (mqtt.connect(clientId)) {
      Serial.println("MQTT OK");
    } else {
      Serial.printf("failed (rc=%d), retrying\n", mqtt.state());
      delay(2000);
    }
  }
}

void setup() {
  Serial.begin(115200);
  dht.begin();
  connectWifi();
  mqtt.setServer(MQTT_HOST, MQTT_PORT);
}

void loop() {
  if (WiFi.status() != WL_CONNECTED) {
    connectWifi();
  }
  if (!mqtt.connected()) {
    connectMqtt();
  }
  mqtt.loop();

  unsigned long now = millis();
  if (now - lastPublishMs < PUBLISH_INTERVAL_MS) {
    return;
  }
  lastPublishMs = now;

  float t = dht.readTemperature();
  float h = dht.readHumidity();

  if (isnan(t) || isnan(h)) {
    Serial.println("Read failed, check wiring");
    return;
  }

  Serial.printf("Temp: %.1f C | Humidity: %.1f %%\n", t, h);

  char msg[128];
  snprintf(msg, sizeof(msg), "{\"temp_c\":%.1f,\"humidity_pct\":%.1f}", t, h);

  bool ok = mqtt.publish(TOPIC, msg);
  Serial.printf("publish=%d state=%d wifi=%d payload=%s\n",
                ok, mqtt.state(), WiFi.status(), msg);
}