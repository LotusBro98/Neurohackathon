#include <ESP8266WiFi.h>


#define LED_PIN 3


// Replace these with your WiFi network settings
const char* ssid = "Voodoo Restoration"; //replace this with your WiFi network name
const char* password = "gatherroundhealingfire"; //replace this with your WiFi network password

void setup()
{
  delay(1000);
  Serial.begin(115200);
 
  WiFi.begin(ssid, password);

  Serial.println();
  Serial.print("Connecting");
  while (WiFi.status() != WL_CONNECTED)
  {
    delay(500);
    Serial.print(".");
  }

  Serial.println("success!");
  Serial.print("IP Address is: ");
  Serial.println(WiFi.localIP());

  pinMode(LED_PIN, OUTPUT);
  analogWrite(LED_PIN, 16);
}

void loop()
{
  
}
