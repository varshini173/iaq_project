/*
  Indoor Air Quality — Data Logger
  ---------------------------------
  Use this sketch BEFORE you train anything. It just reads your sensors
  and prints CSV rows to Serial so you can build a real dataset instead of
  relying on the synthetic one from generate_dataset.py.

  How to use it:
    1. Flash this sketch, open Serial Monitor at 115200 baud.
    2. In Arduino IDE: Tools -> Serial Monitor, or better, use
       Tools -> "Serial Plotter"'s sibling, the Serial Monitor, with
       "Copy" so you can paste the output into a .csv file. Even simpler:
       on your PC run:
           python3 -c "import serial; s=serial.Serial('COM3',115200); \
                       [print(s.readline().decode().strip()) for _ in range(999999)]" \
                       > dataset_raw.csv
       (swap COM3 for your ESP32's port, e.g. /dev/ttyUSB0 on Linux/Mac)
    3. Move the sensor rig around: near a closed door, after burning
       incense/candle briefly (ventilated space, be sensible), in a
       stuffy room with the window shut for an hour, right after airing
       the room out, etc. You want readings that actually span
       good/moderate/poor conditions, not just your desk at 9am.
    4. YOU decide the label for each session based on your documented
       criteria (see generate_dataset.py's docstring) and add a "label"
       column afterwards — this sketch does NOT auto-label, because that
       labeling decision is exactly what your evaluator will ask you to
       justify. Don't let a heuristic silently pick your ground truth.

  Wiring: same as esp32_firmware/quickstart_version/ — see that file
  (DHT22 + MQ-135 through a level shifter, no CO2 sensor needed).
*/

#include <DHT.h>

#define DHTPIN 4
#define DHTTYPE DHT22
#define MQ135_PIN 34

DHT dht(DHTPIN, DHTTYPE);
const unsigned long SAMPLE_INTERVAL_MS = 2000;
unsigned long lastSample = 0;
bool headerPrinted = false;

void setup() {
  Serial.begin(115200);
  dht.begin();
  analogReadResolution(12);
  delay(2000);  // give DHT22 time to stabilize
}

void loop() {
  if (millis() - lastSample < SAMPLE_INTERVAL_MS) return;
  lastSample = millis();

  if (!headerPrinted) {
    Serial.println("timestamp_ms,temperature,humidity,gas");
    headerPrinted = true;
  }

  float t = dht.readTemperature();
  float h = dht.readHumidity();
  if (isnan(t) || isnan(h)) return;  // skip bad reads, don't log garbage

  int gas = analogRead(MQ135_PIN);

  Serial.printf("%lu,%.2f,%.2f,%d\n", millis(), t, h, gas);
}
