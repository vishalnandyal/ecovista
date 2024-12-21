// Pin configurations
const int irSensorPin = 2;       // IR sensor pin connected to digital pin 2
const int drySensorPin = 18;      // Dry sensor pin connected to digital pin 3
const int recycleSensorPin = 4;  // Recycle sensor pin connected to digital pin 4
const int hazardousSensorPin = 5;// Hazardous sensor pin connected to digital pin 5

void setup() {
  Serial.begin(9600);
  pinMode(irSensorPin, INPUT);
  pinMode(drySensorPin, INPUT);
  pinMode(recycleSensorPin, INPUT);
  pinMode(hazardousSensorPin, INPUT);
}

void loop() {
  // Read the current state of each sensor
  int irValue = digitalRead(irSensorPin);
  int dryValue = digitalRead(drySensorPin);
  int recycleValue = digitalRead(recycleSensorPin);
  int hazardousValue = digitalRead(hazardousSensorPin);
  
  // Print the status of each bin based on the sensor readings
  Serial.print("WET: ");
  Serial.println(irValue == LOW ? "Full" : "Empty");

  Serial.print("Dry: ");
  Serial.println(dryValue == LOW ? "Full" : "Empty");

  Serial.print("Recycle: ");
  Serial.println(recycleValue == LOW ? "Full" : "Empty");

  Serial.print("Hazardous: ");
  Serial.println(hazardousValue == LOW ? "Full" : "Empty");

  delay(5000); // Adjust the delay time as needed
}
