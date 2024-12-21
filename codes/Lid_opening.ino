#include <ESP32Servo.h>

const int trigPin = 5;    // Trigger pin of ultrasonic sensor
const int echoPin = 4;    // Echo pin of ultrasonic sensor
const int servoPin = 2;   // Pin for servo motor

Servo myservo;            // Create a servo object to control a servo

void setup() {
  Serial.begin(9600);
  pinMode(trigPin, OUTPUT);
  pinMode(echoPin, INPUT);
  myservo.attach(servoPin); // Attaches the servo on pin 2 to the servo object
}

void loop() {
  long duration, distance;
  digitalWrite(trigPin, LOW);  // Clear the trigger
  delayMicroseconds(2);
  digitalWrite(trigPin, HIGH); // Send a 10 microsecond pulse to trigger
  delayMicroseconds(10);
  digitalWrite(trigPin, LOW);

  duration = pulseIn(echoPin, HIGH);  // Read the echo pin, pulseIn returns the duration in microseconds
  distance = (duration / 2) * 0.0343; // Calculate distance in cm

  Serial.print("Distance: ");
  Serial.print(distance);
  Serial.println(" cm");

  // If an object is detected within 30cm, rotate the servo
  if (distance <= 30) {
    rotateServo(); 
    delay(2000); // Delay for 2 seconds to allow the lid to open before checking again
  }
}

void rotateServo() {
  myservo.write(90); // Rotate servo to 90 degrees (assuming it opens the lid)
  delay(5000);      // Wait for the servo to reach the position
  myservo.write(0);  // Rotate servo back to 0 degrees
}
