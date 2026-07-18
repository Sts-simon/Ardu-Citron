#include <ESP32Servo.h>

Servo esc;
Servo servo1;
Servo servo2;
Servo servo3;

// Broches
const int PIN_ESC    = 25;

const int PIN_SERVO1 = 26;
const int PIN_SERVO2 = 27;
const int PIN_SERVO3 = 14;

const int PIN_BUZZER = 13;


// Test servo
void testServo(Servo &s)
{
  s.write(90);
  delay(150);

  s.write(30);
  delay(250);

  s.write(150);
  delay(250);

  s.write(90);
  delay(150);

  delay(300);
}


// Test ESC
void testESC()
{
  // Hélice retirée

  esc.writeMicroseconds(1000);  // armement / ralenti
  delay(1000);

  esc.writeMicroseconds(1300);  // petite accélération
  delay(600);

  esc.writeMicroseconds(1000);  // retour ralenti
  delay(800);
}


// Test buzzer passif
void testBuzzer()
{
  tone(PIN_BUZZER, 1000);
  delay(150);
  noTone(PIN_BUZZER);

  delay(100);

  tone(PIN_BUZZER, 1500);
  delay(150);
  noTone(PIN_BUZZER);

  delay(100);

  tone(PIN_BUZZER, 2000);
  delay(150);
  noTone(PIN_BUZZER);

  delay(300);
}


void setup()
{
  servo1.attach(PIN_SERVO1);
  servo2.attach(PIN_SERVO2);
  servo3.attach(PIN_SERVO3);

  esc.attach(PIN_ESC);

  pinMode(PIN_BUZZER, OUTPUT);


  // Initialisation ESC
  esc.writeMicroseconds(1000);
  delay(3000);
}


void loop()
{
  testServo(servo1);

  testServo(servo2);

  testServo(servo3);

  testESC();

  testBuzzer();

  delay(1000);
}