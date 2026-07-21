#include <Wire.h>
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>

// Broches I2C définies selon ton câblage
#define PIN_SDA 23
#define PIN_SCL 22

Adafruit_MPU6050 mpu;

void setup() {
  // Initialisation du port série
  Serial.begin(115200);
  while (!Serial) {
    delay(10); // Attente de l'ouverture du moniteur série
  }

  Serial.println("\n--- TEST MPU6050 ESP32 ---");

  // Initialisation du bus I2C sur tes broches spécifiques
  Wire.begin(PIN_SDA, PIN_SCL);

  // Tentative de connexion au MPU6050
  if (!mpu.begin()) {
    Serial.println("❌ Erreur : MPU6050 non détecté ! Vérifie ton câblage (SDA/SCL).");
    while (1) {
      delay(10); // Bloque le programme si le capteur n'est pas trouvé
    }
  }

  Serial.println("✅ MPU6050 connecté avec succès !");
  
  // Configuration basique du capteur
  mpu.setAccelerometerRange(MPU6050_RANGE_8_G);
  mpu.setGyroRange(MPU6050_RANGE_500_DEG);
  mpu.setFilterBandwidth(MPU6050_BANDWIDTH_21_HZ);

  delay(100);
}

void loop() {
  // Structures pour stocker les données lues
  sensors_event_t a, g, temp;
  mpu.getEvent(&a, &g, &temp);

  // Affichage des données d'accélération (m/s²)
  Serial.print("Accel [m/s²] -> X: ");
  Serial.print(a.acceleration.x, 2);
  Serial.print(" | Y: ");
  Serial.print(a.acceleration.y, 2);
  Serial.print(" | Z: ");
  Serial.print(a.acceleration.z, 2);

  // Affichage des données du gyroscope (rad/s)
  Serial.print(" || Gyro [rad/s] -> X: ");
  Serial.print(g.gyro.x, 2);
  Serial.print(" | Y: ");
  Serial.print(g.gyro.y, 2);
  Serial.print(" | Z: ");
  Serial.println(g.gyro.z, 2);

  delay(100); // Pause de 100ms entre chaque lecture
}