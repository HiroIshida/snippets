#include <Wire.h>

#include "SparkFun_VCNL4040_Arduino_Library.h" //Library: http://librarymanager/All#SparkFun_VCNL4040
VCNL4040 proximitySensor;

#include "SparkFun_LPS25HB_Arduino_Library.h"  //Library: http://librarymanager/All#SparkFun_LPS25HB
LPS25HB pressureSensor;

void setup()
{
  Serial.begin(9600);
  Serial.println("Robotic Finger Sensor v2 Example");

  Wire.begin();
  Wire.setClock(400000); //Increase I2C bus speed to 400kHz

  proximitySensor.begin();
  pressureSensor.begin();
}

void loop()
{
  float pressure = pressureSensor.getPressure_hPa() * 100; //Convert hPa to Pa

  Serial.print(pressure);
  Serial.print(",");

  Serial.println();

  delay(10);
}
