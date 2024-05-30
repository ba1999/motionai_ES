// Libraries
#include <Arduino_LSM9DS1.h>
#include <TensorFlowLite.h>
#include "ki_modell_data.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include <ArduinoBLE.h>
#include <ArduinoJson.h>

// Constants
#define NUM_SAMPLES 100
#define NUM_MOVEMENT 10
#define G_FORCE 9.81
#define G_THRESHOLD G_FORCE * 2

// Variables
float acX, acY, acZ;
float gyX, gyY, gyZ;
float valuesAX[100];
float valuesAY[100];
float valuesAZ[100];
float valuesGX[100];
float valuesGY[100];
float valuesGZ[100];
float minValues[6];
float maxValues[6];
float acXAVG[4] = { 0.848, 0.74, 0.918, 0.914 };
float acYAVG[4] = { 0.068, 0.074, 0.064, 0.02 };
float acZAVG[4] = { 0.482, 0.6, 0.414, 0.264 };
float gyXAVG[4] = { 2.632, 4.114, 8.894, 1.788 };
float gyYAVG[4] = { 31.044, 39.96, 1.684, 9.704 };
float gyZAVG[4] = { -5.488, -6.428, -1.32, -0.042 };
float means[6] = { 0.9786, 0.0745, 0.3037, -4.1858, 2.4569, 0.6723 };
float stdDevs[6] = {
  0.67618344,
  0.43351903,
  0.30864431,
  56.92824807,
  24.40493039,
  19.50881415
};

// TensorFlow Lite model setup.
namespace {
tflite::ErrorReporter *error_reporter = nullptr;
const tflite::Model *model = nullptr;
tflite::MicroInterpreter *interpreter = nullptr;
TfLiteTensor *input = nullptr;
TfLiteTensor *output = nullptr;
constexpr int kTensorArenaSize = 80 * 1024;  // Adjust the size according to your model.
uint8_t tensor_arena[kTensorArenaSize];
}

// BLE service and characteristic setup.
BLEService customService("0000FFE0-0000-1000-8000-00805F9B34FB");
BLEStringCharacteristic dataCharacteristic("0000FFE1-0000-1000-8000-00805F9B34FB", BLERead | BLEWrite | BLENotify, 512);

void setup() {
  Serial.begin(9600);

  // Initialize the IMU sensor.
  if (!IMU.begin()) {
    Serial.println("Failed to initialize IMU!");
    while (1)
      ;
  }
  Serial.println("IMU initialized!");

  // Load and verify the TensorFlow Lite model.
  model = tflite::GetModel(ki_modell_tflite);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("Model schema version mismatch!");
    while (1)
      ;
  }

  // Setup TensorFlow Lite interpreter.
  static tflite::AllOpsResolver resolver;
  static tflite::MicroInterpreter static_interpreter(
    model, resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    Serial.println("Failed to allocate tensors!");
    while (1)
      ;
  }

  // Get pointers to the model's input and output tensors.
  input = interpreter->input(0);
  output = interpreter->output(0);

  // Initialize BLE.
  pinMode(LED_BUILTIN, OUTPUT);
  if (!BLE.begin()) {
    Serial.println("BLE Fehler");
    while (1)
      ;
  }

  // Setup BLE service and characteristics.
  BLE.setLocalName("MotionAI");
  BLE.setAdvertisedService(customService);
  customService.addCharacteristic(dataCharacteristic);
  BLE.addService(customService);
  BLE.advertise();
}

void loop() {
  BLEDevice central = BLE.central();

  if (central) {
    while (central.connected()) {
      digitalWrite(LED_BUILTIN, HIGH);  // Turn on the LED when the central device is connected.
    

    // Read sensor data.
    IMU.readAcceleration(acX, acY, acZ);
    IMU.readGyroscope(gyX, gyY, gyZ);

    // Calculate the sum of the absolute accelerations to detect significant motion.
    float acSum = abs(acX * G_FORCE) + abs(acY * G_FORCE) + abs(acZ * G_FORCE);

    // Collect and process sensor data if motion is detected.
    if (acSum > G_THRESHOLD) {

      // Reset min and max values for new motion detection.
      for (int i = 0; i < 6; i++) {
        minValues[i] = 200.0;
        maxValues[i] = -200.0;
      }

      // Variables for accumulating sensor values and calculating averages.
      float acXSum = 0;
      float acYSum = 0;
      float acZSum = 0;
      float gyXSum = 0;
      float gyYSum = 0;
      float gyZSum = 0;
      float avgACX = 0;
      float avgACY = 0;
      float avgACZ = 0;
      float avgGYX = 0;
      float avgGYY = 0;
      float avgGYZ = 0;

      // Collect NUM_SAMPLES samples and normalize the data.
      for (int i = 0; i < NUM_SAMPLES; i++) {
        IMU.readAcceleration(acX, acY, acZ);
        IMU.readGyroscope(gyX, gyY, gyZ);

        // Accumulate sensor values.
        acXSum = acXSum + acX;
        acYSum = acYSum + acY;
        acZSum = acZSum + acZ;
        gyXSum = gyXSum + gyX;
        gyYSum = gyYSum + gyY;
        gyZSum = gyZSum + gyZ;

        // Normalize and store sensor data in the model's input tensor.
        input->data.f[i * 6 + 0] = (acX - means[0]) / stdDevs[0];
        input->data.f[i * 6 + 1] = (acY - means[1]) / stdDevs[1];
        input->data.f[i * 6 + 2] = (acZ - means[2]) / stdDevs[2];
        input->data.f[i * 6 + 3] = (gyX - means[3]) / stdDevs[3];
        input->data.f[i * 6 + 4] = (gyY - means[4]) / stdDevs[4];
        input->data.f[i * 6 + 5] = (gyZ - means[5]) / stdDevs[5];

        valuesAX[i] = acX;
        valuesAY[i] = acY;
        valuesAZ[i] = acZ;
        valuesGX[i] = gyX;
        valuesGY[i] = gyY;
        valuesGZ[i] = gyZ;
 

        // Update min and max values for each sensor.
        float currentValues[6] = { acX, acY, acZ, gyX, gyY, gyZ };
        for (int i = 0; i < 6; i++) {
          if (minValues[i] > currentValues[i]) {
            minValues[i] = currentValues[i];
          }

          if (maxValues[i] < currentValues[i]) {
            maxValues[i] = currentValues[i];
          }
        }
        delay(10);  // Delay between samples to manage sampling rate.
      }

      // Calculate average sensor values.
      avgACX = acXSum / 100.000;
      avgACY = acYSum / 100.000;
      avgACZ = acZSum / 100.000;
      avgGYX = gyXSum / 100.000;
      avgGYY = gyYSum / 100.000;
      avgGYZ = gyZSum / 100.000;

      // Invoke TensorFlow Lite model to perform inference.
      TfLiteStatus invoke_status = interpreter->Invoke();
      if (invoke_status == kTfLiteOk) {
        // Extract and print the model's predictions.
        float prediction1 = output->data.f[0];
        float prediction2 = output->data.f[1];
        float prediction3 = output->data.f[2];
        float prediction4 = output->data.f[3];
        Serial.println(String(prediction1) + " ; " + String(prediction2) + " ; " + String(prediction3) + " ; " + String(prediction4) + " ; ");
        
        // Determine the class with the highest confidence.
        int predictedClass = -1;
        float maxConfidence = 0.0;
        for (int i = 0; i < 4; i++) {
          float confidence = output->data.f[i];
          if (confidence > maxConfidence) {
            maxConfidence = confidence;
            predictedClass = i;
          }
        }

        Serial.print("Vorhergesagte Bewegung: ");
        Serial.println(predictedClass);

        // Custom motion classification based on average sensor values.
        int motion1 = motionClassification(avgACX, avgACY, avgACZ, avgGYX, avgGYY, avgGYZ);

        // Prepare and send data over BLE.
        StaticJsonDocument<500> doc;
        doc["KI"] = predictedClass;
        doc["THREAD"] = motion1;
        doc["CONF"] = maxConfidence;
        char buffer[500];
        serializeJsonPretty(doc, buffer);
        dataCharacteristic.writeValue(buffer);
        delay(50);

        // Prepare and send data over BLE.
        StaticJsonDocument<500> doc2;
        doc2["MINX"] = minValues[0];
        doc2["MINY"] = minValues[1];
        doc2["MINZ"] = minValues[2];
        doc2["MAXX"] = maxValues[0];
        doc2["MAXY"] = maxValues[1];
        doc2["MAXZ"] = maxValues[2];
        char buffer2[500];
        serializeJsonPretty(doc2, buffer2);
        dataCharacteristic.writeValue(buffer2);
        delay(50);

        // Prepare and send data over BLE.
        StaticJsonDocument<500> doc3;
        doc3["MINGX"] = minValues[3];
        doc3["MINGY"] = minValues[4];
        doc3["MINGZ"] = minValues[5];
        doc3["MAXGX"] = maxValues[3];
        doc3["MAXGY"] = maxValues[4];
        doc3["MAXGZ"] = maxValues[5];
        char buffer3[500];
        serializeJsonPretty(doc3, buffer3);
        dataCharacteristic.writeValue(buffer3);
        delay(50);

        // Prepare and send data over BLE.
        for (int j = 0; j < 10; j++) {
          StaticJsonDocument<500> doc;
          JsonArray valuesArrayX = doc.createNestedArray("AX");
          for (int i = 0; i < 10; i++) {
            valuesArrayX.add(valuesAX[(j * 10) + i]);
          }
          char buffer[500];
          serializeJsonPretty(doc, buffer);
          dataCharacteristic.writeValue(buffer);
          delay(100);
        }

        // Prepare and send data over BLE.
        for (int j = 0; j < 10; j++) {
          StaticJsonDocument<500> doc;
          JsonArray valuesArrayY = doc.createNestedArray("AY");
          for (int i = 0; i < 10; i++) {
            valuesArrayY.add(valuesAY[(j * 10) + i]);
          }
          char buffer[500];
          serializeJsonPretty(doc, buffer);
          dataCharacteristic.writeValue(buffer);
          delay(100);
        }

        // Prepare and send data over BLE.
        for (int j = 0; j < 10; j++) {
          StaticJsonDocument<500> doc;
          JsonArray valuesArrayZ = doc.createNestedArray("AZ");
          for (int i = 0; i < 10; i++) {
            valuesArrayZ.add(valuesAZ[(j * 10) + i]);
          }
          char buffer[500];
          serializeJsonPretty(doc, buffer);
          dataCharacteristic.writeValue(buffer);
          delay(100);
        }

        // Prepare and send data over BLE.
        for (int j = 0; j < 10; j++) {
          StaticJsonDocument<500> doc;
          JsonArray valuesArrayGX = doc.createNestedArray("GX");
          for (int i = 0; i < 10; i++) {
            valuesArrayGX.add(valuesGX[(j * 10) + i]);
          }
          char buffer[500];
          serializeJsonPretty(doc, buffer);
          dataCharacteristic.writeValue(buffer);
          delay(100);
        }

        // Prepare and send data over BLE.
        for (int j = 0; j < 10; j++) {
          StaticJsonDocument<500> doc;
          JsonArray valuesArrayGY = doc.createNestedArray("GY");
          for (int i = 0; i < 10; i++) {
            valuesArrayGY.add(valuesGY[(j * 10) + i]);
          }
          char buffer[500];
          serializeJsonPretty(doc, buffer);
          dataCharacteristic.writeValue(buffer);
          delay(100);
        }

        // Prepare and send data over BLE.
        for (int j = 0; j < 10; j++) {
          StaticJsonDocument<500> doc;
          JsonArray valuesArrayGZ = doc.createNestedArray("GZ");
          for (int i = 0; i < 10; i++) {
            valuesArrayGZ.add(valuesGZ[(j * 10) + i]);
          }
          char buffer[500];
          serializeJsonPretty(doc, buffer);
          dataCharacteristic.writeValue(buffer);
          delay(100);
        }

        // Prepare and send data over BLE.
        StaticJsonDocument<500> doc4;
        int end = 1;
        doc4["END"] = end;
        char buffer4[500];
        serializeJsonPretty(doc4, buffer4);
        dataCharacteristic.writeValue(buffer4);
      }
    }
    delay(10);  // Delay in the main loop.
    }
  }
  digitalWrite(LED_BUILTIN, LOW);  // Turn off the LED when the central device is disconnected.
}

// Function to classify motion based on sensor averages and predefined thresholds.
int motionClassification(float aX, float aY, float aZ, float gX, float gY, float gZ) {
  float smallestDistance = 90000.0;
  int motion = -1;

  // Calculate the Euclidean distance to predefined averages for each motion class.
  for (int i = 0; i < 4; i++) {
    float distance = sqrt(pow((aX - acXAVG[i]), 2) + pow((aY - acYAVG[i]), 2)+pow((aZ - acZAVG[i]), 2)+pow((gX - gyXAVG[i]), 2)+pow((gY - gyYAVG[i]), 2)+pow((gZ - gyZAVG[i]), 2));
    if (distance < smallestDistance) {
      smallestDistance = distance;
      motion = i;
    }
  }

  return motion;  // Return the closest motion class.
}
