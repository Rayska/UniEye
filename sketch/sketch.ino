// UniEye - MCU side
// Reads the Modulino Knob / Thermo, drives the Modulino Buzzer and the LED matrix.
// All app logic lives in Python; this sketch only reports input events and
// renders whatever frame Python sends.

#include <Arduino_RouterBridge.h>
#include <Arduino_LED_Matrix.h>
#include <Arduino_Modulino.h>
#include <vector>

Arduino_LED_Matrix matrix;
ModulinoKnob knob;
ModulinoThermo thermo;
ModulinoBuzzer buzzer;

const uint8_t FRAME_SIZE = 8 * 13;
uint8_t frame[FRAME_SIZE] = {0};

const unsigned long LONG_PRESS_MS = 700;   // hold this long for a "long click"
const unsigned long DEBOUNCE_MS = 30;

// Knob filtering (the user can't really turn +1, +1, +1 and then -20 in one go):
// - a jump bigger than MAX_JUMP is a glitch: the Modulino library returns 0 when an
//   I2C read fails, and the counter can roll over. A jump straight to 0 is skipped as
//   a failed read; any other big jump counts as one step in the current direction.
// - a single step against the current direction within REVERSAL_MS of the last step
//   is contact bounce and is ignored (deliberate reversals are slower than that).
// - every reported turn is limited to -1 / +1.
const int MAX_JUMP = 4;
const unsigned long REVERSAL_MS = 200;

int16_t lastPos = 0;
int lastDir = 0;
unsigned long lastStepMs = 0;
bool wasPressed = false;
bool longSent = false;
unsigned long pressStart = 0;

// Python -> MCU: new 8x13 frame, brightness 0-7 per pixel
void draw(std::vector<uint8_t> newFrame) {
  size_t len = min(newFrame.size(), (size_t)FRAME_SIZE);
  memcpy(frame, newFrame.data(), len);
}

// Python -> MCU: play a tone on the Modulino Buzzer
void beep(int freq, int ms) {
  buzzer.tone(freq, ms);
}

// Python -> MCU: current temperature in Celsius
float get_temperature() {
  return thermo.getTemperature();
}

void setup() {
  Serial.begin(115200);
  matrix.begin();
  matrix.setGrayscaleBits(3);
  matrix.clear();

  Modulino.begin(Wire1);
  knob.begin();
  thermo.begin();
  buzzer.begin();
  lastPos = knob.get();

  Bridge.begin();
  Bridge.provide("draw", draw);
  Bridge.provide("beep", beep);
  Bridge.provide("get_temperature", get_temperature);
}

void readKnobTurn(unsigned long now) {
  int16_t pos = knob.get();
  int delta = pos - lastPos;
  if (delta == 0) return;

  if (abs(delta) > MAX_JUMP) {
    if (pos == 0) {
      Serial.print("knob: read 0 after "); Serial.print(lastPos);
      Serial.println(" -> failed read, skipped");
      return;                                    // keep lastPos, wait for a real read
    }
    if (lastDir == 0) {                           // no turn yet (e.g. the start-up read failed)
      lastPos = pos;
      return;
    }
    int step = lastDir;
    Serial.print("knob: jump "); Serial.print(delta);
    Serial.print(" -> counted as "); Serial.println(step);
    lastPos = pos;                                // resync to the real counter
    delta = step;
  } else {
    int dir = delta > 0 ? 1 : -1;
    if (lastDir != 0 && dir != lastDir && abs(delta) == 1 && now - lastStepMs < REVERSAL_MS) {
      Serial.println("knob: bounce against the turn -> ignored");
      return;                                     // keep lastPos: the bounce settles back
    }
    lastPos = pos;
  }

  lastDir = delta > 0 ? 1 : -1;
  lastStepMs = now;
  Bridge.notify("knob_turn", lastDir);            // always exactly -1 or +1
}

void loop() {
  unsigned long now = millis();
  readKnobTurn(now);

  bool pressed = knob.isPressed();
  if (pressed && !wasPressed) {
    pressStart = now;
    longSent = false;
  }
  if (pressed && !longSent && now - pressStart >= LONG_PRESS_MS) {
    Bridge.notify("knob_long", 1);  // fires while still held
    longSent = true;
  }
  if (!pressed && wasPressed && !longSent && now - pressStart >= DEBOUNCE_MS) {
    Bridge.notify("knob_click", 1);
  }
  wasPressed = pressed;

  matrix.draw(frame);
  delay(10);
}
