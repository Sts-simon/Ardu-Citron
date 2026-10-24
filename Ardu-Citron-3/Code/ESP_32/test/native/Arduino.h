#ifndef TEST_ARDUINO_H
#define TEST_ARDUINO_H

#include <stdint.h>

template <typename T>
static inline T constrain(T value, T low, T high) {
  return value < low ? low : (value > high ? high : value);
}

#endif // TEST_ARDUINO_H
