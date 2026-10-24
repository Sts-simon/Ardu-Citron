#include <stdio.h>
#include "protocol.h"

int main() {
  const VisionPoseFrame frame{65535, 0xFFFFFFFEu, 1.25f, -2.5f, 0.75f,
                              10.0f, -20.0f, 179.0f, 0.9f};
  uint8_t bytes[VISION_FRAME_SIZE];
  if (!encodeVisionFrame(frame, bytes, sizeof(bytes))) return 1;
  for (size_t i = 0; i < sizeof(bytes); ++i) printf("%02x", bytes[i]);
  putchar('\n');
  return 0;
}
