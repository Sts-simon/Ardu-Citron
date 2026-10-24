#include <stdio.h>
#include "protocol.h"

int main() {
  const GuidanceCommandFrame command{9, 1234, 12.0f, -3.0f, 8.0f, 150, 0.95f};
  uint8_t bytes[GUIDANCE_FRAME_SIZE];
  if (!encodeGuidanceFrame(command, bytes, sizeof(bytes))) return 1;
  for (size_t i = 0; i < sizeof(bytes); ++i) printf("%02x", bytes[i]);
  putchar('\n');
  return 0;
}
