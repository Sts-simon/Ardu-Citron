#!/usr/bin/env python3
import argparse
import pathlib
import subprocess
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quality", type=float)
    args = parser.parse_args()
    sources = [
        ROOT / "test/native/test_firmware.cpp",
        *(ROOT / "main" / name for name in (
            "kalman.cpp", "pid.cpp", "mixer.cpp", "servo_output.cpp",
            "imu_health.cpp", "flight_mode.cpp", "protocol.cpp")),
    ]
    with tempfile.TemporaryDirectory(prefix="ardu-citron-native-") as temp:
        executable = pathlib.Path(temp) / "firmware-tests"
        subprocess.run([
            "g++", "-std=c++17", "-Wall", "-Wextra", "-Wpedantic", "-Werror",
            "-I" + str(ROOT / "test/native"), "-I" + str(ROOT / "main"),
            *(str(source) for source in sources), "-o", str(executable),
        ], check=True)
        subprocess.run([str(executable)], check=True)
    if args.quality is not None and not (0.0 <= args.quality <= 1.0):
        raise SystemExit("detector score outside protocol range")


if __name__ == "__main__":
    main()
