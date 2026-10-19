#!/usr/bin/env python3

import argparse
import gc
import sys
from pathlib import Path

import torch
import whisperx


def format_time(seconds):
    """Convertit des secondes en HH:MM:SS.mmm."""
    if seconds is None:
        return "--:--:--.---"

    total_ms = round(float(seconds) * 1000)
    hours, remainder = divmod(total_ms, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, milliseconds = divmod(remainder, 1000)

    return (
        f"{hours:02d}:{minutes:02d}:"
        f"{secs:02d}.{milliseconds:03d}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Transcription audio avec WhisperX."
    )

    parser.add_argument(
        "audio",
        type=Path,
        help="Chemin du fichier audio"
    )

    parser.add_argument(
        "-o", "--output",
        type=Path,
        help="Chemin du fichier TXT de sortie"
    )

    parser.add_argument(
        "-m", "--model",
        default="small",
        choices=["tiny", "base", "small", "medium", "large-v3"],
        help="Modele Whisper (defaut : small)"
    )

    parser.add_argument(
        "-l", "--language",
        default=None,
        help="Langue audio, par exemple fr ou en. Auto si omis."
    )

    parser.add_argument(
        "--words",
        action="store_true",
        help="Un horodatage par mot plutot que par phrase"
    )

    args = parser.parse_args()

    audio_path = args.audio.expanduser().resolve()

    if not audio_path.is_file():
        sys.exit(f"Erreur : fichier introuvable : {audio_path}")

    output_path = (
        args.output.expanduser().resolve()
        if args.output
        else audio_path.with_suffix(".txt")
    )

    # Configuration pour processeur ou GPU CUDA.
    device = "cpu"
    compute_type = "int8"

    print(f"Audio : {audio_path}")
    print(f"Appareil : {device}")
    print(f"Modele : {args.model}")

    # Chargement du modele de transcription.
    model = whisperx.load_model(
        args.model,
        device=device,
        compute_type=compute_type,
        language=args.language
    )

    audio = whisperx.load_audio(str(audio_path))

    print("Transcription en cours...")

    result = model.transcribe(
        audio,
        batch_size=4 if device == "cuda" else 1
    )

    language = result.get("language", args.language or "unknown")
    segments = result.get("segments", [])

    del model
    gc.collect()

    if device == "cuda":
        torch.cuda.empty_cache()

    # Alignement temporel des mots.
    aligned = False

    if segments:
        try:
            print("Alignement des horodatages...")

            align_model, metadata = whisperx.load_align_model(
                language_code=language,
                device=device
            )

            result = whisperx.align(
                segments,
                align_model,
                metadata,
                audio,
                device,
                return_char_alignments=False
            )

            segments = result.get("segments", [])
            aligned = True

            del align_model
            gc.collect()

            if device == "cuda":
                torch.cuda.empty_cache()

        except Exception as exc:
            print(
                f"Attention : alignement impossible ({exc}). "
                "Utilisation des horodatages des segments."
            )

    # Ecriture du TXT.
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        file.write(f"Fichier audio : {audio_path.name}\n")
        file.write(f"Langue : {language}\n")
        file.write(
            "Mode : horodatages par mot\n"
            if args.words and aligned
            else "Mode : horodatages par segment\n"
        )
        file.write("=" * 65 + "\n\n")

        for segment in segments:
            if args.words and aligned:
                for word in segment.get("words", []):
                    text = word.get("word", "").strip()

                    if not text:
                        continue

                    start = format_time(word.get("start"))
                    end = format_time(word.get("end"))

                    file.write(f"[{start} --> {end}] {text}\n")
            else:
                text = segment.get("text", "").strip()

                if not text:
                    continue

                start = format_time(segment.get("start"))
                end = format_time(segment.get("end"))

                file.write(f"[{start} --> {end}] {text}\n")

    print(f"\nTermine : {output_path}")


if __name__ == "__main__":
    main()
