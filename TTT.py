import whisper
import sys

def get_word_timestamps(audio_path):
    print("Chargement du modèle Whisper (cela peut prendre quelques secondes)...")
    # Utilisation du modèle 'tiny' ou 'base' pour aller vite
    model = whisper.load_model("base")
    
    print("Transcription et calcul des timecodes par mot...")
    # Demande à Whisper de calculer les horodatages par mot (word_timestamps=True)
    result = model.transcribe(audio_path, word_timestamps=True)
    
    print("\n--- Horodatages mot par mot ---\n")
    for segment in result["segments"]:
        if "words" in segment:
            for word_info in segment["words"]:
                start = word_info["start"]
                word = word_info["word"]
                
                # Formatage du temps en MM:SS.ms
                minutes = int(start // 60)
                seconds = int(start % 60)
                millis = int((start % 1) * 1000)
                
                print(f"[{minutes:02d}:{seconds:02d}.{millis:03d}] {word.strip()}")

if __name__ == "__main__":
    # Exemple avec l'audio de la vidéo (tu peux utiliser yt-dlp pour télécharger l'audio avant)
    print("Téléchargement de l'audio de la vidéo...")
    import subprocess
    
    video_url = "https://www.youtube.com/watch?v=bZe5J8SVCYQ"
    audio_file = "audio.mp3"
    
    # Télécharge uniquement l'audio en mp3 avec yt-dlp
    subprocess.run(["yt-dlp", "-x", "--audio-format", "mp3", "-o", "audio.%(ext)s", video_url])
    
    # Lance l'analyse Whisper
    get_word_timestamps(audio_file)
