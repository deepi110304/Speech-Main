import argparse
import tempfile
from pathlib import Path
import asyncio

# ASR
from faster_whisper import WhisperModel

# Translation
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

# Mic recording & audio utils
import sounddevice as sd
import soundfile as sf
from pydub import AudioSegment

# TTS (Microsoft Neural voices)
import edge_tts


VOICE_BY_LANG = {
    "hi": "hi-IN-SwaraNeural",
    "ta": "ta-IN-PallaviNeural",
    "te": "te-IN-ShrutiNeural",
    "ml": "ml-IN-SobhanaNeural",
    "kn": "kn-IN-SapnaNeural",
    "en": "en-US-AriaNeural",
    "es": "es-ES-ElviraNeural",
    "fr": "fr-FR-DeniseNeural",
    "de": "de-DE-KatjaNeural",
    "it": "it-IT-ElsaNeural",
    "pt": "pt-BR-FranciscaNeural",
    "ru": "ru-RU-DariyaNeural",
    "ja": "ja-JP-NanamiNeural",
    "ko": "ko-KR-SunHiNeural",
    "zh": "zh-CN-XiaoxiaoNeural",
}

LANG_CODE_TO_M2M = {
    "en": "en",
    "hi": "hi",
    "ta": "ta",
    "te": "te",
    "ml": "ml",
    "kn": "kn",
    "es": "es",
    "fr": "fr",
    "de": "de",
    "it": "it",
    "pt": "pt",
    "ru": "ru",
    "ja": "ja",
    "ko": "ko",
    "zh": "zh",
}


def record_from_mic(seconds: int, samplerate: int = 16000, channels: int = 1) -> str:
    """Record audio from microphone and save as a temporary WAV file. Returns path."""
    print(f"\n[Mic] Recording for {seconds} seconds… Speak now!")
    audio = sd.rec(int(seconds * samplerate), samplerate=samplerate, channels=channels, dtype="float32")
    sd.wait()
    tmp_wav = Path(tempfile.mkstemp(suffix=".wav")[1])
    sf.write(tmp_wav, audio, samplerate)
    print(f"[Mic] Saved recording to: {tmp_wav}")
    return str(tmp_wav)


def load_audio_any(input_path: str) -> str:
    """Load audio file of many types and convert to WAV (16k mono) for ASR. Returns path."""
    print(f"[Audio] Loading: {input_path}")
    audio = AudioSegment.from_file(input_path)
    audio = audio.set_frame_rate(16000).set_channels(1)
    tmp_wav = Path(tempfile.mkstemp(suffix=".wav")[1])
    audio.export(tmp_wav, format="wav")
    print(f"[Audio] Converted to 16kHz mono WAV: {tmp_wav}")
    return str(tmp_wav)


def transcribe_with_whisper(wav_path: str, device: str = "auto"):
    """Transcribe audio using faster-whisper. Returns (text, detected_lang)."""
    print("\n[ASR] Loading faster-whisper model (small) …")
    model = WhisperModel("small", device=("cuda" if device == "cuda" else "auto"), compute_type="auto")

    print(f"[ASR] Transcribing: {wav_path}")
    segments, info = model.transcribe(
        wav_path,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500),
    )

    text_parts = [seg.text for seg in segments]
    transcript = " ".join(t.strip() for t in text_parts).strip()

    detected = info.language or ""
    print(f"[ASR] Detected language: {detected}")
    print(f"[ASR] Transcript: {transcript}")
    return transcript, detected


def translate_text_m2m(text: str, src_lang: str, tgt_lang: str) -> str:
    """Translate text using facebook/m2m100_418M. Returns translated string."""
    print("\n[MT] Loading M2M100 model (418M) … (first time may take a while)")
    tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M")
    model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M")

    src_tag = LANG_CODE_TO_M2M.get(src_lang, src_lang)
    tgt_tag = LANG_CODE_TO_M2M.get(tgt_lang, tgt_lang)

    tokenizer.src_lang = src_tag
    encoded = tokenizer(text, return_tensors="pt")
    generated_tokens = model.generate(
        **encoded,
        forced_bos_token_id=tokenizer.get_lang_id(tgt_tag),
        max_length=512,
    )
    out = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
    print(f"[MT] {src_lang} → {tgt_lang}: {out}")
    return out


async def tts_edge(text: str, voice: str, outfile_mp3: str):
    """Use edge-tts to synthesize speech to an MP3 file."""
    print(f"\n[TTS] Using voice: {voice}")
    communicate = edge_tts.Communicate(text, voice=voice)
    await communicate.save(outfile_mp3)
    print(f"[TTS] Saved audio to: {outfile_mp3}")


def pick_voice_for_lang(lang: str, explicit_voice: str | None = None) -> str:
    if explicit_voice:
        return explicit_voice
    return VOICE_BY_LANG.get(lang, VOICE_BY_LANG["en"])


def main():
    parser = argparse.ArgumentParser(description="Speech-to-Speech Translation demo")
    parser.add_argument("--input_audio", type=str, default=None, help="Path to input audio file (wav/mp3/m4a/flac).")
    parser.add_argument("--mic", type=int, default=None, help="Record from mic for N seconds (e.g., 8).")
    parser.add_argument("--source_lang", type=str, default=None, help="Source language code (auto if omitted)")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"], help="ASR device preference")
    parser.add_argument("--out_dir", type=str, default="outputs", help="Output folder for audio files")
    args = parser.parse_args()

    if not args.input_audio and not args.mic:
        parser.error("Provide --input_audio PATH or --mic SECONDS")

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    # Ask user for target languages interactively
    print("\nAvailable languages:", ", ".join(LANG_CODE_TO_M2M.keys()))
    lang_input = input("Enter target languages (space separated, e.g., 'hi es fr'): ").strip()
    target_langs = lang_input.split()

    # 1) Get audio
    if args.input_audio:
        wav_path = load_audio_any(args.input_audio)
    else:
        wav_path = record_from_mic(args.mic)

    # 2) ASR
    transcript, detected_lang = transcribe_with_whisper(wav_path, device=args.device)
    source_lang = args.source_lang or detected_lang or "en"

    print("\n───────── MULTI-LANGUAGE TRANSLATION ─────────")
    for tgt in target_langs:
        # 3) MT
        translated = translate_text_m2m(transcript, source_lang, tgt)

        # 4) TTS
        voice = pick_voice_for_lang(tgt)
        out_file = Path(args.out_dir) / f"output_{tgt}.mp3"
        asyncio.run(tts_edge(translated, voice, str(out_file)))

        print(f"\n[✔] {source_lang} → {tgt}: {translated}")
        print(f"    Audio saved to: {out_file}")

    print("\nAll translations done!")


def run_pipeline(audio_path, target_langs_str):
    """Reusable pipeline for Gradio app."""
    transcript, detected_lang = transcribe_with_whisper(audio_path)
    source_lang = detected_lang or "en"

    target_langs = target_langs_str.strip().split()
    results = []

    for tgt in target_langs:
        translated = translate_text_m2m(transcript, source_lang, tgt)
        voice = pick_voice_for_lang(tgt)
        out_file = f"outputs/output_{tgt}.mp3"
        asyncio.run(tts_edge(translated, voice, out_file))
        results.append((tgt, translated, out_file))

    return results


if __name__ == "__main__":
    main()