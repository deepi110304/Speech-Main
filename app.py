import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # hide INFO and WARNING logs

import gradio as gr
from speech_to_speech_translation import (
    transcribe_with_whisper,
    translate_text_m2m,
    pick_voice_for_lang,
    tts_edge,
)
import asyncio


def translate_speech(input_audio, target_langs):
    # 1. Transcribe
    transcript, detected_lang = transcribe_with_whisper(input_audio)

    results = {}
    for lang in target_langs:
        # 2. Translate
        translated = translate_text_m2m(transcript, detected_lang, lang)

        # 3. TTS
        voice = pick_voice_for_lang(lang)
        out_file = f"output_{lang}.mp3"
        asyncio.run(tts_edge(translated, voice, out_file))

        results[lang] = (translated, out_file)

    return transcript, results


def ui(audio, langs):
    transcript, results = translate_speech(audio, langs)

    out_text = f"Transcript: {transcript}\n\n"
    out_audios = []

    for lang, (txt, audio_file) in results.items():
        out_text += f"{lang}: {txt}\n"
        out_audios.append(audio_file)

    return out_text, out_audios


demo = gr.Interface(
    fn=ui,
    inputs=[
        gr.Audio(sources=["microphone", "upload"], type="filepath", label="Input Audio"),
        gr.CheckboxGroup(
            choices=["hi", "es", "fr", "de", "ja", "ko", "zh"],
            label="Select Target Languages",
        ),
    ],
    outputs=[
        gr.Textbox(label="Translations"),
        gr.File(label="Translated Speech Files"),  # return list of files
    ],
    title="Speech-to-Speech Translation",
    description="Upload or record speech, select target languages, and get translations + spoken output.",
)

if __name__ == "__main__":
    demo.launch()