"""Talking Avatar: type a script, give a portrait, get a lip-synced talking video.

Step 1: OmniVoice turns the script into speech, either with a voice designed from a description
        (voice design) or cloned from a reference clip (recorded, uploaded or a saved voice).
Step 2: InfiniteTalk animates the portrait to that speech, using WanGP's own Deepy
        "Infinitalk" template (lightx2v accelerator, 4 steps, 81-frame sliding windows).
Both jobs go through WanGP's normal queue, so only one model is in memory at a time.
"""
import html
import json
import math
import os
import re
import shutil
import time
from urllib.parse import quote

import gradio as gr

from shared.gradio.progress import WangpProgress
from shared.utils.plugins import WAN2GPPlugin

PlugIn_Name = "Talking Avatar"
PlugIn_Id = "TalkingAvatar"

TTS_MODEL = "omnivoice"
VIDEO_TOOL = "gen_video_with_speech"
VIDEO_TEMPLATE = "Infinitalk"
VIDEO_FPS = 25  # InfiniteTalk / MultiTalk output frame rate

# OmniVoice language selector values. Finnish is not in OmniVoice's dropdown but is
# recognised through auto-detection, so it maps to "auto".
LANGUAGES = [
    ("Auto-detect", "auto"),
    ("Italian", "italian"),
    ("Finnish (auto-detect)", "auto"),
    ("English", "english"),
    ("French", "french"),
    ("German", "german"),
    ("Spanish", "spanish"),
    ("Portuguese", "portuguese"),
]
LANGUAGE_LABELS = [label for label, _ in LANGUAGES]
LANGUAGE_VALUES = dict(LANGUAGES)

DEFAULT_VOICE = "male, middle-aged, low pitch"
DEFAULT_LANGUAGE = "English"
DEFAULT_SPEED = 0.85  # 1.0 = OmniVoice's natural pace, lower = slower
OMNIVOICE_TOKENS_PER_SECOND = 24000 / 960  # audio tokenizer: 24 kHz, hop 960 -> 25 tokens/s
DEFAULT_SCENE = "A man is speaking to the camera with natural facial expressions, subtle head movements and accurate lip sync."
RESOLUTIONS = ["832x480", "480x832", "640x640"]
AUDIO_EXTENSIONS = (".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac")

VOICE_DESIGNED = "Designed from a description"
VOICE_REFERENCE = "Cloned from a reference clip (same voice every time)"
VOICE_MODES = [VOICE_DESIGNED, VOICE_REFERENCE]
# Saved reference voices live in WanGP's git-ignored settings folder, not in the plugin folder,
# so they survive plugin reinstalls and are never committed.
VOICES_DIR = os.path.join("settings", "talking_avatar_voices")
# Above this length a reference is used without transcript: OmniVoice then trims it to ~15 s and
# transcribes it with Whisper (a full transcript would no longer match the trimmed audio).
MAX_TRANSCRIBED_REFERENCE_SECONDS = 20.0


def _audio_duration_seconds(path):
    try:
        import soundfile
        return float(soundfile.info(path).duration)
    except Exception:
        return None


def _spoken_seconds(path):
    """Length of a reference clip once silences are removed the way OmniVoice does it before cloning,
    so pauses or silence at the start/end of a recording don't make the speaker look slower."""
    try:
        import numpy as np
        import soundfile
        from models.TTS.omnivoice.modeling_omnivoice import (
            OMNIVOICE_AUTO_REF_LEAD_SILENCE_MS,
            OMNIVOICE_AUTO_REF_MID_SILENCE_MS,
            OMNIVOICE_AUTO_REF_TRAIL_SILENCE_MS,
        )
        from models.TTS.omnivoice.utils.audio import remove_silence

        data, sample_rate = soundfile.read(path, dtype="float32", always_2d=True)
        mono = np.mean(data, axis=1)[np.newaxis, :]
        trimmed = remove_silence(
            mono,
            sample_rate,
            mid_sil=OMNIVOICE_AUTO_REF_MID_SILENCE_MS,
            lead_sil=OMNIVOICE_AUTO_REF_LEAD_SILENCE_MS,
            trail_sil=OMNIVOICE_AUTO_REF_TRAIL_SILENCE_MS,
        )
        seconds = trimmed.shape[-1] / float(sample_rate)
        return seconds if seconds > 0.3 else None
    except Exception:
        return _audio_duration_seconds(path)


def _estimated_speech_seconds(text, ref_text=None, ref_audio=None):
    """OmniVoice's own rule-based estimate of how long the text takes at normal pace.
    With a transcribed reference clip, the estimate follows that speaker's own pace."""
    from models.TTS.omnivoice.utils.duration import RuleDurationEstimator

    estimator = RuleDurationEstimator()
    ref_seconds = _spoken_seconds(ref_audio) if ref_text and ref_audio else None
    if ref_seconds:
        tokens = estimator.estimate_duration(text, ref_text, ref_seconds * OMNIVOICE_TOKENS_PER_SECOND)
        if tokens > 0:
            return max(0.5, float(tokens) / OMNIVOICE_TOKENS_PER_SECOND)
    tokens = estimator.estimate_duration(text, "Nice to meet you.", 25)
    return max(0.5, float(tokens) / OMNIVOICE_TOKENS_PER_SECOND)


# ---- saved reference voices ------------------------------------------------
def _voice_name(name):
    """A file-safe voice name (letters incl. accented ones, digits, spaces, - and _)."""
    return re.sub(r"[^\w\- ]", "", str(name or "")).strip()[:60]


def _list_voices():
    if not os.path.isdir(VOICES_DIR):
        return []
    names = {os.path.splitext(f)[0] for f in os.listdir(VOICES_DIR) if f.lower().endswith(AUDIO_EXTENSIONS)}
    return sorted(names, key=str.lower)


def _voice_audio_path(name):
    for ext in AUDIO_EXTENSIONS:
        path = os.path.join(VOICES_DIR, name + ext)
        if os.path.isfile(path):
            return path
    return None


def _delete_voice_files(name):
    for ext in AUDIO_EXTENSIONS + (".json",):
        path = os.path.join(VOICES_DIR, name + ext)
        if os.path.isfile(path):
            os.remove(path)


def _save_voice(name, audio_path, transcript):
    os.makedirs(VOICES_DIR, exist_ok=True)
    data = None
    try:
        import soundfile

        data, sample_rate = soundfile.read(audio_path, always_2d=False)
    except Exception:
        pass  # a format soundfile cannot decode (e.g. m4a): keep the original file, OmniVoice loads it itself
    _delete_voice_files(name)
    if data is not None:
        target = os.path.join(VOICES_DIR, name + ".wav")
        soundfile.write(target, data, sample_rate)
    else:
        target = os.path.join(VOICES_DIR, name + os.path.splitext(audio_path)[1].lower())
        shutil.copyfile(audio_path, target)
    with open(os.path.join(VOICES_DIR, name + ".json"), "w", encoding="utf-8") as handle:
        json.dump({"transcript": (transcript or "").strip()}, handle, ensure_ascii=False, indent=2)
    return target


def _transcribe_clip(state, audio_path):
    """Whisper transcript of a reference clip (WanGP's configured Whisper, language auto-detected).
    Like Deepy's voice input, it takes the GPU only when WanGP is idle, claiming it through WanGP's
    GPU lock so no generation starts meanwhile; otherwise it runs on the CPU."""
    import torch
    from shared.deepy.transcription import transcribe_media
    from shared.utils.process_locks import release_GPU_ressources, try_acquire_GPU_ressources

    process_id = "talking_avatar_whisper"
    device = "cpu"
    if torch.cuda.is_available() and isinstance(state, dict):
        try:
            if try_acquire_GPU_ressources(state, process_id, "Talking Avatar voice transcription"):
                device = "cuda"
        except Exception:
            device = "cpu"
    try:
        return transcribe_media(audio_path, timestamp_type="none", device=device)["text"].strip(), device
    finally:
        if device == "cuda":
            release_GPU_ressources(state, process_id)


def _load_voice(name):
    """-> (audio_path, transcript) of a saved voice."""
    transcript = ""
    try:
        with open(os.path.join(VOICES_DIR, name + ".json"), encoding="utf-8") as handle:
            transcript = json.load(handle).get("transcript", "")
    except Exception:
        pass
    return _voice_audio_path(name), transcript


def _servable_copy(path):
    """Gradio refuses to serve files from settings/ (403), so the browser gets a copy in Gradio's own upload folder."""
    from gradio.utils import get_upload_folder

    folder = os.path.join(get_upload_folder(), "talking_avatar_voices")
    os.makedirs(folder, exist_ok=True)
    stamp = int(os.path.getmtime(path))
    target = os.path.join(folder, f"{stamp}_{os.path.basename(path)}")
    if not os.path.isfile(target):
        shutil.copyfile(path, target)
    return target


def _pick_file(files, extensions=None):
    files = [str(f) for f in (files or []) if f]
    if extensions:
        for f in files:
            if f.lower().endswith(extensions):
                return f
    return files[0] if files else None


VIDEO_EXTENSIONS = (".mp4", ".mkv", ".webm", ".mov")


def _video_duration_seconds(path):
    try:
        import cv2

        capture = cv2.VideoCapture(path)
        try:
            frames, fps = capture.get(cv2.CAP_PROP_FRAME_COUNT), capture.get(cv2.CAP_PROP_FPS)
        finally:
            capture.release()
        return frames / fps if fps > 0 else 0.0
    except Exception:
        return 0.0


def _pick_final_video(files):
    """The complete video of a sliding-window job. With WanGP's "keep intermediate sliding windows" option
    (on by default) the job also returns a partial video for each finished window, so take the longest
    one (ties: the most recent file)."""
    videos = [str(f) for f in (files or []) if f and str(f).lower().endswith(VIDEO_EXTENSIONS) and os.path.isfile(str(f))]
    if not videos:
        return None
    return max(videos, key=lambda path: (round(_video_duration_seconds(path), 2), os.path.getmtime(path)))


def _status_html(text):
    """A finished-style WanGP progress bar showing a plain status line."""
    bar = WangpProgress()
    bar.status(text)
    return bar.render(active=False)


SPEECH_PLACEHOLDER = '<div class="ta-speech"><div class="ta-speech-title">Speech</div><div class="ta-speech-empty">No speech yet</div></div>'


def _speech_player_html(audio_path):
    """A plain browser audio player. Gradio's waveform player sometimes keeps the previous take or fails
    to load a new one when its value changes during a running event; a fresh <audio> element never does."""
    if not audio_path:
        return SPEECH_PLACEHOLDER
    path = os.path.abspath(audio_path)
    url = "/gradio_api/file=" + quote(path.replace("\\", "/"), safe="/:")
    version = int(os.path.getmtime(path)) if os.path.exists(path) else 0
    name = html.escape(os.path.basename(path))
    return (
        '<div class="ta-speech"><div class="ta-speech-title">Speech</div>'
        f'<audio controls preload="auto" src="{url}?v={version}" style="width:100%"></audio>'
        f'<div class="ta-speech-name">{name} &middot; <a href="{url}" download>download</a></div></div>'
    )


def _relay(job_updates, wrap):
    """yield from a run_job generator, wrapping each progress HTML; returns run_job's return value."""
    while True:
        try:
            html = next(job_updates)
        except StopIteration as done:
            return done.value
        yield wrap(html)


class TalkingAvatarPlugin(WAN2GPPlugin):
    def __init__(self):
        super().__init__()
        self.name = PlugIn_Name
        self.version = "1.1.0"
        self.description = "Script + portrait -> talking video (OmniVoice + InfiniteTalk), designed or cloned voice."

    def setup_ui(self):
        # The WebUI-backed api_session needs the live Gradio state to reach WanGP's queue.
        self.request_component("state")
        self.add_tab(tab_id=PlugIn_Id, label=PlugIn_Name, component_constructor=self.create_config_ui)

    def on_tab_select(self, state):
        return None

    def on_tab_deselect(self, state):
        return None

    # ---- settings builders -------------------------------------------------
    @staticmethod
    def _tts_settings(api_session, script, language_label, voice_seed, speed, voice_mode, voice, ref_audio, ref_text):
        settings = api_session.get_default_settings(TTS_MODEL)
        cloned = voice_mode == VOICE_REFERENCE
        ref_text = (ref_text or "").strip() if cloned else ""
        if cloned and ref_text and (_audio_duration_seconds(ref_audio) or 0) > MAX_TRANSCRIBED_REFERENCE_SECONDS:
            ref_text = ""  # let OmniVoice trim the long clip and transcribe the part it keeps
        settings.update(
            {
                "prompt": script.strip(),
                # Voice design: alt_prompt holds the voice tags. Cloning: it holds the clip's
                # transcript (blank = OmniVoice transcribes the clip with Whisper).
                "alt_prompt": ref_text if cloned else (voice.strip() or DEFAULT_VOICE),
                "audio_prompt_type": "A" if cloned else "",
                "audio_guide": ref_audio if cloned else None,
                "model_mode": LANGUAGE_VALUES.get(language_label, "auto"),
                "seed": int(voice_seed),
                "repeat_generation": 1,
                "batch_size": 1,
            }
        )
        speed = float(speed or 1.0)
        if abs(speed - 1.0) > 0.01:
            # OmniVoice has no direct speed control in WanGP: ask for a target length
            # (estimated natural length / speed) and let it stretch the speech to fit.
            custom = dict(settings.get("custom_settings") or {})
            custom["adjust_speed_to_max_duration"] = "Yes"
            settings["custom_settings"] = custom
            natural = _estimated_speech_seconds(script, ref_text, ref_audio if cloned else None)
            settings["duration_seconds"] = round(natural / speed, 2)
        return settings

    @staticmethod
    def _video_settings(photo, audio_path, scene, resolution, video_seed):
        from shared.deepy import tool_settings

        settings = tool_settings.clone_tool_preset(VIDEO_TOOL, VIDEO_TEMPLATE)
        for key in ("client_id", "type", "model_filename"):
            settings.pop(key, None)
        duration = _audio_duration_seconds(audio_path)
        # InfiniteTalk trims the video to the audio; this is only an upper bound.
        video_length = max(81, math.ceil(duration * VIDEO_FPS) + VIDEO_FPS) if duration else settings.get("video_length", 737)
        settings.update(
            {
                "prompt": scene.strip() or DEFAULT_SCENE,
                "audio_guide": audio_path,
                "audio_prompt_type": "A",
                "image_refs": [photo],
                "video_prompt_type": "KI",
                "image_prompt_type": "",
                "resolution": resolution,
                "video_length": int(video_length),
                "seed": int(video_seed),
                "repeat_generation": 1,
                "batch_size": 1,
            }
        )
        return settings

    # ---- UI ----------------------------------------------------------------
    def _queue_session(self, api_session):
        """Server-side session that joins WanGP's queue directly (no browser round-trip); falls back to the UI bridge."""
        try:
            return self.deepy.web_session()
        except Exception:
            return api_session

    def create_config_ui(self, api_session):
        active = {"job": None, "stop": False, "last_speech": None}

        def live_progress_html():
            """The same progress bar the Media Generator shows (downloads, model loading, denoising steps)."""
            try:
                return self.deepy._service.generation_view().get("html")
            except Exception:
                return None

        def run_job(settings, label):
            """Generator: yields WanGP's progress-bar HTML while the job runs, returns the generated files."""
            job = self._queue_session(api_session).submit_task(settings)
            active["job"] = job
            last = None
            try:
                while not job.done:
                    html = live_progress_html()
                    if html and html != last:
                        last = html
                        yield html
                    time.sleep(0.3)
                result = job.result()
            finally:
                active["job"] = None
            if result.cancelled or active["stop"]:
                raise gr.Error("Stopped.")
            if not result.success:
                errors = list(result.errors or [])
                raise gr.Error(f"{label} failed: {errors[0].message if errors else 'no output'}")
            files = [str(f) for f in (result.generated_files or []) if f]
            return files

        def check_voice(voice_mode, ref_audio):
            if voice_mode == VOICE_REFERENCE and not (ref_audio and os.path.isfile(ref_audio)):
                raise gr.Error("Add a reference clip first: record it, upload it, or pick a saved voice.")

        def tts_settings(script, language, voice_mode, voice, ref_audio, ref_text, voice_seed, speed):
            return self._tts_settings(
                self._queue_session(api_session), script, language, voice_seed, speed, voice_mode, voice, ref_audio, ref_text
            )

        def generate(photo, script, language, voice_mode, voice, ref_audio, ref_text, voice_seed, speed, scene, resolution, video_seed):
            """Generator: yields (progress_html, speech, video)."""
            if not photo:
                raise gr.Error("Add a portrait photo first.")
            if not script or not script.strip():
                raise gr.Error("Write the text to speak first.")
            check_voice(voice_mode, ref_audio)
            active["stop"] = False

            yield _status_html("Step 1/2: speech - queued in WanGP..."), SPEECH_PLACEHOLDER, None
            settings = tts_settings(script, language, voice_mode, voice, ref_audio, ref_text, voice_seed, speed)
            files = yield from _relay(run_job(settings, "Speech"), lambda html: (html, gr.update(), gr.update()))
            audio_path = _pick_file(files, AUDIO_EXTENSIONS)
            if not audio_path or not os.path.isfile(audio_path):
                raise gr.Error(f"OmniVoice returned no audio file ({files}).")
            active["last_speech"] = (audio_path, script.strip())
            duration = _audio_duration_seconds(audio_path)
            ready = f"Speech ready ({duration:.1f}s)" if duration else "Speech ready"
            yield _status_html(f"{ready}. Step 2/2: video - queued in WanGP..."), _speech_player_html(audio_path), gr.update()
            settings = self._video_settings(photo, audio_path, scene, resolution, video_seed)
            files = yield from _relay(run_job(settings, "Video"), lambda html: (html, gr.update(), gr.update()))
            video_path = _pick_final_video(files)
            if not video_path:
                raise gr.Error(f"InfiniteTalk returned no video file ({files}).")
            yield _status_html("Done: video ready."), gr.update(), video_path

        def speech_only(script, language, voice_mode, voice, ref_audio, ref_text, voice_seed, speed):
            """Generator: yields (progress_html, speech)."""
            if not script or not script.strip():
                raise gr.Error("Write the text to speak first.")
            check_voice(voice_mode, ref_audio)
            active["stop"] = False
            yield _status_html("Speech - queued in WanGP..."), SPEECH_PLACEHOLDER
            settings = tts_settings(script, language, voice_mode, voice, ref_audio, ref_text, voice_seed, speed)
            files = yield from _relay(run_job(settings, "Speech"), lambda html: (html, gr.update()))
            audio_path = _pick_file(files, AUDIO_EXTENSIONS)
            if audio_path:
                active["last_speech"] = (audio_path, script.strip())
            yield _status_html("Done: speech ready."), _speech_player_html(audio_path)

        def switch_voice_mode(voice_mode):
            cloned = voice_mode == VOICE_REFERENCE
            # Re-read the voices folder: the list was built when WanGP started.
            return gr.update(visible=not cloned), gr.update(visible=cloned), gr.update(choices=_list_voices())

        def pick_saved_voice(name):
            """-> (ref_audio, ref_text, voice_name)"""
            if not name:
                return gr.update(), gr.update(), gr.update()
            audio_path, transcript = _load_voice(name)
            if not audio_path:
                raise gr.Error(f"Saved voice '{name}' not found.")
            return _servable_copy(audio_path), transcript, name

        def clear_clip():
            """-> (ref_audio, ref_text, saved_voice, voice_name): back to an empty clip, ready to record or load."""
            return None, "", None, ""

        def load_clip_file(state, file):
            """-> (ref_audio, ref_text, saved_voice, voice_name) from an audio file picked on disk."""
            path = getattr(file, "name", file)
            if not path or not os.path.isfile(path):
                return gr.update(), gr.update(), gr.update(), gr.update()
            if not path.lower().endswith(AUDIO_EXTENSIONS):
                raise gr.Error("Pick an audio file (" + ", ".join(AUDIO_EXTENSIONS) + ").")
            return path, transcribe_reference(state, path), None, ""

        def transcribe_reference(state, ref_audio):
            """-> ref_text. A new clip gets a new transcript (the previous one belonged to the old clip)."""
            if not (ref_audio and os.path.isfile(ref_audio)):
                return ""
            try:
                text, device = _transcribe_clip(state, ref_audio)
            except Exception as error:
                gr.Warning(f"Could not transcribe the clip ({error}). Type what is said, or leave it blank.")
                return ""
            if not text:
                gr.Warning("No speech detected in the clip.")
                return ""
            if device == "cpu":
                gr.Info("WanGP was busy, so the clip was transcribed on the CPU.")
            return text

        def save_voice(name, ref_audio, ref_text):
            """-> saved_voice dropdown"""
            name = _voice_name(name)
            if not name:
                raise gr.Error("Give the voice a name first.")
            if not (ref_audio and os.path.isfile(ref_audio)):
                raise gr.Error("Record, upload or reuse a clip first.")
            seconds = _audio_duration_seconds(ref_audio)
            if seconds is not None and seconds < 2.0:
                raise gr.Error(f"The clip is only {seconds:.1f}s long: record at least 3 seconds (5-15 s works best).")
            _save_voice(name, ref_audio, ref_text)
            gr.Info(f"Voice '{name}' saved.")
            return gr.update(choices=_list_voices(), value=name)

        def delete_voice(name):
            """-> (saved_voice dropdown, voice_name)"""
            if not name:
                return gr.update(), gr.update()
            _delete_voice_files(name)
            gr.Info(f"Voice '{name}' deleted.")
            return gr.update(choices=_list_voices(), value=None), ""

        def use_last_speech():
            """-> (voice_mode, ref_audio, ref_text, saved_voice)"""
            last = active.get("last_speech")
            if not last or not os.path.isfile(last[0]):
                raise gr.Error("No speech generated yet: use 'Preview voice' first.")
            audio_path, transcript = last
            seconds = _audio_duration_seconds(audio_path)
            if seconds and seconds > MAX_TRANSCRIBED_REFERENCE_SECONDS:
                gr.Info(
                    f"That speech is {seconds:.0f}s long: OmniVoice will keep about 15 s of it and transcribe that part itself. "
                    "A short preview (5-15 s) makes a cleaner reference."
                )
                transcript = ""
            return VOICE_REFERENCE, audio_path, transcript, None

        def stop():
            active["stop"] = True
            job = active.get("job")
            if job is not None and not job.done:
                job.cancel()

        with gr.Column():
            gr.Markdown(
                "### Talking Avatar\n"
                "Write the text, add your portrait, and get a lip-synced video of you speaking it.\n"
                "**1.** OmniVoice creates the speech from the voice description (no voice sample needed) · "
                "**2.** InfiniteTalk animates your photo to that speech. The video length follows the speech.\n\n"
                "**Voice:** a *designed* voice is invented from the description, so it can change when the text changes, even with "
                "the same seed. For the **same voice every time**, choose *Cloned from a reference clip*: record or upload 5-15 s "
                "of clear speech (or turn a preview you like into the reference with *Use last generated speech*), then save it as a named voice."
            )
            with gr.Row():
                with gr.Column(scale=1):
                    photo = gr.Image(label="Portrait (front-facing, well lit)", type="filepath", height=360)
                    resolution = gr.Dropdown(RESOLUTIONS, value=RESOLUTIONS[0], label="Video resolution (landscape / portrait / square)")
                    scene = gr.Textbox(label="Scene description (optional)", value=DEFAULT_SCENE, lines=2)
                    video_seed = gr.Number(label="Video seed (-1 = random)", value=-1, precision=0)
                with gr.Column(scale=1):
                    script = gr.Textbox(label="Text to speak", lines=8, placeholder="Scrivi qui il testo... / Kirjoita teksti tähän...")
                    language = gr.Dropdown(LANGUAGE_LABELS, value=DEFAULT_LANGUAGE, label="Language")
                    voice_mode = gr.Radio(VOICE_MODES, value=VOICE_DESIGNED, label="Voice")
                    with gr.Column(visible=True) as designed_box:
                        voice = gr.Textbox(
                            label="Voice description",
                            value=DEFAULT_VOICE,
                            info="Combine: male/female · child/teenager/young adult/middle-aged/elderly · very low/low/moderate/high/very high pitch · whisper · british/american/... accent",
                        )
                    with gr.Column(visible=False) as reference_box:
                        with gr.Row(equal_height=True):
                            saved_voice = gr.Dropdown(_list_voices(), value=None, label="Saved voices", scale=4)
                            delete_voice_btn = gr.Button("Delete", variant="secondary", size="sm", scale=1, min_width=80)
                        ref_audio = gr.Audio(
                            label="Reference clip: record 5-15 s of clear speech (quiet room, natural tone)",
                            sources=["microphone", "upload"],
                            type="filepath",
                        )
                        with gr.Row():
                            load_clip_btn = gr.UploadButton(
                                "Load an audio file...",
                                file_types=list(AUDIO_EXTENSIONS),
                                file_count="single",
                                variant="secondary",
                                size="sm",
                            )
                            clear_clip_btn = gr.Button("Clear clip (record a new one)", variant="secondary", size="sm")
                        ref_text = gr.Textbox(
                            label="What is said in the clip",
                            lines=2,
                            placeholder="Filled in automatically (Whisper) when you record or upload a clip. Correct it if needed: an exact transcript gives the best clone.",
                        )
                        transcribe_btn = gr.Button("Transcribe the clip again", variant="secondary", size="sm")
                        with gr.Row(equal_height=True):
                            voice_name = gr.Textbox(label="Voice name", placeholder="e.g. Lorenzo IT", scale=3)
                            save_voice_btn = gr.Button("Save voice", variant="secondary", scale=1, min_width=100)
                        use_last_btn = gr.Button("Use last generated speech as reference", variant="secondary", size="sm")
                    voice_seed = gr.Number(label="Voice seed (keep fixed for repeatable takes)", value=1234, precision=0)
                    speed = gr.Slider(0.6, 1.2, value=DEFAULT_SPEED, step=0.05, label="Speaking speed (1.0 = normal, lower = slower)")
            with gr.Row():
                preview_btn = gr.Button("Preview voice", variant="secondary")
                generate_btn = gr.Button("Generate talking video", variant="primary")
                stop_btn = gr.Button("Stop", variant="stop")
            # WanGP's own progress bar, refreshed while a job runs.
            status = WangpProgress.component(visible=True)
            # Show the video at its real pixel size (only shrink it to fit the window), centred.
            gr.HTML(
                "<style>"
                "#ta-video, #ta-video > div, #ta-video .wrap, #ta-video [data-testid] {height:auto !important; max-height:none !important}"
                "#ta-video video {width:auto !important; height:auto !important; max-width:100% !important; max-height:90vh !important;"
                " object-fit:contain !important; display:block; margin:0 auto}"
                "</style>",
                padding=False,
            )
            with gr.Row():
                audio_out = gr.HTML(SPEECH_PLACEHOLDER, container=True)
                video_out = gr.Video(label="Talking video", elem_id="ta-video")

        voice_mode.change(fn=switch_voice_mode, inputs=[voice_mode], outputs=[designed_box, reference_box, saved_voice], queue=False)
        saved_voice.change(fn=pick_saved_voice, inputs=[saved_voice], outputs=[ref_audio, ref_text, voice_name], queue=False)
        # The state component gives access to WanGP's GPU lock (requested in setup_ui).
        state_input = getattr(self, "state", None) or gr.State(None)
        for clip_event in (ref_audio.stop_recording, ref_audio.upload):
            clip_event(fn=transcribe_reference, inputs=[state_input, ref_audio], outputs=[ref_text])
        transcribe_btn.click(fn=transcribe_reference, inputs=[state_input, ref_audio], outputs=[ref_text])
        clip_outputs = [ref_audio, ref_text, saved_voice, voice_name]
        clear_clip_btn.click(fn=clear_clip, outputs=clip_outputs, queue=False)
        ref_audio.clear(fn=clear_clip, outputs=clip_outputs, queue=False)
        load_clip_btn.upload(fn=load_clip_file, inputs=[state_input, load_clip_btn], outputs=clip_outputs)
        save_voice_btn.click(fn=save_voice, inputs=[voice_name, ref_audio, ref_text], outputs=[saved_voice], queue=False)
        delete_voice_btn.click(
            fn=delete_voice,
            inputs=[saved_voice],
            outputs=[saved_voice, voice_name],
            js="(name) => (name && confirm('Delete the saved voice \"' + name + '\"?')) ? name : null",
            queue=False,
        )
        use_last_btn.click(fn=use_last_speech, outputs=[voice_mode, ref_audio, ref_text, saved_voice], queue=False)

        preview_btn.click(
            fn=speech_only,
            inputs=[script, language, voice_mode, voice, ref_audio, ref_text, voice_seed, speed],
            outputs=[status, audio_out],
            show_progress="hidden",
        )
        generate_btn.click(
            fn=generate,
            inputs=[photo, script, language, voice_mode, voice, ref_audio, ref_text, voice_seed, speed, scene, resolution, video_seed],
            outputs=[status, audio_out, video_out],
            show_progress="hidden",
        )
        stop_btn.click(fn=stop, queue=False)
