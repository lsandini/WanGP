"""Talking Avatar: type a script, give a portrait, get a lip-synced talking video.

Step 1: OmniVoice (voice design, no reference audio) turns the script into speech.
Step 2: InfiniteTalk animates the portrait to that speech, using WanGP's own Deepy
        "Infinitalk" template (lightx2v accelerator, 4 steps, 81-frame sliding windows).
Both jobs go through WanGP's normal queue, so only one model is in memory at a time.
"""
import html
import math
import os
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


def _audio_duration_seconds(path):
    try:
        import soundfile
        return float(soundfile.info(path).duration)
    except Exception:
        return None


def _estimated_speech_seconds(text):
    """OmniVoice's own rule-based estimate of how long the text takes at normal pace."""
    from models.TTS.omnivoice.utils.duration import RuleDurationEstimator

    tokens = RuleDurationEstimator().estimate_duration(text, "Nice to meet you.", 25)
    return max(0.5, float(tokens) / OMNIVOICE_TOKENS_PER_SECOND)


def _pick_file(files, extensions=None):
    files = [str(f) for f in (files or []) if f]
    if extensions:
        for f in files:
            if f.lower().endswith(extensions):
                return f
    return files[0] if files else None


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
        self.version = "1.0.0"
        self.description = "Script + portrait -> talking video (OmniVoice + InfiniteTalk)."

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
    def _tts_settings(api_session, script, voice, language_label, voice_seed, speed=1.0):
        settings = api_session.get_default_settings(TTS_MODEL)
        settings.update(
            {
                "prompt": script.strip(),
                "alt_prompt": voice.strip() or DEFAULT_VOICE,
                "audio_prompt_type": "",  # voice design: no reference audio
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
            settings["duration_seconds"] = round(_estimated_speech_seconds(script) / speed, 2)
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
        active = {"job": None, "stop": False}

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

        def generate(photo, script, language, voice, voice_seed, speed, scene, resolution, video_seed):
            """Generator: yields (progress_html, speech, video)."""
            if not photo:
                raise gr.Error("Add a portrait photo first.")
            if not script or not script.strip():
                raise gr.Error("Write the text to speak first.")
            active["stop"] = False

            yield _status_html("Step 1/2: speech - queued in WanGP..."), SPEECH_PLACEHOLDER, None
            settings = self._tts_settings(self._queue_session(api_session), script, voice, language, voice_seed, speed)
            files = yield from _relay(run_job(settings, "Speech"), lambda html: (html, gr.update(), gr.update()))
            audio_path = _pick_file(files, AUDIO_EXTENSIONS)
            if not audio_path or not os.path.isfile(audio_path):
                raise gr.Error(f"OmniVoice returned no audio file ({files}).")
            duration = _audio_duration_seconds(audio_path)
            ready = f"Speech ready ({duration:.1f}s)" if duration else "Speech ready"
            yield _status_html(f"{ready}. Step 2/2: video - queued in WanGP..."), _speech_player_html(audio_path), gr.update()
            settings = self._video_settings(photo, audio_path, scene, resolution, video_seed)
            files = yield from _relay(run_job(settings, "Video"), lambda html: (html, gr.update(), gr.update()))
            video_path = _pick_file(files, (".mp4", ".mkv", ".webm", ".mov"))
            if not video_path:
                raise gr.Error(f"InfiniteTalk returned no video file ({files}).")
            yield _status_html("Done: video ready."), gr.update(), video_path

        def speech_only(script, language, voice, voice_seed, speed):
            """Generator: yields (progress_html, speech)."""
            if not script or not script.strip():
                raise gr.Error("Write the text to speak first.")
            active["stop"] = False
            yield _status_html("Speech - queued in WanGP..."), SPEECH_PLACEHOLDER
            settings = self._tts_settings(self._queue_session(api_session), script, voice, language, voice_seed, speed)
            files = yield from _relay(run_job(settings, "Speech"), lambda html: (html, gr.update()))
            audio_path = _pick_file(files, AUDIO_EXTENSIONS)
            yield _status_html("Done: speech ready."), _speech_player_html(audio_path)

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
                "Tip: use **Preview voice** first to try voices quickly; keep the **Voice seed** fixed to get the same voice every time."
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
                    voice = gr.Textbox(
                        label="Voice description",
                        value=DEFAULT_VOICE,
                        info="Combine: male/female · child/teenager/young adult/middle-aged/elderly · very low/low/moderate/high/very high pitch · whisper · british/american/... accent",
                    )
                    voice_seed = gr.Number(label="Voice seed (keep fixed for a consistent voice)", value=1234, precision=0)
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

        preview_btn.click(
            fn=speech_only,
            inputs=[script, language, voice, voice_seed, speed],
            outputs=[status, audio_out],
            show_progress="hidden",
        )
        generate_btn.click(
            fn=generate,
            inputs=[photo, script, language, voice, voice_seed, speed, scene, resolution, video_seed],
            outputs=[status, audio_out, video_out],
            show_progress="hidden",
        )
        stop_btn.click(fn=stop, queue=False)
