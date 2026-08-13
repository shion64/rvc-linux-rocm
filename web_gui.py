import os
import sys
import json
import threading
import numpy as np

now_dir = os.path.dirname(os.path.abspath(__file__))

from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO, emit

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ["OMP_NUM_THREADS"] = "4"

realtime_config_path = os.path.join(now_dir, "configs", "config.json")
presets_dir = os.path.join(now_dir, "configs", "presets")

app = Flask(__name__)
app.config["SECRET_KEY"] = "rvc-web-gui"
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

import librosa
from tools.torchgate import TorchGate
import sounddevice as sd
import torch
import torch.nn.functional as F
import torchaudio.transforms as tat

from configs.config import Config
from infer import rtrvc as rvc_for_realtime
from i18n.i18n import I18nAuto
from tools.cuda_graph import cuda_graph_enabled, run_cuda_graph

i18n = I18nAuto()
flag_vc = False


class GUIConfig:
    def __init__(self):
        self.pth_path = ""
        self.index_path = ""
        self.pitch = 0
        self.formant = 0.0
        self.sr_type = "sr_model"
        self.block_time = 0.13
        self.threhold = -60
        self.crossfade_time = 0.08
        self.extra_time = 2.0
        self.I_noise_reduce = False
        self.O_noise_reduce = False
        self.rms_mix_rate = 0.0
        self.index_rate = 0.0
        self.f0method = "rmvpe"
        self.sg_hostapi = ""
        self.wasapi_exclusive = False
        self.sg_input_device = ""
        self.sg_output_device = ""


class VoiceChanger:
    def __init__(self):
        self.gui_config = GUIConfig()
        self.config = Config()
        self.function = "vc"
        self.delay_time = 0
        self.stream = None
        self.rvc = None
        self._stats = {"latency": 0, "inference_time": 0, "sola_offset": 0}
        self.update_devices()

    def update_devices(self, hostapi_name=None):
        hostapis = sd.query_hostapis()
        devices = sd.query_devices()
        hostapis_name = [h["name"] for h in hostapis]

        if hostapi_name not in hostapis_name:
            hostapi_name = hostapis_name[0]

        idx = hostapis_name.index(hostapi_name)
        self.hostapis = hostapis_name
        self.input_devices = [
            devices[d]["name"]
            for d in hostapis[idx]["devices"]
            if devices[d]["max_input_channels"] > 0
        ]
        self.output_devices = [
            devices[d]["name"]
            for d in hostapis[idx]["devices"]
            if devices[d]["max_output_channels"] > 0
        ]
        self.input_devices_indices = [
            d
            for d in hostapis[idx]["devices"]
            if devices[d]["max_input_channels"] > 0
        ]
        self.output_devices_indices = [
            d
            for d in hostapis[idx]["devices"]
            if devices[d]["max_output_channels"] > 0
        ]

    def set_devices(self, input_device, output_device):
        sd.default.device[0] = self.input_devices_indices[
            self.input_devices.index(input_device)
        ]
        sd.default.device[1] = self.output_devices_indices[
            self.output_devices.index(output_device)
        ]

    def get_device_samplerate(self):
        return int(sd.query_devices(device=sd.default.device[1])["default_samplerate"])

    def get_device_channels(self):
        max_channels = sd.query_devices(device=sd.default.device[0])["max_input_channels"]
        return min(max_channels, 2)

    def start_vc(self, values):
        global flag_vc
        self.set_devices(values["sg_input_device"], values["sg_output_device"])
        self.gui_config.pth_path = values["pth_path"]
        self.gui_config.index_path = values["index_path"]
        self.gui_config.pitch = float(values["pitch"])
        self.gui_config.formant = float(values["formant"])
        self.gui_config.index_rate = float(values["index_rate"])
        self.gui_config.rms_mix_rate = float(values["rms_mix_rate"])
        self.gui_config.threhold = float(values["threhold"])
        self.gui_config.block_time = float(values["block_time"])
        self.gui_config.crossfade_time = float(values["crossfade_length"])
        self.gui_config.extra_time = float(values["extra_time"])
        self.gui_config.f0method = values["f0method"]
        self.gui_config.sr_type = values["sr_type"]
        self.gui_config.I_noise_reduce = values.get("I_noise_reduce", False)
        self.gui_config.O_noise_reduce = values.get("O_noise_reduce", False)

        torch.cuda.empty_cache()
        self.rvc = rvc_for_realtime.RVC(
            self.gui_config.pitch,
            self.gui_config.formant,
            self.gui_config.pth_path,
            self.gui_config.index_path,
            self.gui_config.index_rate,
            self.config,
            self.rvc if self.rvc else None,
        )
        self.gui_config.samplerate = (
            self.rvc.tgt_sr
            if self.gui_config.sr_type == "sr_model"
            else self.get_device_samplerate()
        )
        self.gui_config.channels = self.get_device_channels()
        self.zc = self.gui_config.samplerate // 100
        self.block_frame = int(np.round(self.gui_config.block_time * self.gui_config.samplerate / self.zc)) * self.zc
        self.block_frame_16k = 160 * self.block_frame // self.zc
        self.crossfade_frame = int(np.round(self.gui_config.crossfade_time * self.gui_config.samplerate / self.zc)) * self.zc
        self.sola_buffer_frame = min(self.crossfade_frame, 4 * self.zc)
        self.sola_search_frame = self.zc
        self.extra_frame = int(np.round(self.gui_config.extra_time * self.gui_config.samplerate / self.zc)) * self.zc

        self.input_wav = torch.zeros(
            self.extra_frame + self.crossfade_frame + self.sola_search_frame + self.block_frame,
            device=self.config.device, dtype=torch.float32,
        )
        self.input_wav_denoise = self.input_wav.clone()
        self.input_wav_res = torch.zeros(160 * self.input_wav.shape[0] // self.zc, device=self.config.device, dtype=torch.float32)
        self.rms_buffer = np.zeros(4 * self.zc, dtype="float32")
        self.sola_buffer = torch.zeros(self.sola_buffer_frame, device=self.config.device, dtype=torch.float32)
        self.sola_den_kernel = torch.ones(1, 1, self.sola_buffer_frame, device=self.config.device, dtype=torch.float32)
        self.nr_buffer = self.sola_buffer.clone()
        self.output_buffer = self.input_wav.clone()
        self.skip_head = self.extra_frame // self.zc
        self.return_length = (self.block_frame + self.sola_buffer_frame + self.sola_search_frame) // self.zc
        self.fade_in_window = (torch.sin(0.5 * np.pi * torch.linspace(0.0, 1.0, steps=self.sola_buffer_frame, device=self.config.device, dtype=torch.float32)) ** 2)
        self.fade_out_window = 1 - self.fade_in_window
        self.resampler = tat.Resample(orig_freq=self.gui_config.samplerate, new_freq=16000, dtype=torch.float32).to(self.config.device)
        if self.rvc.tgt_sr != self.gui_config.samplerate:
            self.resampler2 = tat.Resample(orig_freq=self.rvc.tgt_sr, new_freq=self.gui_config.samplerate, dtype=torch.float32).to(self.config.device)
        else:
            self.resampler2 = None
        self.tg = TorchGate(sr=self.gui_config.samplerate, n_fft=4 * self.zc, prop_decrease=0.9).to(self.config.device)
        self.start_stream()
        flag_vc = True

    def stop_stream(self):
        global flag_vc
        flag_vc = False
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.stream = None

    def start_stream(self):
        global flag_vc
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
        extra_settings = {}
        if self.gui_config.wasapi_exclusive and sys.platform == "win32":
            extra_settings = sd.WasapiSettings(exclusive=True)
        self.stream = sd.Stream(
            channels=self.gui_config.channels,
            callback=self.audio_callback,
            blocksize=self.block_frame,
            samplerate=self.gui_config.samplerate,
            dtype="float32",
            extra_settings=extra_settings if extra_settings else None,
        )
        self.delay_time = (
            self.stream.latency[-1]
            + self.gui_config.block_time
            + self.gui_config.crossfade_time
            + 0.01
        )
        if self.gui_config.I_noise_reduce:
            self.delay_time += min(self.gui_config.crossfade_time, 0.04)
        self.stream.start()

    def audio_callback(self, indata, outdata, frames, times, status):
        import time as time_module
        start_time = time_module.perf_counter()
        indata = librosa.to_mono(indata.T)

        if self.gui_config.threhold > -60:
            indata = np.append(self.rms_buffer, indata)
            rms = librosa.feature.rms(
                y=indata, frame_length=4 * self.zc, hop_length=self.zc
            )[:, 2:]
            self.rms_buffer[:] = indata[-4 * self.zc:]
            indata = indata[2 * self.zc - self.zc // 2:]
            db_threhold = (
                librosa.amplitude_to_db(rms, ref=1.0)[0] < self.gui_config.threhold
            )
            for i in range(db_threhold.shape[0]):
                if db_threhold[i]:
                    indata[i * self.zc: (i + 1) * self.zc] = 0
            indata = indata[self.zc // 2:]

        self.input_wav[: -self.block_frame] = self.input_wav[self.block_frame:].clone()
        self.input_wav[-indata.shape[0]:] = torch.from_numpy(indata).to(self.config.device)
        self.input_wav_res[: -self.block_frame_16k] = self.input_wav_res[self.block_frame_16k:].clone()

        if self.gui_config.I_noise_reduce:
            self.input_wav_denoise[: -self.block_frame] = self.input_wav_denoise[self.block_frame:].clone()
            input_wav = self.input_wav[-self.sola_buffer_frame - self.block_frame:]
            input_wav = self.tg(input_wav.unsqueeze(0), self.input_wav.unsqueeze(0)).squeeze(0)
            input_wav[: self.sola_buffer_frame] *= self.fade_in_window
            input_wav[: self.sola_buffer_frame] += self.nr_buffer * self.fade_out_window
            self.input_wav_denoise[-self.block_frame:] = input_wav[: self.block_frame]
            self.nr_buffer[:] = input_wav[self.block_frame:]
            resample_input = self.input_wav_denoise[-self.block_frame - 2 * self.zc:]
            self.input_wav_res[-self.block_frame_16k - 160:] = run_cuda_graph(
                self.resampler, "realtime-input-resample",
                lambda audio: self.resampler(audio), resample_input,
            )[160:]
        else:
            resample_input = self.input_wav[-indata.shape[0] - 2 * self.zc:]
            self.input_wav_res[-160 * (indata.shape[0] // self.zc + 1):] = run_cuda_graph(
                self.resampler, "realtime-input-resample",
                lambda audio: self.resampler(audio), resample_input,
            )[160:]

        infer_wav = self.rvc.infer(
            self.input_wav_res,
            self.block_frame_16k,
            self.skip_head,
            self.return_length,
            self.gui_config.f0method,
        )

        if self.resampler2 is not None:
            infer_wav = run_cuda_graph(
                self.resampler2, "realtime-output-resample",
                lambda audio: self.resampler2(audio), infer_wav,
            )

        if self.gui_config.O_noise_reduce:
            self.output_buffer[: -self.block_frame] = self.output_buffer[self.block_frame:].clone()
            self.output_buffer[-self.block_frame:] = infer_wav[-self.block_frame:]
            infer_wav = self.tg(infer_wav.unsqueeze(0), self.output_buffer.unsqueeze(0)).squeeze(0)

        if self.gui_config.rms_mix_rate < 1:
            if self.gui_config.I_noise_reduce:
                input_wav = self.input_wav_denoise[self.extra_frame:]
            else:
                input_wav = self.input_wav[self.extra_frame:]
            rms1 = librosa.feature.rms(
                y=input_wav[: infer_wav.shape[0]].cpu().numpy(),
                frame_length=4 * self.zc, hop_length=self.zc,
            )
            rms1 = torch.from_numpy(rms1).to(self.config.device)
            rms1 = F.interpolate(rms1.unsqueeze(0), size=infer_wav.shape[0] + 1, mode="linear", align_corners=True)[0, 0, :-1]
            rms2 = librosa.feature.rms(
                y=infer_wav[:].cpu().numpy(),
                frame_length=4 * self.zc, hop_length=self.zc,
            )
            rms2 = torch.from_numpy(rms2).to(self.config.device)
            rms2 = F.interpolate(rms2.unsqueeze(0), size=infer_wav.shape[0] + 1, mode="linear", align_corners=True)[0, 0, :-1]
            rms2 = torch.max(rms2, torch.zeros_like(rms2) + 1e-3)
            infer_wav *= torch.pow(rms1 / rms2, 1.0 - self.gui_config.rms_mix_rate)

        conv_input = infer_wav[None, None, : self.sola_buffer_frame + self.sola_search_frame]
        cor_nom = F.conv1d(conv_input, self.sola_buffer[None, None, :])
        cor_den = torch.sqrt(F.conv1d(conv_input ** 2, self.sola_den_kernel) + 1e-8)
        sola_offset = torch.argmax(cor_nom[0, 0] / cor_den[0, 0])

        infer_wav = infer_wav[sola_offset:]
        infer_wav[: self.sola_buffer_frame] *= self.fade_in_window
        infer_wav[: self.sola_buffer_frame] += self.sola_buffer * self.fade_out_window
        self.sola_buffer[:] = infer_wav[self.block_frame: self.block_frame + self.sola_buffer_frame]

        outdata[:] = (
            infer_wav[: self.block_frame]
            .repeat(self.gui_config.channels, 1)
            .t()
            .cpu()
            .numpy()
        )

        total_time = time_module.perf_counter() - start_time
        self._stats["inference_time"] = round(total_time * 1000)
        self._stats["latency"] = round(self.delay_time * 1000)
        self._stats["sola_offset"] = int(sola_offset)
        socketio.emit("stats", self._stats)

    def get_stats(self):
        return self._stats


vc = VoiceChanger()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/devices")
def get_devices():
    hostapi = request.args.get("hostapi", vc.hostapis[0])
    vc.update_devices(hostapi_name=hostapi)
    return jsonify({
        "hostapis": vc.hostapis,
        "input_devices": vc.input_devices,
        "output_devices": vc.output_devices,
    })


@app.route("/api/models")
def get_models():
    weights_dir = os.path.join(now_dir, "assets", "weights")
    indices_dir = os.path.join(now_dir, "assets", "indices")
    models = [f for f in os.listdir(weights_dir) if f.endswith(".pth")] if os.path.exists(weights_dir) else []
    indices = [f for f in os.listdir(indices_dir) if f.endswith(".index")] if os.path.exists(indices_dir) else []
    return jsonify({"models": models, "indices": indices})


@app.route("/api/config")
def get_config():
    try:
        with open(realtime_config_path, "r", encoding="utf-8") as f:
            return jsonify(json.load(f))
    except Exception:
        return jsonify({})


@app.route("/api/preset/<model_name>")
def get_preset(model_name):
    preset_path = os.path.join(presets_dir, f"{model_name}.json")
    if os.path.exists(preset_path):
        with open(preset_path, "r", encoding="utf-8") as f:
            return jsonify(json.load(f))
    return jsonify({})


@app.route("/api/preset/<model_name>", methods=["POST"])
def save_preset(model_name):
    os.makedirs(presets_dir, exist_ok=True)
    preset_path = os.path.join(presets_dir, f"{model_name}.json")
    with open(preset_path, "w", encoding="utf-8") as f:
        json.dump(request.json, f)
    return jsonify({"status": "ok"})


@socketio.on("start")
def handle_start(data):
    global flag_vc
    if flag_vc:
        return
    try:
        threading.Thread(target=vc.start_vc, args=(data,), daemon=True).start()
        emit("status", {"running": True, "auto_monitor": data.get("auto_monitor", False)})
        # 設定保存
        config = {
            "pth_path": data["pth_path"],
            "index_path": data["index_path"],
            "sg_hostapi": data.get("sg_hostapi", ""),
            "sg_wasapi_exclusive": False,
            "sg_input_device": data["sg_input_device"],
            "sg_output_device": data["sg_output_device"],
            "sr_type": data["sr_type"],
            "threhold": float(data["threhold"]),
            "pitch": float(data["pitch"]),
            "rms_mix_rate": float(data["rms_mix_rate"]),
            "index_rate": float(data["index_rate"]),
            "block_time": float(data["block_time"]),
            "crossfade_length": float(data["crossfade_length"]),
            "extra_time": float(data["extra_time"]),
            "f0method": data["f0method"],
        }
        with open(realtime_config_path, "w", encoding="utf-8") as f:
            json.dump(config, f)
    except Exception as e:
        emit("error", {"message": str(e)})


@socketio.on("stop")
def handle_stop():
    vc.stop_stream()
    emit("status", {"running": False})


@socketio.on("update_param")
def handle_update_param(data):
    if hasattr(vc, "rvc") and vc.rvc:
        key = data.get("key")
        value = data.get("value")
        if key == "pitch":
            vc.gui_config.pitch = float(value)
            vc.rvc.change_key(float(value))
        elif key == "formant":
            vc.gui_config.formant = float(value)
            vc.rvc.change_formant(float(value))
        elif key == "index_rate":
            vc.gui_config.index_rate = float(value)
            vc.rvc.change_index_rate(float(value))
        elif key == "threhold":
            vc.gui_config.threhold = float(value)
        elif key == "rms_mix_rate":
            vc.gui_config.rms_mix_rate = float(value)




@app.route("/api/model_info")
def get_model_info():
    pth_path = request.args.get("pth", "")
    result = {"samplerate": None, "has_image": False}
    if pth_path and os.path.exists(pth_path):
        try:
            import torch
            cpt = torch.load(pth_path, map_location="cpu", weights_only=False)
            sr = cpt.get("sr", None)
            result["samplerate"] = str(sr) if sr else None
        except Exception:
            pass
        img_path = pth_path.replace(".pth", ".png")
        result["has_image"] = os.path.exists(img_path)
    return jsonify(result)


@app.route("/api/model_image")
def get_model_image():
    from flask import send_file
    pth_path = request.args.get("pth", "")
    img_path = pth_path.replace(".pth", ".png")
    if os.path.exists(img_path):
        return send_file(img_path, mimetype="image/png")
    return "", 404



@app.route("/api/model_image_upload", methods=["POST"])
def upload_model_image():
    from flask import request
    import shutil
    pth = request.form.get("pth", "")
    file = request.files.get("image")
    if not pth or not file:
        return jsonify({"status": "error"}), 400
    img_path = pth.replace(".pth", ".png")
    file.save(img_path)
    return jsonify({"status": "ok"})

@app.route("/api/monitor")
def monitor_route():
    import subprocess
    action = request.args.get("action", "enable")
    sink = request.args.get("sink", "")
    src_fl = "alsa_playback.python3.12:output_FL"
    src_fr = "alsa_playback.python3.12:output_FR"
    dst_fl = f"{sink}:playback_FL"
    dst_fr = f"{sink}:playback_FR"
    if action == "enable":
        subprocess.Popen(["pw-link", src_fl, dst_fl])
        subprocess.Popen(["pw-link", src_fr, dst_fr])
    else:
        subprocess.Popen(["pw-link", "-d", src_fl, dst_fl])
        subprocess.Popen(["pw-link", "-d", src_fr, dst_fr])
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    print("RVC WebGUI起動中: http://localhost:7865")
    socketio.run(app, host="0.0.0.0", port=7865, debug=False)