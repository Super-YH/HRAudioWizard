import numpy as np
import scipy
import librosa
import pyaudio
import soundfile as sf
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QLabel, QFileDialog, QProgressBar, QComboBox,
                             QCheckBox, QSpinBox, QListWidget, QGroupBox, QRadioButton,
                             QMessageBox, QTabWidget)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
import warnings
import argparse
import os
import sys
import hashlib
import base64
import datetime
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import secrets
import queue
import sys
import tqdm

# Constants
FFTSIZE = 4096
HOPSIZE = 2048

CHUNK_SIZE = 1024  # Increased for better frequency resolution
SAMPLE_RATE = 44100

class Authed:
    def __init__(self):
        self.auth_mode = 1
        self.authorized = -1
        self.date = -1
        self.licensekey = b""

auth_info = Authed()

def generate_key(password, salt):
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=100000,
    )
    return base64.urlsafe_b64encode(kdf.derive(password.encode()))

def encrypt_license(license_data, password):
    salt = secrets.token_bytes(16)
    key = generate_key(password, salt)
    f = Fernet(key)
    return salt + f.encrypt(license_data.encode())

def decrypt_license(encrypted_license, password):
    salt, encrypted_data = encrypted_license[:16], encrypted_license[16:]
    key = generate_key(password, salt)
    f = Fernet(key)
    return f.decrypt(encrypted_data).decode()

def auth(licensefile):
    try:
        with open(licensefile, "rb") as f:
            encrypted_license = f.read()
    except FileNotFoundError:
        return 0

    # ハードコードされたパスワードの代わりに環境変数を使用
    password = os.environ.get('LICENSE_PASSWORD', 'default_password')

    try:
        decrypted_license = decrypt_license(encrypted_license, password)
        dateval, hash_c = decrypted_license.split(",")

        # より複雑な認証ロジック
        key_1 = int(hashlib.sha256(password.encode()).hexdigest(), 16)
        authdate = int((int(dateval) ^ key_1) ** (1/8))

        if authdate <= 1707513155:  # この基準日付も暗号化して保存することを検討
            auth_info.authorized = 0
        else:
            dt = datetime.datetime.fromtimestamp(authdate)
            if hashlib.sha256(authdate.to_bytes(32, 'big')).hexdigest() == hash_c:
                auth_info.date = dt
                auth_info.authorized = 1
                auth_info.licensekey = base64.b64encode(int(dateval).to_bytes(32, "big")).decode()
            else:
                auth_info.authorized = 0

    except Exception as e:
        print(f"認証エラー: {str(e)}")
        auth_info.authorized = 0

    if auth_info.auth_mode == 1 and auth_info.authorized == 0:
        print("無効なライセンスファイルです")
        sys.exit(1)

    return auth_info.authorized

# ライセンス生成関数（開発者用）
def generate_license(expiration_date, password):
    key_1 = int(hashlib.sha256(password.encode()).hexdigest(), 16)
    authdate = int(expiration_date.timestamp())
    dateval = str(int(authdate ** 8) ^ key_1)
    hash_c = hashlib.sha256(authdate.to_bytes(32, 'big')).hexdigest()
    license_data = f"{dateval},{hash_c}"
    return encrypt_license(license_data, password)

warnings.simplefilter('ignore')

def griffin_lim(mag_spec, n_iter=6, n_fft=3073, hop_length=1537):
    phase_spec = np.exp(1j * np.random.uniform(-np.pi, np.pi, mag_spec.shape))
    for _ in range(n_iter):
        wav = librosa.istft(mag_spec * phase_spec, n_fft=n_fft, hop_length=hop_length) / 2
        _, phase_spec = librosa.magphase(librosa.stft(wav, n_fft=n_fft, hop_length=hop_length))
    return phase_spec

def connect_spectra_smooth(spectrum1, spectrum2, overlap_size=16):
    if overlap_size > min(len(spectrum1), len(spectrum2)):
        raise ValueError("too big overlap_size")

    overlap1 = spectrum1[-overlap_size:]
    overlap2 = spectrum2[:overlap_size]
    level_diff = np.mean(1 + spectrum1) / np.mean(1 + spectrum2)
    adjusted_spectrum2 = spectrum2 * (level_diff)

    fade_out = np.linspace(1, 0, overlap_size)
    fade_in = np.linspace(0, 1, overlap_size)
    crossfaded = overlap1 * fade_out + adjusted_spectrum2[:overlap_size] * fade_in

    result = np.concatenate([
        spectrum1[:-overlap_size//2],
        crossfaded,
        adjusted_spectrum2[overlap_size//2:]
    ])

    return result

def flatten_spectrum(signal, window_size=6):
    padded_signal = np.pad(signal, (window_size//2, window_size//2), mode='edge')
    smoothed_signal = scipy.signal.fftconvolve(scipy.signal.hilbert(padded_signal).real, np.ones(window_size)/window_size, mode='valid')
    return smoothed_signal[:len(signal)]

def griffin_lim_rt(mag_spec, n_iter=1, n_fft=FFTSIZE, hop_length=HOPSIZE):
    phase_spec = np.exp(1j * np.random.uniform(-np.pi, np.pi, mag_spec.shape))
    return phase_spec

def remaster(dat, fs, scale):
    mid = (dat[:,0] + dat[:,1]) / 2
    side = (dat[:,0] - dat[:,1]) / 2
    mid_ffted = librosa.stft(mid, n_fft=FFTSIZE, hop_length=HOPSIZE)
    side_ffted = librosa.stft(side, n_fft=FFTSIZE, hop_length=HOPSIZE)
    mid_proc = np.zeros([mid_ffted.shape[0]*scale, mid_ffted.shape[1]], dtype=np.complex128)
    side_proc = np.zeros([mid_ffted.shape[0]*scale, mid_ffted.shape[1]], dtype=np.complex128)

    for i in tqdm.tqdm(range(mid_ffted.shape[1])):
        sample_mid = np.hstack([mid_ffted[:,i], np.zeros(len(mid_ffted[:,i])*(scale-1))])
        sample_side = np.hstack([side_ffted[:,i], np.zeros(len(side_ffted[:,i])*(scale-1))])
        db_mid = librosa.amplitude_to_db(sample_mid)
        db_side = librosa.amplitude_to_db(sample_side)

        db_mid[0] = db_side[0] = -18

        mid_proc[:,i] = librosa.db_to_amplitude(db_mid) * np.exp(1.j * np.angle(sample_mid))
        side_proc[:,i] = librosa.db_to_amplitude(db_side) * np.exp(1.j * np.angle(sample_side))

    mid = librosa.istft(mid_proc, n_fft=FFTSIZE*scale, hop_length=HOPSIZE*scale)
    side = librosa.istft(side_proc, n_fft=FFTSIZE*scale, hop_length=HOPSIZE*scale)
    return np.array([mid + side, mid - side]).T * scale

def proc_for_compressed(mid, side):
    mid_db = librosa.amplitude_to_db(mid)
    side_db = librosa.amplitude_to_db(side)
    mid, mid_phs = librosa.magphase(mid)
    side, side_phs = librosa.magphase(side)

    for i in tqdm.tqdm(range(mid.shape[1])):
        if np.mean(np.abs(mid_db[:,i])) > 80 or np.mean(np.abs(side_db[:,i])) > 80:
            continue

        mask_mid = (mid_db[:,i] < -12) & (np.min(mid_db[:,i]) > -120)
        mid[:,i][mask_mid] += (np.roll(mid[:,i], -1)[mask_mid] + np.roll(mid[:,i], 1)[mask_mid])

        mask_side = (side_db[:,i] < -12) & (np.min(side_db[:,i]) > -120)
        side[:,i][mask_side] += (np.roll(side[:,i], -1)[mask_side] + np.roll(side[:,i], 1)[mask_side])

        mask_side_high = mask_side & (np.arange(len(side[:,i])) > 192)
        side[:,i][mask_side_high] += np.abs(scipy.signal.hilbert(mid[:,i][mask_side_high] / 4))

    return mid * np.exp(1.j * np.angle(mid_phs)), side * np.exp(1.j * np.angle(side_phs))

def look_for_audio_input():
    pa = pyaudio.PyAudio()
    devices = [pa.get_device_info_by_index(i)["name"] for i in range(pa.get_device_count())]
    pa.terminate()
    return devices

def hires_playback(dev, dev2):
    fs = 48000
    devs = look_for_audio_input()
    p = pyaudio.PyAudio()
    p2 = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paFloat32, channels=2, rate=fs, input=True,
                    input_device_index=devs.index(dev), output_device_index=devs.index(dev2),
                    frames_per_buffer=8192)
    playstream = p2.open(format=pyaudio.paFloat32, channels=2, rate=fs, output=True,
                         output_device_index=devs.index(dev2), frames_per_buffer=8192)
    while True:
        d = stream.read(4096*2)
        data = np.frombuffer(d, dtype=np.float32)
        print(data)
        hf = hfp_2(data.copy(), fs)
        playstream.write(hf.astype(np.float32).tobytes())

class OVERTONE:
    def __init__(self):
        self.width = 2
        self.amplitude = 0
        self.base_freq = 0
        self.slope = []
        self.loop = 0

def hfp(dat, lowpass, fs, compressd_mode=False, use_hpss=True, temporal_filter_size=5,
        griffin_lim_iter=2):
    # STFTパラメータ（サンプルレートに合わせウィンドウサイズを調整）
    fft_size = 3072
    hop_length = fft_size // 2

    # mid/side 信号に変換
    mid = (dat[:, 0] + dat[:, 1]) / 2
    side = (dat[:, 0] - dat[:, 1]) / 2
    sample_length = len(mid)

    # STFT実施
    mid_ffted = librosa.stft(mid, n_fft=fft_size, hop_length=hop_length)
    side_ffted = librosa.stft(side, n_fft=fft_size, hop_length=hop_length)
    mid_mag, mid_phs = librosa.magphase(mid_ffted)
    side_mag, side_phs = librosa.magphase(side_ffted)

    # --- 低域／高域境界の自動検出 ---
    if lowpass != -1:
        lowpass_fft = int((fft_size // 2 + 1) * (lowpass / (fs / 2)))
    else:
        indices = np.where(mid_mag < 0.000001)[0]
        if len(indices) > 0:
            for i in range(len(indices)):
                if indices[0] < 64:
                    continue
                else:
                    lowpass_fft = int(indices[i]) - 1
                    break
        else:
            lowpass_fft = fft_size // 2
        lowpass_fft = max(0, min(lowpass_fft, fft_size // 2))
        print("Detected lowpass Filter Index:", lowpass_fft)

    # HPSS の適用
    if use_hpss:
        mid_hm, mid_pc = librosa.decompose.hpss(mid_mag, kernel_size=31)
        side_hm, side_pc = librosa.decompose.hpss(side_mag, kernel_size=31)
    else:
        mid_hm, mid_pc = mid_mag.copy(), np.zeros_like(mid_mag)
        side_hm, side_pc = side_mag.copy(), np.zeros_like(side_mag)

    # サンプルレートに比例したスケーリング
    scale = fs / 48000.0

    # --- 各フレーム毎の再構成 ---
    for i in tqdm.tqdm(range(mid_hm.shape[1])):
        # 低域部分はそのまま、補完対象はゼロ化
        mid_ffted[lowpass_fft:, i] = 0
        side_ffted[lowpass_fft:, i] = 0

        # 各成分のコピー（低域部分をクリア）
        sample_mid_hm = mid_hm[:, i].copy()
        sample_mid_pc = mid_pc[:, i].copy()
        sample_side_hm = side_hm[:, i].copy()
        sample_side_pc = side_pc[:, i].copy()
        sample_mid_hm[lowpass_fft:] = 0
        sample_mid_pc[lowpass_fft:] = 0
        sample_side_hm[lowpass_fft:] = 0
        sample_side_pc[lowpass_fft:] = 0

        # 再構成用の配列の初期化
        rebuild_mid = np.zeros_like(sample_mid_hm)
        rebuild_noise_mid = np.zeros_like(sample_mid_pc)
        rebuild_side = np.zeros_like(sample_mid_hm)
        rebuild_noise_side = np.zeros_like(sample_mid_pc)

        # --- ピーク検出の改善 ---
        # ピーク検出時の閾値を上げ、ノイズ由来のピークを除外
        db_mid_hm = librosa.amplitude_to_db(sample_mid_hm, ref=np.max)
        peaks_mid_hm = scipy.signal.find_peaks(db_mid_hm, distance=4)[0]
        peaks_mid_hm = peaks_mid_hm[peaks_mid_hm > lowpass_fft // 2]

        db_mid_pc = librosa.amplitude_to_db(sample_mid_pc, ref=np.max)
        peaks_mid_pc = scipy.signal.find_peaks(db_mid_pc, distance=4)[0]
        peaks_mid_pc = peaks_mid_pc[peaks_mid_pc > lowpass_fft // 2]

        db_side_hm = librosa.amplitude_to_db(sample_side_hm, ref=np.max)
        peaks_side_hm = scipy.signal.find_peaks(db_side_hm, distance=4)[0]
        peaks_side_hm = peaks_side_hm[peaks_side_hm > lowpass_fft // 2]

        db_side_pc = librosa.amplitude_to_db(sample_side_pc, ref=np.max)
        peaks_side_pc = scipy.signal.find_peaks(db_side_pc, distance=4)[0]
        peaks_side_pc = peaks_side_pc[peaks_side_pc > lowpass_fft // 2]

        # --- 調波成分の除外 ---
        def remove_harmonics(peaks):
            filtered = []
            peaks = np.sort(peaks)
            for idx, f in enumerate(peaks):
                if any(abs(f - f0) < 6 for f0 in filtered):
                    continue
                max_k = int(fft_size / (2 * f))
                is_harmonic = False
                for k in range(2, max_k + 1):
                    if any(abs(f * k - other) < 6 for other in peaks):
                        is_harmonic = True
                        break
                if not is_harmonic:
                    filtered.append(f)
            return np.array(filtered)

        peaks_mid_hm = remove_harmonics(peaks_mid_hm)
        peaks_mid_pc = remove_harmonics(peaks_mid_pc)
        peaks_side_hm = remove_harmonics(peaks_side_hm)
        peaks_side_pc = remove_harmonics(peaks_side_pc)

        # --- overtone 再構成処理 ---
        def process_peaks(peaks, sample, rebuild):
            for peak in peaks:
                ot = OVERTONE()
                ot.base_freq = peak // 4
                ot.loop = (fft_size // 2) // ot.base_freq

                # 調波振幅列の抽出
                harmonics = np.array([
                    sample[ot.base_freq * l]
                    for l in range(1, ot.loop)
                    if ot.base_freq * l < len(sample)
                ])
                if len(harmonics) == 0 or harmonics[0] == 0:
                    continue

                template = scipy.signal.gaussian(len(harmonics), std=len(harmonics) / 1.3)
                slope = scipy.signal.fftconvolve(harmonics / 12, template, mode='same')
                ot.slope = slope[:ot.loop * 12]  # 未来倍音まで予測

                ot.width = 2
                for k in range(2, 3):
                    if peak - k // 2 >= 0 and peak + k // 2 < len(sample):
                        if abs(abs(sample[peak - k // 2]) - abs(sample[peak + k // 2])) < 4:
                            ot.width = k
                            break

                start_power = max(peak - ot.width // 2, 0)
                end_power = min(peak + ot.width // 2, len(sample))
                ot.power = sample[start_power:end_power]

                # 倍音再合成
                for k in range(1, ot.loop * 2 + 1):
                    start = int(ot.base_freq * k - ot.width // 2)
                    end = int(ot.base_freq * k + ot.width // 2)
                    if start < 0 or end > len(rebuild):
                        continue
                    if k < len(ot.slope):
                        rebuild[start:end] = ot.power * abs(ot.slope[k - 1])

            return rebuild

        rebuild_mid = process_peaks(peaks_mid_hm, sample_mid_hm, rebuild_mid)
        rebuild_noise_mid = process_peaks(peaks_mid_pc, sample_mid_pc, rebuild_noise_mid)
        rebuild_side = process_peaks(peaks_side_hm, sample_side_hm, rebuild_side)
        rebuild_noise_side = process_peaks(peaks_side_pc, sample_side_pc, rebuild_noise_side)

        # 補完対象は lowpass より上。下部はゼロパディング
        mid_hm[:, i] = np.concatenate([np.zeros(lowpass_fft), rebuild_mid[lowpass_fft:]])
        mid_pc[:, i] = np.concatenate([np.zeros(lowpass_fft), rebuild_noise_mid[lowpass_fft:]])
        side_hm[:, i] = np.concatenate([np.zeros(lowpass_fft), rebuild_side[lowpass_fft:]])
        side_pc[:, i] = np.concatenate([np.zeros(lowpass_fft), rebuild_noise_side[lowpass_fft:]])

        # --- 周波数方向の平滑化 ---
        # 高域側のウィンドウサイズを大きめに設定して急激な変化を抑制
        mid_hm[:, i] = flatten_spectrum(mid_hm[:, i], window_size=3) * np.random.uniform(0.15125, 1., mid_pc[:, i].shape)
        mid_pc[:, i] = flatten_spectrum(mid_pc[:, i], window_size=40) * np.random.uniform(0.15125, 1., mid_pc[:, i].shape)
        side_hm[:, i] = flatten_spectrum(side_hm[:, i], window_size=5) * np.random.uniform(0.15125, 1., mid_pc[:, i].shape)
        side_pc[:, i] = flatten_spectrum(side_pc[:, i], window_size=40) * np.random.uniform(0.15125, 1., mid_pc[:, i].shape)

    # --- 時間軸方向の平滑化 ---
    def temporal_smoothing(spec, filter_size=temporal_filter_size):
        num_bins = spec.shape[0]
        smoothed = np.zeros_like(spec)
        for b in range(num_bins):
            # 周波数が高いほど平滑化ウィンドウサイズを大きくする
            cur_filter_size = int(filter_size + (b / num_bins) * filter_size * 1)
            cur_filter_size = max(1, cur_filter_size)
            kernel = np.ones(cur_filter_size) / cur_filter_size
            smoothed[b, :] = scipy.signal.fftconvolve(spec[b, :], kernel, mode='same')
        return smoothed

    mid_hm = temporal_smoothing(mid_hm)
    side_hm = temporal_smoothing(side_hm)

    # --- 位相再構成 ---
    mid_reconstructed_mag = mid_hm + mid_pc
    side_reconstructed_mag = side_hm + side_pc

    # Griffin-Lim の反復回数を増やし、より安定した位相推定を実現
    mid_phs_reconstructed = griffin_lim(mid_reconstructed_mag, n_iter=griffin_lim_iter)
    side_phs_reconstructed = griffin_lim(side_reconstructed_mag, n_iter=griffin_lim_iter)

    # lowpass より上の位相を置換
    mid_phs[lowpass_fft:] = mid_phs_reconstructed[lowpass_fft:]
    side_phs[lowpass_fft:] = side_phs_reconstructed[lowpass_fft:]

    rebuilt_mid = mid_reconstructed_mag * np.exp(1j * np.angle(mid_phs))
    rebuilt_side = side_reconstructed_mag * np.exp(1j * np.angle(side_phs))

    # 各フレームごとにスペクトルをスムーズに接続
    for i in range(mid_ffted.shape[1]):
        mid_ffted[:, i] = connect_spectra_smooth(mid_ffted[:lowpass_fft, i], rebuilt_mid[lowpass_fft:, i] * (np.linspace(1,0.25,len(rebuilt_mid[lowpass_fft:, i])) ** 3))
        side_ffted[:, i] = connect_spectra_smooth(side_ffted[:lowpass_fft, i], rebuilt_side[lowpass_fft:, i] * (np.linspace(1,0.25,len(rebuilt_mid[lowpass_fft:, i])) ** 3))

    if compressd_mode:
        mid_ffted, side_ffted = proc_for_compressed(mid_ffted, side_ffted)

    # 逆 STFT による波形再構成
    iffted_mid = librosa.istft(mid_ffted, hop_length=hop_length)
    iffted_side = librosa.istft(side_ffted, hop_length=hop_length)

    # 長さ調整
    if len(iffted_mid) > sample_length:
        iffted_mid = iffted_mid[:sample_length]
        iffted_side = iffted_side[:sample_length]
    elif len(iffted_mid) < sample_length:
        padding = sample_length - len(iffted_mid)
        iffted_mid = np.pad(iffted_mid, (0, padding))
        iffted_side = np.pad(iffted_side, (0, padding))

    # mid/side から元のステレオ信号に再変換
    output = np.array([iffted_mid + iffted_side, iffted_mid - iffted_side]).T
    return output

class AudioProcessor(QThread):
    error_occurred = pyqtSignal(str)

    def __init__(self, input_device, output_device, settings):
        super().__init__()
        self.input_device = input_device
        self.output_device = output_device
        self.settings = settings
        self.running = False
        self.audio_queue = queue.Queue(maxsize=10)
        self.buffer_size = CHUNK_SIZE * 16  # Increased buffer size

    def run(self):
        try:
            p = pyaudio.PyAudio()
            input_stream = p.open(format=pyaudio.paFloat32,
                                  channels=2,
                                  rate=SAMPLE_RATE,
                                  input=True,
                                  input_device_index=self.input_device,
                                  frames_per_buffer=self.buffer_size)

            output_stream = p.open(format=pyaudio.paFloat32,
                                   channels=2,
                                   rate=SAMPLE_RATE,
                                   output=True,
                                   output_device_index=self.output_device,
                                   frames_per_buffer=self.buffer_size)

            self.running = True
            while self.running:
                data = input_stream.read(self.buffer_size, exception_on_overflow=False)
                audio_chunk = np.frombuffer(data, dtype=np.float32).reshape(-1, 2)

                processed_chunk = self.process_audio(audio_chunk)

                output_stream.write(processed_chunk.astype(np.float32).tobytes())

        except Exception as e:
            self.error_occurred.emit(str(e))
        finally:
            if 'input_stream' in locals():
                input_stream.stop_stream()
                input_stream.close()
            if 'output_stream' in locals():
                output_stream.stop_stream()
                output_stream.close()
            p.terminate()

    def process_audio(self, audio_chunk):
        processed_chunk = audio_chunk.copy()

        if self.settings['high_freq_comp']:
            processed_chunk = self.apply_hfc(processed_chunk)
        if self.settings['remaster']:
            processed_chunk = self.apply_remaster(processed_chunk)

        return processed_chunk - audio_chunk

    def apply_hfc(self, audio_chunk):
        return hfp(audio_chunk, self.settings.get('lowpass', 11000), SAMPLE_RATE)

    def apply_remaster(self, audio_chunk):
        return remaster(audio_chunk, SAMPLE_RATE, self.settings.get('scale', 1))

    def stop(self):
        self.running = False

def hfp_2(dat, fs):
    fft_size = FFTSIZE
    dat = np.nan_to_num(dat)
    mid = (dat[::2] + dat[1::2]) / 2
    side = (dat[::2] - dat[1::2]) / 2
    lowpass_fft = 384
    mid_ffted = librosa.stft(mid, n_fft=fft_size, hop_length=fft_size//2)
    side_ffted = librosa.stft(side, n_fft=fft_size, hop_length=fft_size//2)
    mid_mag, mid_phs = librosa.magphase(mid_ffted)
    side_mag, side_phs = librosa.magphase(side_ffted)
    mid_hm, mid_pc = librosa.decompose.hpss(mid_mag, kernel_size=31, mask=True)
    side_hm, side_pc = librosa.decompose.hpss(side_mag, kernel_size=31, mask=True)
    scale = int(fs / 48000 + 0.555555)

    for i in tqdm.tqdm(range(mid_hm.shape[1])):
        sample_mid_hm = mid_hm[:,i].copy()
        sample_mid_pc = mid_pc[:,i].copy()
        sample_side_hm = side_hm[:,i].copy()
        sample_side_pc = side_pc[:,i].copy()

        sample_mid_hm[lowpass_fft:] = 0
        sample_mid_pc[lowpass_fft:] = 0
        sample_side_hm[lowpass_fft:] = 0
        sample_side_pc[lowpass_fft:] = 0

        rebuild_mid = np.zeros_like(sample_mid_hm)
        rebuild_noise_mid = np.zeros_like(sample_mid_pc)
        rebuild_side = np.zeros_like(sample_mid_hm)
        rebuild_noise_side = np.zeros_like(sample_mid_pc)

        for sample, rebuild in [(sample_mid_hm, rebuild_mid),
                                (sample_mid_pc, rebuild_noise_mid),
                                (sample_side_hm, rebuild_side),
                                (sample_side_pc, rebuild_noise_side)]:
            peaks = scipy.signal.find_peaks(sample)[0]
            peaks = peaks[peaks > len(peaks)//2]

            for j in peaks:
                ot = OVERTONE()
                ot.base_freq = j // 2
                ot.loop = fft_size // (2 * ot.base_freq)

                harmonics = np.array([sample[ot.base_freq * l] for l in range(ot.loop) if ot.base_freq * l < len(sample)])
                ot.slope = np.fft.fft(np.fft.ifft(harmonics / harmonics[0]) + np.linspace(1, 0.33333, len(harmonics))) / (22 if 'hm' in locals() else 8)

                for k in range(2, 4, 2):
                    if j-k//2 >= 0 and j+k//2 < len(sample) and abs(abs(sample[j-k//2]) - abs(sample[j+k//2])) < 4:
                        ot.width = k
                        break

                ot.power = sample[j-ot.width//2:j+ot.width//2]

                for k in range(1, ot.loop):
                    start = int(ot.base_freq * (k//2 + 1) - ot.width//2)
                    end = int(ot.base_freq * (k//2 + 1) + ot.width//2)
                    if start < 0 or end > len(rebuild):
                        break
                    rebuild[start:end] = ot.power * abs(ot.slope[k//2])

        # Applying the rebuilt spectra
        mid_hm[:,i] = np.concatenate([np.zeros(lowpass_fft), rebuild_mid[lowpass_fft:]])
        mid_pc[:,i] = np.concatenate([np.zeros(lowpass_fft), rebuild_noise_mid[lowpass_fft:]])
        side_hm[:,i] = np.concatenate([np.zeros(lowpass_fft), rebuild_side[lowpass_fft:]])
        side_pc[:,i] = np.concatenate([np.zeros(lowpass_fft), rebuild_noise_side[lowpass_fft:]])

        fade_out = np.linspace(1, 0, len(mid_hm[:,i][lowpass_fft:]))
        mid_hm[:,i][lowpass_fft:] *= fade_out
        mid_pc[:,i][lowpass_fft:] *= fade_out
        side_hm[:,i][lowpass_fft:] *= fade_out
        side_pc[:,i][lowpass_fft:] *= fade_out

        # Level matching
        for original, rebuilt in [(sample_mid_hm, rebuild_mid),
                                  (sample_mid_pc, rebuild_noise_mid),
                                  (sample_side_hm, rebuild_side),
                                  (sample_side_pc, rebuild_noise_side)]:
            db_original = librosa.amplitude_to_db(abs(original[lowpass_fft-60:lowpass_fft]))
            db_rebuilt = librosa.amplitude_to_db(abs(rebuilt[lowpass_fft:lowpass_fft+60]))
            diff_db = np.mean(db_original) - np.mean(db_rebuilt)
            diff_db = min(diff_db, 36)
            rebuilt = librosa.db_to_amplitude(librosa.amplitude_to_db(rebuilt) + diff_db - 15 - scale*2)

        # Envelope shaping
        for spec in [mid_hm[:,i], mid_pc[:,i], side_hm[:,i], side_pc[:,i]]:
            env = np.abs(np.fft.fft(np.fft.ifft(spec) + np.linspace(0.5, -0.5, len(spec)))) / 12
            env = np.clip(env, 0, 1)
            spec *= env

    # Phase reconstruction
    mid_phs[lowpass_fft:] = griffin_lim_rt(mid_ffted)[lowpass_fft:]
    side_phs[lowpass_fft:] = griffin_lim_rt(side_ffted)[lowpass_fft:]

    rebuilt_mid = (mid_hm + mid_pc) * np.exp(1.j * mid_phs)
    rebuilt_side = (side_hm + side_pc) * np.exp(1.j * side_phs)

    # Replace very low amplitude parts
    low_amp_mask = librosa.amplitude_to_db(mid_ffted) < -80
    mid_ffted[low_amp_mask] = rebuilt_mid[low_amp_mask]
    side_ffted[low_amp_mask] = rebuilt_side[low_amp_mask]

    mid_ffted[:lowpass_fft] = 0
    side_ffted[:lowpass_fft] = 0

    iffted_mid = librosa.istft(mid_ffted, hop_length=fft_size//2)
    iffted_side = librosa.istft(side_ffted, hop_length=fft_size//2)

    return np.array([iffted_mid+iffted_side, iffted_mid-iffted_side]).T

def get_score(original_file, processed_file, sr, lowpass):
    original_y = (original_file[:,0] + original_file[:,1]) / 2
    processed_y = (processed_file[:,0] + processed_file[:,1]) / 2
    lowpass_fft = int((1024//2+1) * (lowpass/(sr/2)))

    # Ensure same length
    min_length = min(len(original_y), len(processed_y))
    original_y = original_y[:min_length]
    processed_y = processed_y[:min_length]

    original_stft = librosa.stft(original_y)
    processed_stft = librosa.stft(processed_y)

    score = 1
    for i in range(original_stft.shape[1]):
        or_ = original_stft[:,i]
        cv = processed_stft[:,i]
        score = (score + np.corrcoef(or_.real, cv.real)[0][1]) / 2

    return score

def decode(dat, bit):
    bit *= -1
    mid = (dat[:,0] + dat[:,1]) / 2
    side = (dat[:,0] - dat[:,1]) / 2

    for channel in [mid, side]:
        ffted = librosa.stft(channel, n_fft=1024)
        for i in tqdm.tqdm(range(ffted.shape[1])):
            mag, phs = librosa.magphase(ffted[:,i])
            db = librosa.amplitude_to_db(mag)
            avg = int(np.mean(db)) - 6

            mask = db < bit
            db[mask] = -120

            smooth_mask = (db == -120) & (np.roll(db, -1) != -120) & (np.roll(db, 1) != -120)
            db[smooth_mask] = (np.roll(db, -1)[smooth_mask] + np.roll(db, 1)[smooth_mask]) / 2

            random_mask = db == -120
            db[random_mask] = np.random.randint(avg-15, avg-8, size=np.sum(random_mask))

            idb = librosa.db_to_amplitude(db)
            idb[-400:] *= np.linspace(1, 0, 400)
            ffted[:,i] = idb * np.exp(1.j * np.angle(phs))

        channel[:] = librosa.istft(ffted, n_fft=1024)

    return np.array([mid+side, mid-side]).T

class AudioConversionThread(QThread):
    progress_updated = pyqtSignal(int, int)
    conversion_complete = pyqtSignal()
    file_converted = pyqtSignal(str)

    def __init__(self, files, settings):
        super().__init__()
        self.files = files
        self.settings = settings

    def run(self):
        total_files = len(self.files)
        for i, file in enumerate(self.files):
            self.convert_file(file)
            self.file_converted.emit(file)
            overall_progress = int((i + 1) / total_files * 100)
            self.progress_updated.emit(100, overall_progress)
        self.conversion_complete.emit()

    def convert_file(self, file):
        dat, fs = sf.read(file)
        dat = dat[:,:48000*30]

        if self.settings['noise_reduction']:
            dat = decode(dat, 120)

        if self.settings['remaster']:
            dat = remaster(dat, fs, self.settings['scale'])
        elif self.settings['scale'] != 1:
            dat = remaster(dat, fs, self.settings['scale'])

        if self.settings['high_freq_comp']:
            dat = hfp(dat, self.settings['lowpass'], fs * self.settings['scale'],
                      compressd_mode=self.settings['compressed_mode'])

        output_file = f"{os.path.splitext(file)[0]}_converted.wav"
        sf.write(output_file, dat, fs * self.settings['scale'],
                 format="WAV", subtype=f"PCM_{self.settings['bit_depth']}")

        for progress in range(101):
            self.progress_updated.emit(progress, 0)
            self.msleep(10)

class HRAudioWizard(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.audio_processor = None

    def initUI(self):
        self.setWindowTitle('HRAudioWizard')
        self.setGeometry(100, 100, 800, 600)

        main_widget = QWidget()
        main_layout = QVBoxLayout()
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)

        # Create tabs
        tabs = QTabWidget()
        main_layout.addWidget(tabs)

        # Offline processing tab
        offline_tab = QWidget()
        offline_layout = QVBoxLayout()
        offline_tab.setLayout(offline_layout)
        tabs.addTab(offline_tab, "Offline Processing")

        # File selection
        file_group = QGroupBox("File Selection")
        file_layout = QVBoxLayout()
        self.file_list = QListWidget()

        file_buttons_layout = QHBoxLayout()
        add_file_btn = QPushButton("Add Files")
        add_file_btn.clicked.connect(self.add_files)
        clear_files_btn = QPushButton("Clear Files")
        clear_files_btn.clicked.connect(self.clear_files)

        file_buttons_layout.addWidget(add_file_btn)
        file_buttons_layout.addWidget(clear_files_btn)

        file_layout.addWidget(self.file_list)
        file_layout.addLayout(file_buttons_layout)
        file_group.setLayout(file_layout)
        offline_layout.addWidget(file_group)

        # Conversion settings
        settings_group = QGroupBox("Conversion Settings")
        settings_layout = QVBoxLayout()

        # Bit depth
        bit_depth_layout = QHBoxLayout()
        bit_depth_layout.addWidget(QLabel("Bit Depth:"))
        self.bit_depth_24 = QRadioButton("24-bit")
        self.bit_depth_32 = QRadioButton("32-bit")
        self.bit_depth_64 = QRadioButton("64-bit")
        self.bit_depth_24.setChecked(True)
        bit_depth_layout.addWidget(self.bit_depth_24)
        bit_depth_layout.addWidget(self.bit_depth_32)
        bit_depth_layout.addWidget(self.bit_depth_64)
        settings_layout.addLayout(bit_depth_layout)

        # Sampling rate
        sampling_rate_layout = QHBoxLayout()
        sampling_rate_layout.addWidget(QLabel("Sampling Rate:"))
        self.sampling_rate = QComboBox()
        self.sampling_rate.addItems(["x1", "x2", "x4", "x8"])
        sampling_rate_layout.addWidget(self.sampling_rate)
        settings_layout.addLayout(sampling_rate_layout)

        # Checkboxes
        self.remaster_cb = QCheckBox("Remaster")
        self.noise_reduction_cb = QCheckBox("Noise Reduction")
        self.high_freq_comp_cb = QCheckBox("High Frequency Compensation")
        self.compressed_mode_cb = QCheckBox("Compressed Source Mode")
        settings_layout.addWidget(self.remaster_cb)
        settings_layout.addWidget(self.noise_reduction_cb)
        settings_layout.addWidget(self.high_freq_comp_cb)
        settings_layout.addWidget(self.compressed_mode_cb)

        # Lowpass filter
        lowpass_layout = QHBoxLayout()
        lowpass_layout.addWidget(QLabel("Lowpass Filter (Hz):"))
        self.lowpass_filter = QSpinBox()
        self.lowpass_filter.setRange(-1, 50000)
        self.lowpass_filter.setSingleStep(1000)
        self.lowpass_filter.setValue(16000)
        lowpass_layout.addWidget(self.lowpass_filter)
        settings_layout.addLayout(lowpass_layout)

        settings_group.setLayout(settings_layout)
        offline_layout.addWidget(settings_group)

        # Conversion progress
        progress_layout = QVBoxLayout()
        self.current_file_label = QLabel("Current File: ")
        progress_layout.addWidget(self.current_file_label)
        self.file_progress_bar = QProgressBar()
        progress_layout.addWidget(self.file_progress_bar)
        self.overall_progress_bar = QProgressBar()
        progress_layout.addWidget(self.overall_progress_bar)
        offline_layout.addLayout(progress_layout)

        # Conversion button
        self.convert_btn = QPushButton("Start Conversion")
        self.convert_btn.clicked.connect(self.start_conversion)
        offline_layout.addWidget(self.convert_btn)

        # Realtime processing tab
        realtime_tab = QWidget()
        realtime_layout = QVBoxLayout()
        realtime_tab.setLayout(realtime_layout)
        tabs.addTab(realtime_tab, "Realtime Processing")

        # Device selection and reload button
        device_layout = QHBoxLayout()

        # Input device selection
        input_layout = QVBoxLayout()
        input_layout.addWidget(QLabel("Input Device:"))
        self.input_device_combo = QComboBox()
        input_layout.addWidget(self.input_device_combo)
        device_layout.addLayout(input_layout)

        # Output device selection
        output_layout = QVBoxLayout()
        output_layout.addWidget(QLabel("Output Device:"))
        self.output_device_combo = QComboBox()
        output_layout.addWidget(self.output_device_combo)
        device_layout.addLayout(output_layout)

        # Reload devices button
        self.reload_devices_btn = QPushButton("Reload Devices")
        self.reload_devices_btn.clicked.connect(self.reload_audio_devices)
        device_layout.addWidget(self.reload_devices_btn)

        realtime_layout.addLayout(device_layout)

        # Realtime processing options
        self.realtime_hfc_checkbox = QCheckBox("High Frequency Compensation")
        self.realtime_hfc_checkbox.stateChanged.connect(self.toggle_hfc_settings)
        realtime_layout.addWidget(self.realtime_hfc_checkbox)

        # HFC Lowpass filter
        self.realtime_hfc_layout = QHBoxLayout()
        self.realtime_hfc_layout.addWidget(QLabel("HFC Lowpass Filter (Hz):"))
        self.realtime_lowpass_filter = QSpinBox()
        self.realtime_lowpass_filter.setRange(6000, 50000)
        self.realtime_lowpass_filter.setSingleStep(1000)
        self.realtime_lowpass_filter.setValue(16000)
        self.realtime_hfc_layout.addWidget(self.realtime_lowpass_filter)
        realtime_layout.addLayout(self.realtime_hfc_layout)

        self.realtime_remaster_checkbox = QCheckBox("Remaster")
        realtime_layout.addWidget(self.realtime_remaster_checkbox)

        # Start/Stop button
        self.toggle_button = QPushButton("Start Processing")
        self.toggle_button.clicked.connect(self.toggle_processing)
        realtime_layout.addWidget(self.toggle_button)

        # Auth info button
        auth_info_btn = QPushButton("Show Auth Info")
        auth_info_btn.clicked.connect(self.show_auth_info)
        main_layout.addWidget(auth_info_btn)

        self.populate_audio_devices()

    def add_files(self):
        files, _ = QFileDialog.getOpenFileNames(self, "Select audio files", "", "WAV Files (*.wav)")
        self.file_list.addItems(files)

    def clear_files(self):
        self.file_list.clear()

    def start_conversion(self):
        files = [self.file_list.item(i).text() for i in range(self.file_list.count())]
        if not files:
            QMessageBox.warning(self, "No Files", "Please add files to convert.")
            return

        settings = {
            'bit_depth': 24 if self.bit_depth_24.isChecked() else (32 if self.bit_depth_32.isChecked() else 64),
            'scale': int(self.sampling_rate.currentText()[1:]),
            'remaster': self.remaster_cb.isChecked(),
            'noise_reduction': self.noise_reduction_cb.isChecked(),
            'high_freq_comp': self.high_freq_comp_cb.isChecked(),
            'compressed_mode': self.compressed_mode_cb.isChecked(),
            'lowpass': self.lowpass_filter.value()
        }

        self.conversion_thread = AudioConversionThread(files, settings)
        self.conversion_thread.progress_updated.connect(self.update_progress)
        self.conversion_thread.conversion_complete.connect(self.conversion_finished)
        self.conversion_thread.file_converted.connect(self.file_converted)

        # Disable UI elements
        self.convert_btn.setEnabled(False)
        self.file_list.setEnabled(False)
        self.set_settings_enabled(False)

        self.conversion_thread.start()

    def update_progress(self, file_progress, overall_progress):
        self.file_progress_bar.setValue(file_progress)
        self.overall_progress_bar.setValue(overall_progress)

    def file_converted(self, file):
        self.current_file_label.setText(f"Current File: {os.path.basename(file)}")
        self.file_list.takeItem(0)

    def conversion_finished(self):
        self.file_progress_bar.setValue(100)
        self.overall_progress_bar.setValue(100)
        self.current_file_label.setText("Conversion Complete")

        # Re-enable UI elements
        self.convert_btn.setEnabled(True)
        self.file_list.setEnabled(True)
        self.set_settings_enabled(True)

        QMessageBox.information(self, "Conversion Complete", "All files have been converted successfully.")

    def set_settings_enabled(self, enabled):
        self.bit_depth_24.setEnabled(enabled)
        self.bit_depth_32.setEnabled(enabled)
        self.bit_depth_64.setEnabled(enabled)
        self.sampling_rate.setEnabled(enabled)
        self.remaster_cb.setEnabled(enabled)
        self.noise_reduction_cb.setEnabled(enabled)
        self.high_freq_comp_cb.setEnabled(enabled)
        self.compressed_mode_cb.setEnabled(enabled)
        self.lowpass_filter.setEnabled(enabled)

    def populate_audio_devices(self):
        p = pyaudio.PyAudio()

        # Store current selections
        current_input = self.input_device_combo.currentData()
        current_output = self.output_device_combo.currentData()

        # Clear existing items
        self.input_device_combo.clear()
        self.output_device_combo.clear()

        # Populate input devices
        for i in range(p.get_device_count()):
            dev_info = p.get_device_info_by_index(i)
            if dev_info['maxInputChannels'] > 0:
                self.input_device_combo.addItem(dev_info['name'], i)

        # Populate output devices
        for i in range(p.get_device_count()):
            dev_info = p.get_device_info_by_index(i)
            if dev_info['maxOutputChannels'] > 0:
                self.output_device_combo.addItem(dev_info['name'], i)

        p.terminate()

        # Restore previous selections if still available
        input_index = self.input_device_combo.findData(current_input)
        if input_index >= 0:
            self.input_device_combo.setCurrentIndex(input_index)

        output_index = self.output_device_combo.findData(current_output)
        if output_index >= 0:
            self.output_device_combo.setCurrentIndex(output_index)

        # Enable/disable combos based on available devices
        self.input_device_combo.setEnabled(self.input_device_combo.count() > 0)
        self.output_device_combo.setEnabled(self.output_device_combo.count() > 0)

    def reload_audio_devices(self):
        self.populate_audio_devices()
        QMessageBox.information(self, "Devices Reloaded", "Audio devices have been reloaded.")

    def toggle_hfc_settings(self, state):
        self.realtime_lowpass_filter.setEnabled(state == Qt.CheckState.Checked)

    def toggle_processing(self):
        if self.audio_processor is None or not self.audio_processor.running:
            self.start_processing()
        else:
            self.stop_processing()

    def start_processing(self):
        input_device = self.input_device_combo.currentData()
        output_device = self.output_device_combo.currentData()
        settings = {
            'high_freq_comp': self.realtime_hfc_checkbox.isChecked(),
            'remaster': self.realtime_remaster_checkbox.isChecked(),
            'lowpass': self.realtime_lowpass_filter.value()
        }
        self.audio_processor = AudioProcessor(input_device, output_device, settings)
        self.audio_processor.error_occurred.connect(self.handle_error)
        self.audio_processor.start()
        self.toggle_button.setText("Stop Processing")

    def stop_processing(self):
        if self.audio_processor:
            self.audio_processor.stop()
            self.audio_processor.wait()
            self.audio_processor = None
        self.toggle_button.setText("Start Processing")

    def handle_error(self, error_msg):
        QMessageBox.critical(self, "Error", f"An error occurred: {error_msg}")
        self.stop_processing()

    def show_auth_info(self):
        info = ""
        if auth_info.authorized == 1:
            info = (f"Using the full version\n"
                    f"Serial Number: {auth_info.licensekey}\n"
                    f"Purchase Time: {auth_info.date.strftime('%Y/%m/%d %H:%M:%S')}")
        elif auth_info.authorized == -1:
            info = "Using the unlimited free version\nPlease visit the author's website to purchase."
        elif auth_info.authorized == 0:
            info = "Using the free trial version\nPlease visit the author's website to purchase."

        QMessageBox.information(self, "Authorization Information", info)

    def closeEvent(self, event):
        self.stop_processing()
        event.accept()

class AudioConversionThread(QThread):
    progress_updated = pyqtSignal(int, int)
    conversion_complete = pyqtSignal()
    file_converted = pyqtSignal(str)

    def __init__(self, files, settings):
        super().__init__()
        self.files = files
        self.settings = settings

    def run(self):
        total_files = len(self.files)
        for i, file in enumerate(self.files):
            self.convert_file(file)
            self.file_converted.emit(file)
            overall_progress = int((i + 1) / total_files * 100)
            self.progress_updated.emit(100, overall_progress)
        self.conversion_complete.emit()

    def convert_file(self, file):
        dat, fs = sf.read(file)

        if self.settings['noise_reduction']:
            dat = decode(dat, 120)

        if self.settings['remaster']:
            dat = remaster(dat, fs, self.settings['scale'])
        elif self.settings['scale'] != 1:
            dat = remaster(dat, fs, self.settings['scale'])

        if self.settings['high_freq_comp']:
            dat = hfp(dat, self.settings['lowpass'], fs * self.settings['scale'],
                      compressd_mode=self.settings['compressed_mode'])

        output_file = f"{os.path.splitext(file)[0]}_converted.wav"
        sf.write(output_file, dat, fs * self.settings['scale'],
                 format="WAV", subtype=f"PCM_{self.settings['bit_depth']}")

        for progress in range(101):
            self.progress_updated.emit(progress, 0)
            self.msleep(10)

# ライセンス生成関数（開発者用）
def generate_license(expiration_date, password):
    key_1 = int(hashlib.sha256(password.encode()).hexdigest(), 16)
    authdate = int(expiration_date.timestamp())
    dateval = str(int(authdate ** 8) ^ key_1)
    hash_c = hashlib.sha256(authdate.to_bytes(32, 'big')).hexdigest()
    license_data = f"{dateval},{hash_c}"
    return encrypt_license(license_data, password)

if __name__ == '__main__':
    app = QApplication(sys.argv)

    license_path = os.environ.get('LICENSE_PATH', os.path.join(os.path.dirname(sys.argv[0]), "license.lc_hraw"))
    auth_result = auth(license_path)

    if auth_result == 1:
        print("認証成功")
    elif auth_result == 0:
        print("認証失敗")
    else:
        print("無制限フリー版を使用中")

    try:
        # Create and show the main window
        ex = HRAudioWizard()
        ex.show()
        sys.exit(app.exec())
    except:
        pass
