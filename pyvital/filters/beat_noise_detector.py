import numpy as np
import vitaldb
import os
import torch
import pyvital.arr
import torch.nn.functional as F
import torch.nn as nn
from scipy.signal import butter, filtfilt, iirnotch, find_peaks
import warnings
warnings.filterwarnings('ignore')

# ================================================================
# 1. 모델 및 유틸리티 함수 정의
# ================================================================

class UniMSNet(nn.Module):
    def __init__(self, input_length=256, num_classes=5):
        super().__init__()
        self.small_conv = nn.Conv1d(1, 16, kernel_size=5, stride=1, padding=2)
        self.medium_conv = nn.Conv1d(1, 8, kernel_size=15, stride=1, padding=7)
        self.large_conv = nn.Conv1d(1, 8, kernel_size=31, stride=1, padding=15)
        combined_channels = 32
        self.conv11 = nn.Conv1d(combined_channels, combined_channels, kernel_size=5, stride=1, padding='same')
        self.conv12 = nn.Conv1d(combined_channels, combined_channels, kernel_size=5, stride=1, padding='same')
        self.pool1 = nn.MaxPool1d(kernel_size=5, stride=2)
        self.conv21 = nn.Conv1d(combined_channels, 48, kernel_size=5, stride=1, padding='same')
        self.conv22 = nn.Conv1d(48, 48, kernel_size=5, stride=1, padding='same')
        self.shortcut2 = nn.Conv1d(combined_channels, 48, kernel_size=1)
        self.pool2 = nn.MaxPool1d(kernel_size=5, stride=2)
        self.conv31 = nn.Conv1d(48, 64, kernel_size=5, stride=1, padding='same')
        self.conv32 = nn.Conv1d(64, 64, kernel_size=5, stride=1, padding='same')
        self.shortcut3 = nn.Conv1d(48, 64, kernel_size=1)
        self.pool3 = nn.MaxPool1d(kernel_size=5, stride=2)
        self.conv41 = nn.Conv1d(64, 96, kernel_size=3, stride=1, padding='same')
        self.conv42 = nn.Conv1d(96, 96, kernel_size=3, stride=1, padding='same')
        self.shortcut4 = nn.Conv1d(64, 96, kernel_size=1)
        self.pool4 = nn.MaxPool1d(kernel_size=5, stride=2)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(96, 64)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        elif x.dim() == 3 and x.size(1) == 2:
            x = x[:, 0, :].unsqueeze(1)
        
        x_small = F.relu(self.small_conv(x))
        x_medium = F.relu(self.medium_conv(x))
        x_large = F.relu(self.large_conv(x))
        
        x = torch.cat([x_small, x_medium, x_large], dim=1)
        
        identity = x
        x = F.relu(self.conv11(x))
        x = self.conv12(x)
        x += identity
        x = F.relu(x)
        x = self.pool1(x)
        
        identity = self.shortcut2(x)
        x = F.relu(self.conv21(x))
        x = self.conv22(x)
        x += identity
        x = F.relu(x)
        x = self.pool2(x)
        
        identity = self.shortcut3(x)
        x = F.relu(self.conv31(x))
        x = self.conv32(x)
        x += identity
        x = F.relu(x)
        x = self.pool3(x)
        
        identity = self.shortcut4(x)
        x = F.relu(self.conv41(x))
        x = self.conv42(x)
        x += identity
        x = F.relu(x)
        x = self.pool4(x)
        
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def filter_signal(signal, fs=100, cutoff=30, notch_freq=50, q=30, order=4):
    """신호 필터링 함수"""
    nyq = 0.5 * fs
    if notch_freq > 0 and notch_freq < nyq:
        b_notch, a_notch = iirnotch(notch_freq, q, fs)
        signal = filtfilt(b_notch, a_notch, signal)
    
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, signal)

def process_peaks(segment_data, srate=100):
    """세그먼트 내 R-peak 검출 함수"""
    max_amp = np.max(np.abs(segment_data))
    scipy_peaks, _ = find_peaks(
        segment_data,
        distance=int(0.15 * srate),
    )
    return scipy_peaks

# ================================================================
# 2. 노이즈 탐지 규칙 헬퍼 함수
# (ECGAnalyzer 클래스에서 가져와 독립 함수로 수정)
# ================================================================

def merge_intervals(intervals):
    if not intervals: return []
    intervals.sort(key=lambda x: x[0])
    merged = [intervals[0]]
    for current in intervals[1:]:
        last = merged[-1]
        if current[0] <= last[1]:
            last[1] = max(last[1], current[1])
        else:
            merged.append(current)
    return merged

def calculate_noise_score(segment_rr, freq_threshold=0.3):
    if len(segment_rr) < 2 or np.ptp(segment_rr) < 1e-8: return 0.0
    norm_seg = (segment_rr - np.min(segment_rr)) / np.ptp(segment_rr)
    grad = np.diff(norm_seg)
    if len(grad) == 0: return 0.0
    magnitude_score = np.max(np.abs(grad))
    frequency_score = np.sum(np.abs(grad) > freq_threshold)
    return magnitude_score * (frequency_score + 1)

def define_noise_from_u_beat(ecg_chunk, r_peaks, predictions, class_map, srate):
    rr_scores = []
    for i in range(len(r_peaks) - 1):
        start, end = r_peaks[i], r_peaks[i+1]
        score = calculate_noise_score(ecg_chunk[start:end])
        rr_scores.append({'start_peak': start, 'end_peak': end, 'score': score})

    n_code, u_code = class_map['N'], class_map['U']
    n_n_scores = [item['score'] for item in rr_scores if predictions.get(item['start_peak']) == n_code and predictions.get(item['end_peak']) == n_code]
    if not n_n_scores: return []
    
    noise_threshold = np.mean(n_n_scores) * 1.5
    noisy_u_peaks = set()
    for item in rr_scores:
        start_pred, end_pred = predictions.get(item['start_peak']), predictions.get(item['end_peak'])
        if u_code in [start_pred, end_pred] and item['score'] > noise_threshold:
            if start_pred == u_code: noisy_u_peaks.add(item['start_peak'])
            if end_pred == u_code: noisy_u_peaks.add(item['end_peak'])
    
    return [[max(0, p - srate), min(len(ecg_chunk), p + srate)] for p in noisy_u_peaks]

def detect_isoelectric(ecg_chunk, srate, threshold_mv=0.05, min_dur_sec=0.5):
    min_len = int(min_dur_sec * srate)
    padded_sections = []
    is_flat = np.zeros(len(ecg_chunk), dtype=bool)
    for i in range(len(ecg_chunk) - min_len):
        window = ecg_chunk[i : i + min_len]
        if (np.max(window) - np.min(window)) < threshold_mv:
            is_flat[i:i+min_len] = True

    in_section, start_idx = False, 0
    for i, flat in enumerate(is_flat):
        if flat and not in_section:
            in_section, start_idx = True, i
        elif not flat and in_section:
            in_section = False
            if (i - start_idx) >= min_len:
                padded_sections.append([max(0, start_idx - srate), min(len(ecg_chunk), i + srate)])
    if in_section and (len(ecg_chunk) - start_idx) >= min_len:
        padded_sections.append([max(0, start_idx - srate), len(ecg_chunk)])
    return padded_sections

def detect_irregular_u(ecg_chunk, predictions, class_map, srate, std_thresh=10, min_seq=2):
    u_code, p_code, s_code = class_map['U'], class_map['P'], class_map['S']
    noise_sections = []
    
    u_sequences, current_sequence = [], []
    for peak in sorted(predictions.keys()):
        pred = predictions.get(peak)
        if pred == u_code: current_sequence.append(peak)
        elif pred != p_code:
            if len(current_sequence) >= min_seq: u_sequences.append(current_sequence)
            current_sequence = []
    if len(current_sequence) >= min_seq: u_sequences.append(current_sequence)

    for seq in u_sequences:
        if len(seq) > 1 and np.std(np.diff(seq)) > std_thresh:
            start_peak, end_peak = seq[0], seq[-1]
            padded_start = max(0, start_peak - 2 * srate)
            padded_end = min(len(ecg_chunk), end_peak + 2 * srate)
            
            s_beat_present = any(pred == s_code and padded_start <= p <= padded_end for p, pred in predictions.items())
            if not s_beat_present:
                noise_sections.append([padded_start, padded_end])
    return noise_sections

def detect_steep_slope(ecg_chunk, predictions, srate, slope_thresh_mv=0.5, pad_sec=0.5, peak_exclude_sec=0.15):
    analysis_mask = np.ones(len(ecg_chunk), dtype=bool)
    exclude_samples = int(peak_exclude_sec * srate)
    for peak in predictions.keys():
        start = max(0, peak - exclude_samples)
        end = min(len(ecg_chunk), peak + exclude_samples)
        analysis_mask[start:end] = False
    
    gradient = np.diff(ecg_chunk)
    steep_indices = np.where(np.abs(gradient) > slope_thresh_mv)[0]
    valid_steep_points = [i for i in steep_indices if analysis_mask[i]]
    if not valid_steep_points: return []

    pad_samples = int(pad_sec * srate)
    intervals = [[max(0, p - pad_samples), min(len(ecg_chunk), p + pad_samples)] for p in set(valid_steep_points)]
    return intervals

def detect_consecutive_p(ecg_chunk, predictions, class_map, srate, min_p_seq=4, pad_sec=1.0):
    n_count = list(predictions.values()).count(class_map['N'])
    s_count = list(predictions.values()).count(class_map['S'])
    if s_count > n_count: return []

    noise_sections, current_p_seq = [], []
    p_code = class_map['P']
    for peak in sorted(predictions.keys()):
        if predictions.get(peak) == p_code:
            current_p_seq.append(peak)
        else:
            if len(current_p_seq) >= min_p_seq:
                pad_samples = int(pad_sec * srate)
                start = max(0, current_p_seq[0] - pad_samples)
                end = min(len(ecg_chunk), current_p_seq[-1] + pad_samples)
                noise_sections.append([start, end])
            current_p_seq = []
    if len(current_p_seq) >= min_p_seq:
        pad_samples = int(pad_sec * srate)
        start = max(0, current_p_seq[0] - pad_samples)
        end = min(len(ecg_chunk), current_p_seq[-1] + pad_samples)
        noise_sections.append([start, end])
    return noise_sections

def detect_amp_variation(ecg_chunk, predictions, class_map, srate, ratio=10.0, baseline_hz=0.5):
    nyq = 0.5 * srate
    b, a = butter(1, baseline_hz / nyq, btype='high', analog=False)
    corrected_ecg = filtfilt(b, a, ecg_chunk)
    
    r_codes = {class_map['N'], class_map['S'], class_map['V']}
    heights = [corrected_ecg[p] for p, pred in predictions.items() if pred in r_codes]
    if len(heights) < 2: return []
    
    min_h, max_h = np.min(heights), np.max(heights)
    if min_h > 1e-6 and (max_h / min_h) > ratio:
        return [[0, len(ecg_chunk)]]
    return []


# ================================================================
# 3. 메인 실행 함수 (run)
# ================================================================

# 모델을 한 번만 로드하기 위한 전역 변수
model_beat = None

def run(inp, opt, cfg):
    global model_beat

    # --- 초기 설정 ---
    trk_name = list(inp.keys())[0]
    if 'srate' not in inp[trk_name]:
        return [[]]

    data = inp[trk_name]['vals']
    srate_orig = inp[trk_name]['srate']
    srate = 100
    if srate_orig != srate:
        data = pyvital.arr.resample_hz(data, srate_orig, srate)

    # --- 모델 로드 ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if model_beat is None:
        print("Loading Beat Classification Model...")
        model_beat = UniMSNet(input_length=200, num_classes=5).to(device)
        modelname = './UniMSNet.pth' # 모델 가중치 파일 경로
        if not os.path.exists(modelname):
            raise FileNotFoundError(f"Model weight file not found: {modelname}")
        model_beat.load_state_dict(torch.load(modelname, map_location=device))
        model_beat.eval()

    # --- 1. 비트 분류 (Beat Classification) ---
    all_peaks = []
    segment_len = int(10 * srate)
    step = int(9 * srate) # 1초 오버랩
    for seg_start in range(0, len(data), step):
        seg_end = min(seg_start + segment_len, len(data))
        data_segment = data[seg_start:seg_end]
        if len(data_segment) < srate * 0.5: continue
        
        segment_peaks = process_peaks(data_segment, srate)
        all_peaks.extend([p + seg_start for p in segment_peaks])
    
    peaks = sorted(list(set(all_peaks)))

    beat_signals, valid_peaks = [], []
    for peak in peaks:
        beat_start = max(0, peak - srate)
        beat_end = min(len(data), peak + srate)
        if beat_end - beat_start == srate * 2:
            beat_signal = data[beat_start:beat_end]
            filtered_beat = filter_signal(beat_signal, fs=srate, cutoff=20)
            if np.ptp(filtered_beat) > 1e-8:
                normalized_beat = (filtered_beat - np.min(filtered_beat)) / np.ptp(filtered_beat)
                beat_signals.append(normalized_beat)
                valid_peaks.append(peak)

    out_bstr = []
    predictions_dict = {}
    class_names = {0: 'N', 1: 'S', 2: 'V', 3: 'U', 4: 'P'}

    if beat_signals:
        beat_tensors = torch.FloatTensor(np.array(beat_signals)).unsqueeze(1).to(device)
        with torch.no_grad():
            outputs = model_beat(beat_tensors)
            predictions = torch.argmax(outputs, dim=1).cpu().numpy()

            for peak, pred_id in zip(valid_peaks, predictions):
                predictions_dict[peak] = pred_id
                label = class_names.get(pred_id)
                if label and label != 'P': # P-beat는 결과에 저장하지 않음
                    out_bstr.append({'dt': peak / srate, 'val': label})

    # --- 2. 노이즈 구간 분석 (Noise Detection) ---
    all_noise_intervals = []
    class_map = {v: k for k, v in class_names.items()}
    
    # 최종 R-peak 정의 (P-beat, Filtered-U 제외) - 노이즈 분석에 사용
    # 여기서는 단순화를 위해 모든 non-P peak를 r_peaks로 간주
    r_peaks = sorted([p for p, pred in predictions_dict.items() if class_names.get(pred) != 'P'])

    if r_peaks:
        # 각 노이즈 규칙 호출
        all_noise_intervals.extend(define_noise_from_u_beat(data, r_peaks, predictions_dict, class_map, srate))
        all_noise_intervals.extend(detect_isoelectric(data, srate, threshold_mv=0.05))
        all_noise_intervals.extend(detect_irregular_u(data, predictions_dict, class_map, srate, std_thresh=10))
        all_noise_intervals.extend(detect_steep_slope(data, predictions_dict, srate, slope_thresh_mv=0.9))
        all_noise_intervals.extend(detect_consecutive_p(data, predictions_dict, class_map, srate, min_p_seq=4))
        all_noise_intervals.extend(detect_amp_variation(data, predictions_dict, class_map, srate, ratio=10.0))

    # 노이즈 구간 병합 및 포맷팅
    merged_noise = merge_intervals(all_noise_intervals)
    out_noise = []
    for i, (start_sample, end_sample) in enumerate(merged_noise):
        out_noise.append({'dt': start_sample / srate, 'val': f'Noise_start{i+1}'})
        out_noise.append({'dt': end_sample / srate, 'val': f'Noise_end{i+1}'})

    # --- 3. 결과 통합 및 반환 ---
    return [out_bstr + out_noise]


# ================================================================
# 4. PyVital 필터 실행 설정
# ================================================================

cfg = {
    'name': 'ECG - AI Beat & Noise Detector',
    'group': 'Medical algorithms',
    'desc': 'Detects ECG beat type (N, S, V, U) and noise intervals.',
    'reference': '',
    'overlap': 5,
    'interval': 15,
    'inputs': [{'name': 'ECG', 'type': 'wav'}],
    'outputs': [{'name': 'BEAT_NOISE', 'type': 'str'}]
}