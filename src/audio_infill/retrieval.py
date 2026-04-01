from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import time

import numpy as np
import torch


def ensure_mono_float32(audio: np.ndarray | torch.Tensor) -> np.ndarray:
    if isinstance(audio, torch.Tensor):
        arr = audio.detach().cpu().numpy()
    else:
        arr = np.asarray(audio)
    out = np.asarray(arr, dtype=np.float32).reshape(-1)
    if out.size == 0:
        raise ValueError("Audio is empty.")
    return out


def peak_normalize(audio: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x = ensure_mono_float32(audio).copy()
    peak = float(np.max(np.abs(x)))
    if peak > eps:
        x /= peak
    return x


def frame_bounds_to_sample_bounds(
    total_samples: int,
    total_frames: int,
    start_frame: int,
    end_frame: int,
) -> Tuple[int, int]:
    if total_samples <= 0:
        return 0, 0
    samples_per_frame = float(total_samples) / max(1, int(total_frames))
    start_sample = int(round(max(0, start_frame) * samples_per_frame))
    end_sample = int(round(max(start_frame + 1, end_frame) * samples_per_frame))
    start_sample = min(max(0, start_sample), total_samples - 1)
    end_sample = min(max(start_sample + 1, end_sample), total_samples)
    return start_sample, end_sample


def slice_audio_for_frames(
    waveform: np.ndarray,
    *,
    total_frames: int,
    start_frame: int,
    end_frame: int,
) -> np.ndarray:
    sample_bounds = frame_bounds_to_sample_bounds(
        total_samples=int(waveform.shape[0]),
        total_frames=total_frames,
        start_frame=start_frame,
        end_frame=end_frame,
    )
    return waveform[sample_bounds[0] : sample_bounds[1]].copy()


def extract_mel_feature(
    audio: np.ndarray,
    sr: int,
    n_fft: int,
    hop_length: int,
    n_mels: int,
    fmin: float,
    normalize: bool,
) -> np.ndarray:
    import librosa

    x = peak_normalize(audio) if normalize else ensure_mono_float32(audio)
    mel = librosa.feature.melspectrogram(
        y=x,
        sr=sr,
        n_fft=int(n_fft),
        hop_length=int(hop_length),
        n_mels=int(n_mels),
        fmin=float(fmin),
        power=2.0,
    )
    return librosa.power_to_db(mel + 1e-10, ref=np.max).astype(np.float32)


def extract_stft_mag_feature(
    audio: np.ndarray,
    sr: int,
    n_fft: int,
    hop_length: int,
    normalize: bool,
) -> np.ndarray:
    import librosa

    del sr
    x = peak_normalize(audio) if normalize else ensure_mono_float32(audio)
    stft = librosa.stft(y=x, n_fft=int(n_fft), hop_length=int(hop_length), win_length=int(n_fft))
    return np.log1p(np.abs(stft)).astype(np.float32)


def extract_fill_feature(audio: np.ndarray, sr: int, cfg: Dict[str, Any], feature_type: Optional[str] = None) -> np.ndarray:
    kind = feature_type or str(cfg["feature_type"])
    if kind == "mel":
        return extract_mel_feature(
            audio,
            sr,
            n_fft=int(cfg["mel_n_fft"]),
            hop_length=int(cfg["mel_hop_length"]),
            n_mels=int(cfg["mel_n_mels"]),
            fmin=float(cfg["mel_fmin"]),
            normalize=bool(cfg["normalize_features"]),
        )
    if kind == "stft_mag":
        return extract_stft_mag_feature(
            audio,
            sr,
            n_fft=int(cfg["stft_n_fft"]),
            hop_length=int(cfg["stft_hop_length"]),
            normalize=bool(cfg["normalize_features"]),
        )
    raise ValueError(f"Unsupported feature type: {kind}")


def flatten_or_pool_feature(feature: np.ndarray, mode: str = "mean_std") -> np.ndarray:
    feat = np.asarray(feature, dtype=np.float32)
    if feat.ndim == 1:
        vec = feat
    elif feat.ndim == 2:
        if feat.shape[1] == 0:
            vec = feat.mean(axis=1)
        elif mode == "mean":
            vec = feat.mean(axis=1)
        elif mode == "mean_std":
            vec = np.concatenate([feat.mean(axis=1), feat.std(axis=1)], axis=0)
        else:
            raise ValueError(f"Unsupported pool mode: {mode}")
    else:
        raise ValueError(f"Expected 1D or 2D feature, got shape {feat.shape}")
    norm = np.linalg.norm(vec) + 1e-8
    return (vec / norm).astype(np.float32)


def extract_context_feature(left_audio: np.ndarray, right_audio: np.ndarray, sr: int, cfg: Dict[str, Any]) -> np.ndarray:
    left_feat = extract_fill_feature(left_audio, sr, cfg)
    right_feat = extract_fill_feature(right_audio, sr, cfg)
    left_vec = flatten_or_pool_feature(left_feat, mode=str(cfg["pool_mode"]))
    right_vec = flatten_or_pool_feature(right_feat, mode=str(cfg["pool_mode"]))
    vec = np.concatenate([left_vec, right_vec], axis=0)
    return (vec / (np.linalg.norm(vec) + 1e-8)).astype(np.float32)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a2 = np.asarray(a, dtype=np.float32).reshape(-1)
    b2 = np.asarray(b, dtype=np.float32).reshape(-1)
    denom = (np.linalg.norm(a2) * np.linalg.norm(b2)) + 1e-8
    return float(np.dot(a2, b2) / denom)


def ranges_overlap(start_a: int, end_a: int, start_b: int, end_b: int) -> bool:
    return start_a < end_b and end_a > start_b


@dataclass
class RetrievalCacheEntry:
    fill_start_frame: int
    fill_end_frame: int
    fill_start_sample: int
    fill_end_sample: int
    left_context_start_frame: int
    left_context_end_frame: int
    left_context_start_sample: int
    left_context_end_sample: int
    right_context_start_frame: int
    right_context_end_frame: int
    right_context_start_sample: int
    right_context_end_sample: int
    key_vector: np.ndarray
    fill_tokens: torch.Tensor
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> Dict[str, Any]:
        return {
            "fill_start_frame": int(self.fill_start_frame),
            "fill_end_frame": int(self.fill_end_frame),
            "fill_start_sample": int(self.fill_start_sample),
            "fill_end_sample": int(self.fill_end_sample),
            "left_context_start_frame": int(self.left_context_start_frame),
            "left_context_end_frame": int(self.left_context_end_frame),
            "left_context_start_sample": int(self.left_context_start_sample),
            "left_context_end_sample": int(self.left_context_end_sample),
            "right_context_start_frame": int(self.right_context_start_frame),
            "right_context_end_frame": int(self.right_context_end_frame),
            "right_context_start_sample": int(self.right_context_start_sample),
            "right_context_end_sample": int(self.right_context_end_sample),
            "key_vector": np.asarray(self.key_vector, dtype=np.float32),
            "fill_tokens": self.fill_tokens.detach().cpu().to(dtype=torch.long),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_payload(cls, payload: Dict[str, Any]) -> "RetrievalCacheEntry":
        return cls(
            fill_start_frame=int(payload["fill_start_frame"]),
            fill_end_frame=int(payload["fill_end_frame"]),
            fill_start_sample=int(payload["fill_start_sample"]),
            fill_end_sample=int(payload["fill_end_sample"]),
            left_context_start_frame=int(payload["left_context_start_frame"]),
            left_context_end_frame=int(payload["left_context_end_frame"]),
            left_context_start_sample=int(payload["left_context_start_sample"]),
            left_context_end_sample=int(payload["left_context_end_sample"]),
            right_context_start_frame=int(payload["right_context_start_frame"]),
            right_context_end_frame=int(payload["right_context_end_frame"]),
            right_context_start_sample=int(payload["right_context_start_sample"]),
            right_context_end_sample=int(payload["right_context_end_sample"]),
            key_vector=np.asarray(payload["key_vector"], dtype=np.float32),
            fill_tokens=torch.as_tensor(payload["fill_tokens"], dtype=torch.long).cpu(),
            metadata=dict(payload.get("metadata", {})),
        )


@dataclass
class RetrievalBank:
    mask_len: int
    left_context_frames: int
    right_context_frames: int
    stride_frames: int
    entries: List[RetrievalCacheEntry]
    key_matrix: torch.Tensor

    def to_payload(self) -> Dict[str, Any]:
        return {
            "mask_len": int(self.mask_len),
            "left_context_frames": int(self.left_context_frames),
            "right_context_frames": int(self.right_context_frames),
            "stride_frames": int(self.stride_frames),
            "entries": [entry.to_payload() for entry in self.entries],
            "key_matrix": self.key_matrix.detach().cpu(),
        }

    @classmethod
    def from_payload(cls, payload: Dict[str, Any]) -> "RetrievalBank":
        entries = [RetrievalCacheEntry.from_payload(item) for item in payload.get("entries", [])]
        key_matrix = torch.as_tensor(payload.get("key_matrix", torch.empty((0, 0))), dtype=torch.float32).cpu()
        return cls(
            mask_len=int(payload["mask_len"]),
            left_context_frames=int(payload["left_context_frames"]),
            right_context_frames=int(payload["right_context_frames"]),
            stride_frames=int(payload["stride_frames"]),
            entries=entries,
            key_matrix=key_matrix,
        )


@dataclass
class RetrievedCandidate:
    entry: RetrievalCacheEntry
    similarity: float


class SameSongRetrievalCache:
    def __init__(
        self,
        *,
        waveform: torch.Tensor | np.ndarray,
        codes: torch.Tensor,
        gaps: Sequence[Tuple[int, int]],
        target_sr: int,
        cache_dir: str,
        sample_name: str,
        feature_cfg: Dict[str, Any],
        rebuild_cache: bool = False,
        supported_lengths: Optional[Sequence[int]] = None,
    ):
        self.waveform = ensure_mono_float32(waveform)
        self.codes = torch.as_tensor(codes, dtype=torch.long).cpu()
        self.total_frames = int(self.codes.shape[1])
        self.total_samples = int(self.waveform.shape[0])
        self.target_sr = int(target_sr)
        self.gaps = [(int(start), int(end)) for start, end in gaps]
        self.feature_cfg = dict(feature_cfg)
        self.rebuild_cache = bool(rebuild_cache)
        self.sample_name = str(sample_name)
        self.sample_slug = self._slugify_component(sample_name)
        self.cache_root = Path(cache_dir) / self.sample_slug
        self.cache_root.mkdir(parents=True, exist_ok=True)
        self._banks: Dict[Tuple[int, int, int, int], RetrievalBank] = {}
        self.supported_lengths = tuple(sorted({int(v) for v in (supported_lengths or []) if int(v) > 0}))

    @staticmethod
    def _slugify_component(text: str) -> str:
        safe = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(text))
        safe = safe.strip("_")
        return safe or "item"

    def _feature_signature(self) -> str:
        payload = {
            "feature_type": self.feature_cfg["feature_type"],
            "pool_mode": self.feature_cfg["pool_mode"],
            "normalize_features": self.feature_cfg["normalize_features"],
            "mel_n_fft": self.feature_cfg["mel_n_fft"],
            "mel_hop_length": self.feature_cfg["mel_hop_length"],
            "mel_n_mels": self.feature_cfg["mel_n_mels"],
            "mel_fmin": self.feature_cfg["mel_fmin"],
            "stft_n_fft": self.feature_cfg["stft_n_fft"],
            "stft_hop_length": self.feature_cfg["stft_hop_length"],
        }
        digest = hashlib.sha1(repr(sorted(payload.items())).encode("utf-8")).hexdigest()
        return digest[:12]

    def _bank_path(self, mask_len: int, left_context_frames: int, right_context_frames: int, stride_frames: int) -> Path:
        feature_sig = self._feature_signature()
        name = (
            f"len_{int(mask_len):04d}"
            f"__l_{int(left_context_frames):04d}"
            f"__r_{int(right_context_frames):04d}"
            f"__s_{int(stride_frames):04d}"
            f"__{feature_sig}.pt"
        )
        return self.cache_root / name

    def cache_size(self) -> int:
        return sum(len(bank.entries) for bank in self._banks.values())

    def resolve_bucket_length(self, mask_len: int) -> int:
        target = int(mask_len)
        if target <= 0:
            raise ValueError("mask_len must be > 0")
        if not self.supported_lengths:
            return target
        return min(self.supported_lengths, key=lambda item: (abs(item - target), item))

    def _span_overlaps_gap(self, start_frame: int, end_frame: int) -> bool:
        return any(ranges_overlap(start_frame, end_frame, gap_start, gap_end) for gap_start, gap_end in self.gaps)

    def _extract_context_key(
        self,
        *,
        fill_start_frame: int,
        fill_end_frame: int,
        left_context_frames: int,
        right_context_frames: int,
    ) -> Optional[np.ndarray]:
        left_start = fill_start_frame - left_context_frames
        right_end = fill_end_frame + right_context_frames
        if left_start < 0 or right_end > self.total_frames:
            return None
        left_audio = slice_audio_for_frames(
            self.waveform,
            total_frames=self.total_frames,
            start_frame=left_start,
            end_frame=fill_start_frame,
        )
        right_audio = slice_audio_for_frames(
            self.waveform,
            total_frames=self.total_frames,
            start_frame=fill_end_frame,
            end_frame=right_end,
        )
        return extract_context_feature(left_audio, right_audio, self.target_sr, self.feature_cfg)

    def build_bank(
        self,
        *,
        mask_len: int,
        left_context_frames: int,
        right_context_frames: int,
        stride_frames: int,
    ) -> RetrievalBank:
        entries: List[RetrievalCacheEntry] = []
        min_start = int(left_context_frames)
        max_start = int(self.total_frames - mask_len - right_context_frames)
        if max_start < min_start:
            return RetrievalBank(
                mask_len=mask_len,
                left_context_frames=left_context_frames,
                right_context_frames=right_context_frames,
                stride_frames=stride_frames,
                entries=[],
                key_matrix=torch.empty((0, 0), dtype=torch.float32),
            )

        starts = list(range(min_start, max_start + 1, max(1, int(stride_frames))))
        if starts[-1] != max_start:
            starts.append(max_start)

        for start_frame in starts:
            end_frame = int(start_frame + mask_len)
            left_start_frame = int(start_frame - left_context_frames)
            right_end_frame = int(end_frame + right_context_frames)
            if left_start_frame < 0 or right_end_frame > self.total_frames:
                continue
            if self._span_overlaps_gap(left_start_frame, right_end_frame):
                continue
            key_vector = self._extract_context_key(
                fill_start_frame=start_frame,
                fill_end_frame=end_frame,
                left_context_frames=left_context_frames,
                right_context_frames=right_context_frames,
            )
            if key_vector is None:
                continue
            fill_sample_bounds = frame_bounds_to_sample_bounds(
                self.total_samples,
                self.total_frames,
                start_frame,
                end_frame,
            )
            left_sample_bounds = frame_bounds_to_sample_bounds(
                self.total_samples,
                self.total_frames,
                left_start_frame,
                start_frame,
            )
            right_sample_bounds = frame_bounds_to_sample_bounds(
                self.total_samples,
                self.total_frames,
                end_frame,
                right_end_frame,
            )
            entries.append(
                RetrievalCacheEntry(
                    fill_start_frame=start_frame,
                    fill_end_frame=end_frame,
                    fill_start_sample=int(fill_sample_bounds[0]),
                    fill_end_sample=int(fill_sample_bounds[1]),
                    left_context_start_frame=left_start_frame,
                    left_context_end_frame=int(start_frame),
                    left_context_start_sample=int(left_sample_bounds[0]),
                    left_context_end_sample=int(left_sample_bounds[1]),
                    right_context_start_frame=int(end_frame),
                    right_context_end_frame=right_end_frame,
                    right_context_start_sample=int(right_sample_bounds[0]),
                    right_context_end_sample=int(right_sample_bounds[1]),
                    key_vector=key_vector,
                    fill_tokens=self.codes[:, start_frame:end_frame].clone(),
                    metadata={"sample_name": self.sample_name},
                )
            )

        key_matrix = torch.from_numpy(np.stack([entry.key_vector for entry in entries], axis=0)) if entries else torch.empty((0, 0), dtype=torch.float32)
        return RetrievalBank(
            mask_len=int(mask_len),
            left_context_frames=int(left_context_frames),
            right_context_frames=int(right_context_frames),
            stride_frames=int(stride_frames),
            entries=entries,
            key_matrix=key_matrix,
        )

    def get_bank(
        self,
        *,
        mask_len: int,
        left_context_frames: int,
        right_context_frames: int,
        stride_frames: int,
    ) -> RetrievalBank:
        resolved_mask_len = self.resolve_bucket_length(mask_len)
        bank_key = (int(resolved_mask_len), int(left_context_frames), int(right_context_frames), int(stride_frames))
        if bank_key in self._banks:
            return self._banks[bank_key]

        path = self._bank_path(*bank_key)
        if path.exists() and not self.rebuild_cache:
            payload = torch.load(path, map_location="cpu", weights_only=False)
            bank = RetrievalBank.from_payload(payload)
        else:
            bank = self.build_bank(
                mask_len=bank_key[0],
                left_context_frames=bank_key[1],
                right_context_frames=bank_key[2],
                stride_frames=bank_key[3],
            )
            torch.save(bank.to_payload(), path)
        self._banks[bank_key] = bank
        return bank

    def prebuild_banks(
        self,
        *,
        bucket_lengths: Sequence[int],
        left_context_resolver: Any,
        stride_resolver: Any,
        logger: Optional[Any] = None,
    ) -> Dict[str, Any]:
        lengths = tuple(sorted({int(v) for v in bucket_lengths if int(v) > 0}))
        self.supported_lengths = lengths
        started = time.time()
        summaries: List[Dict[str, Any]] = []
        for length in lengths:
            left_ctx, right_ctx = left_context_resolver(int(length))
            stride = int(stride_resolver(int(length)))
            path = self._bank_path(int(length), int(left_ctx), int(right_ctx), int(stride))
            cache_hit = bool(path.exists() and not self.rebuild_cache)
            bank = self.get_bank(
                mask_len=int(length),
                left_context_frames=int(left_ctx),
                right_context_frames=int(right_ctx),
                stride_frames=int(stride),
            )
            item = {
                "mask_len": int(length),
                "left_context_frames": int(left_ctx),
                "right_context_frames": int(right_ctx),
                "stride_frames": int(stride),
                "entries": int(len(bank.entries)),
                "cache_hit": cache_hit,
            }
            summaries.append(item)
            if logger is not None:
                logger.info(
                    "Retrieval bank ready: len=%d ctx=(%d,%d) stride=%d entries=%d source=%s",
                    item["mask_len"],
                    item["left_context_frames"],
                    item["right_context_frames"],
                    item["stride_frames"],
                    item["entries"],
                    "cache" if cache_hit else "built",
                )
        elapsed = time.time() - started
        return {
            "bucket_lengths": lengths,
            "banks": summaries,
            "elapsed_sec": float(elapsed),
            "total_entries": int(sum(item["entries"] for item in summaries)),
        }

    def query(
        self,
        *,
        query_start_frame: int,
        query_end_frame: int,
        left_context_frames: int,
        right_context_frames: int,
        stride_frames: int,
        top_k: int,
        exclusion_margin_frames: int,
    ) -> List[RetrievedCandidate]:
        mask_len = int(query_end_frame - query_start_frame)
        if mask_len <= 0:
            return []
        resolved_mask_len = self.resolve_bucket_length(mask_len)
        query_key = self._extract_context_key(
            fill_start_frame=int(query_start_frame),
            fill_end_frame=int(query_end_frame),
            left_context_frames=int(left_context_frames),
            right_context_frames=int(right_context_frames),
        )
        if query_key is None:
            return []

        bank = self.get_bank(
            mask_len=resolved_mask_len,
            left_context_frames=int(left_context_frames),
            right_context_frames=int(right_context_frames),
            stride_frames=int(stride_frames),
        )
        if not bank.entries or bank.key_matrix.numel() == 0:
            return []

        blocked_start = int(query_start_frame - exclusion_margin_frames)
        blocked_end = int(query_end_frame + exclusion_margin_frames)
        valid_indices = [
            idx
            for idx, entry in enumerate(bank.entries)
            if not ranges_overlap(entry.fill_start_frame, entry.fill_end_frame, blocked_start, blocked_end)
        ]
        if not valid_indices:
            return []

        query_tensor = torch.from_numpy(np.asarray(query_key, dtype=np.float32))
        key_matrix = bank.key_matrix[valid_indices]
        similarities = torch.matmul(key_matrix, query_tensor)
        k = min(int(top_k), len(valid_indices))
        if k <= 0:
            return []
        top_values, top_local_indices = torch.topk(similarities, k=k)
        results: List[RetrievedCandidate] = []
        for sim, local_idx in zip(top_values.tolist(), top_local_indices.tolist()):
            entry = bank.entries[valid_indices[int(local_idx)]]
            results.append(RetrievedCandidate(entry=entry, similarity=float(sim)))
        return results


def build_retrieval_feature_config(cfg: Any) -> Dict[str, Any]:
    return {
        "feature_type": str(cfg.retrieval_feature_type),
        "pool_mode": str(cfg.retrieval_pool_mode),
        "normalize_features": bool(cfg.retrieval_normalize_features),
        "mel_n_fft": int(cfg.retrieval_mel_n_fft),
        "mel_hop_length": int(cfg.retrieval_mel_hop_length),
        "mel_n_mels": int(cfg.retrieval_mel_n_mels),
        "mel_fmin": float(cfg.retrieval_mel_fmin),
        "stft_n_fft": int(cfg.retrieval_stft_n_fft),
        "stft_hop_length": int(cfg.retrieval_stft_hop_length),
    }
