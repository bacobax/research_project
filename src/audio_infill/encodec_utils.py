from pathlib import Path
from typing import Any, Dict, Optional

import torch


SUPPORTED_ENCODEC_MODELS = {"encodec_24khz"}


def build_encodec_model(encodec_model: str):
    from encodec import EncodecModel

    if encodec_model == "encodec_24khz":
        return EncodecModel.encodec_model_24khz()
    raise ValueError(
        f"Unsupported EnCodec model '{encodec_model}'. Supported values: {sorted(SUPPORTED_ENCODEC_MODELS)}"
    )


def codes_to_embeddings(model: Any, codes: torch.Tensor, device: Optional[torch.device] = None) -> torch.Tensor:
    if codes.dim() == 2:
        codes = codes.unsqueeze(0)
    if codes.dim() != 3:
        raise ValueError(f"codes_to_embeddings expects [B,K,T] or [K,T], got shape={tuple(codes.shape)}")

    target_device = device if device is not None else next(model.parameters()).device
    codes = codes.to(target_device, dtype=torch.long)
    quantized_out = None
    layers = model.quantizer.vq.layers
    if codes.shape[1] > len(layers):
        raise ValueError(f"codes K={codes.shape[1]} exceeds quantizer layers={len(layers)}")
    for q in range(codes.shape[1]):
        quantized = layers[q].decode(codes[:, q, :])
        quantized_out = quantized if quantized_out is None else quantized_out + quantized
    assert quantized_out is not None
    return quantized_out


def logits_to_embeddings(
    model: Any,
    logits: torch.Tensor,
    *,
    temperature: float = 1.0,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    if logits.dim() != 4:
        raise ValueError(f"logits_to_embeddings expects [B,K,T,V], got shape={tuple(logits.shape)}")
    if temperature <= 0:
        raise ValueError("temperature must be > 0")

    target_device = device if device is not None else next(model.parameters()).device
    logits = logits.to(target_device, dtype=torch.float32)
    probs = torch.softmax(logits / temperature, dim=-1)
    quantized_out = None
    layers = model.quantizer.vq.layers
    if logits.shape[1] > len(layers):
        raise ValueError(f"logits K={logits.shape[1]} exceeds quantizer layers={len(layers)}")
    for q in range(logits.shape[1]):
        layer = layers[q]
        codebook = layer.codebook.to(device=logits.device, dtype=logits.dtype)
        soft_quantized = torch.matmul(probs[:, q, :, :], codebook)
        soft_quantized = layer.project_out(soft_quantized)
        soft_quantized = soft_quantized.transpose(1, 2)
        quantized_out = soft_quantized if quantized_out is None else quantized_out + soft_quantized
    assert quantized_out is not None
    return quantized_out


def decode_embeddings(
    decoder: torch.nn.Module,
    embeddings: torch.Tensor,
    *,
    device: torch.device,
    scale: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if embeddings.dim() != 3:
        raise ValueError(f"decode_embeddings expects [B,D,T], got shape={tuple(embeddings.shape)}")

    embeddings = embeddings.to(device, dtype=torch.float32)
    decoder_was_training = decoder.training
    if torch.is_grad_enabled():
        # cuDNN LSTM backward requires the decoder forward to run in training mode.
        decoder.train(True)
    try:
        wav_out = decoder(embeddings)
    finally:
        decoder.train(decoder_was_training)
    if scale is not None:
        wav_out = wav_out * scale.view(-1, 1, 1)
    return wav_out


def export_decoder_artifact(
    path: str,
    *,
    decoder_state_dict: Dict[str, Any],
    encodec_model: str,
    bandwidth: float,
    target_sr: int,
    step: int,
    run_name: str,
    config_path: Optional[str] = None,
) -> None:
    payload = {
        "artifact_type": "encodec_decoder",
        "encodec_model": encodec_model,
        "bandwidth": float(bandwidth),
        "target_sr": int(target_sr),
        "step": int(step),
        "run_name": run_name,
        "config_path": config_path,
        "decoder_state_dict": decoder_state_dict,
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_decoder_artifact(path: str) -> Dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"Decoder artifact at {path} must contain a mapping payload")
    if payload.get("artifact_type") != "encodec_decoder":
        raise ValueError(
            f"Decoder artifact at {path} has artifact_type={payload.get('artifact_type')!r}; expected 'encodec_decoder'"
        )
    if "decoder_state_dict" not in payload or not isinstance(payload["decoder_state_dict"], dict):
        raise ValueError(f"Decoder artifact at {path} is missing decoder_state_dict")
    return payload


def load_exported_decoder(
    path: str,
    *,
    decoder: torch.nn.Module,
    expected_encodec_model: str,
    expected_bandwidth: float,
) -> Dict[str, Any]:
    payload = load_decoder_artifact(path)

    artifact_model = payload.get("encodec_model")
    if artifact_model != expected_encodec_model:
        raise ValueError(
            f"Decoder artifact model mismatch: artifact encodec_model={artifact_model!r}, "
            f"expected {expected_encodec_model!r}"
        )

    artifact_bandwidth = payload.get("bandwidth")
    if artifact_bandwidth is None:
        raise ValueError(f"Decoder artifact at {path} is missing bandwidth metadata")
    if abs(float(artifact_bandwidth) - float(expected_bandwidth)) > 1e-6:
        raise ValueError(
            f"Decoder artifact bandwidth mismatch: artifact bandwidth={artifact_bandwidth}, "
            f"expected {expected_bandwidth}"
        )

    decoder_state = payload["decoder_state_dict"]
    current_state = decoder.state_dict()
    missing = sorted(set(current_state.keys()) - set(decoder_state.keys()))
    unexpected = sorted(set(decoder_state.keys()) - set(current_state.keys()))
    mismatched = sorted(
        key
        for key in current_state.keys() & decoder_state.keys()
        if tuple(current_state[key].shape) != tuple(decoder_state[key].shape)
    )
    if missing or unexpected or mismatched:
        raise ValueError(
            "Decoder artifact is incompatible with the current EnCodec decoder: "
            f"missing={missing}, unexpected={unexpected}, mismatched_shapes={mismatched}"
        )

    decoder.load_state_dict(decoder_state, strict=True)
    return payload
