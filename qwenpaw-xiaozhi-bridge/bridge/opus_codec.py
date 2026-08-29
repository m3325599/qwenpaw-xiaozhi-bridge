"""Opus codec helpers and xiaozhi binary protocol framing.

Binary protocol versions (see docs/websocket.md of the xiaozhi firmware):

- v1: raw Opus payload, no header
- v2: BinaryProtocol2 (all fields network byte order / big endian)
      uint16 version, uint16 type, uint32 reserved, uint32 timestamp,
      uint32 payload_size, uint8 payload[]
- v3: BinaryProtocol3
      uint8 type, uint8 reserved, uint16 payload_size, uint8 payload[]
"""

from __future__ import annotations

import struct

from opuslib.classes import Decoder, Encoder

# The xiaozhi protocol uses fixed 60 ms Opus frames.
FRAME_DURATION_MS = 60

# Header sizes
_V2_HEADER = struct.Struct("!HHIII")  # version, type, reserved, timestamp, payload_size
_V3_HEADER = struct.Struct("!BBH")  # type, reserved, payload_size

MSG_TYPE_OPUS = 0
MSG_TYPE_JSON = 1


class OpusDecoder:
    """Decode device uplink Opus frames (16 kHz mono) into PCM16 bytes."""

    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self.frame_size = sample_rate * FRAME_DURATION_MS // 1000  # samples / frame
        self._decoder = Decoder(sample_rate, 1)

    def decode(self, opus: bytes) -> bytes:
        if not opus:
            return b""
        try:
            return self._decoder.decode(opus, self.frame_size)
        except Exception:
            # Corrupted frame or DTX: drop it silently.
            return b""


class OpusEncoder:
    """Encode PCM16 mono chunks into 60 ms Opus frames for the downlink."""

    def __init__(self, sample_rate: int = 24000):
        self.sample_rate = sample_rate
        self.frame_size = sample_rate * FRAME_DURATION_MS // 1000  # samples / frame
        self.frame_bytes = self.frame_size * 2  # bytes per PCM16 mono frame
        self._encoder = Encoder(sample_rate, 1, "voip")

    def encode(self, pcm: bytes) -> bytes:
        if len(pcm) < self.frame_bytes:
            # Pad the tail of a turn with silence.
            pcm = pcm.ljust(self.frame_bytes, b"\x00")
        return self._encoder.encode(pcm, self.frame_size)


def parse_device_binary(data: bytes, version: int) -> bytes | None:
    """Extract the Opus payload from a binary frame sent by the device.

    Returns None when the frame is not an audio frame (or malformed).
    """
    if version <= 1:
        return data
    if version == 2:
        if len(data) < _V2_HEADER.size:
            return None
        _ver, msg_type, _reserved, _ts, payload_size = _V2_HEADER.unpack_from(data)
        if msg_type != MSG_TYPE_OPUS:
            return None
        end = min(_V2_HEADER.size + payload_size, len(data))
        return data[_V2_HEADER.size:end]
    if version == 3:
        if len(data) < _V3_HEADER.size:
            return None
        msg_type, _reserved, payload_size = _V3_HEADER.unpack_from(data)
        if msg_type != MSG_TYPE_OPUS:
            return None
        end = min(_V3_HEADER.size + payload_size, len(data))
        return data[_V3_HEADER.size:end]
    return data


def build_server_binary(opus: bytes, version: int, timestamp_ms: int = 0) -> bytes:
    """Wrap an Opus payload into the binary frame expected by the device."""
    if version <= 1:
        return opus
    if version == 2:
        return _V2_HEADER.pack(
            version, MSG_TYPE_OPUS, 0, timestamp_ms & 0xFFFFFFFF, len(opus)
        ) + opus
    # version 3
    return _V3_HEADER.pack(MSG_TYPE_OPUS, 0, len(opus)) + opus
