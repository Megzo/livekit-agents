# Copyright 2023 LiveKit, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from dataclasses import dataclass

import torch
from kokoro import KPipeline
from livekit import rtc
from livekit.agents import (
    DEFAULT_API_CONNECT_OPTIONS,
    APIConnectionError,
    APIConnectOptions,
    APITimeoutError,
    tts,
    utils,
)

from .log import logger


@dataclass
class _TTSOptions:
    repo_or_dir: str
    model: str
    lang_code: str
    voice: str
    device: torch.device
    cpu_cores: int


class TTS(tts.TTS):
    def __init__(
            self,
            *,
            repo_or_dir: str = 'hexgrad/Kokoro-82M',
            model: str = 'v1.0',
            lang_code: str = 'a', 
            voice: str = 'af_heart',
            sample_rate: int = 24000,
            device: torch.device = torch.device('cpu'),
            cpu_cores: int = 4,
    ) -> None:
        """
        Create a new instance of Kokoro TTS.

        Args:
            repo_or_dir (str): Repository or directory containing the model. Defaults to 'hexgrad/Kokoro-82M'.
            model (str): Model name. Defaults to 'v1.0'.
            lang_code (str): Language code. Defaults to 'a', which means American English.
            voice (str): Voice model. Defaults to 'af_heart'.
            sample_rate (int): Sample rate for the output audio. Defaults to 8000.
            device (torch.device): Device to use for inference. Defaults to 'cpu'.
            cpu_cores (int): Number of CPU cores to use when device is 'cpu'. Defaults to 4.
        """

        super().__init__(
            capabilities=tts.TTSCapabilities(
                streaming=False,
            ),
            sample_rate=sample_rate,
            num_channels=1,
        )

        self._opts = _TTSOptions(
            repo_or_dir=repo_or_dir,
            model=model,
            lang_code=lang_code,
            voice=voice,
            device=device,
            cpu_cores=cpu_cores,
        )

        self._pipeline = KPipeline(lang_code=self._opts.lang_code)

    def update_options(
            self,
            *,
            repo_or_dir: str | None = None,
            model: str | None = None,
            lang_code: str | None = None,
            voice: str | None = None,
            device: torch.device | None = None,
            cpu_cores: int | None = None,
    ) -> None:
        """Update TTS options. Note that changing some options may require reinitializing the model."""
        self._opts.repo_or_dir = repo_or_dir or self._opts.repo_or_dir
        self._opts.model = model or self._opts.model
        self._opts.lang_code = lang_code or self._opts.lang_code
        self._opts.voice = voice or self._opts.voice
        self._opts.device = device or self._opts.device
        self._opts.cpu_cores = cpu_cores or self._opts.cpu_cores

    def synthesize(
            self,
            text: str,
            *,
            conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> "ChunkedStream":
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")
        return ChunkedStream(
            tts=self,
            input_text=text,
            opts=self._opts,
            conn_options=conn_options,
        )


class ChunkedStream(tts.ChunkedStream):
    def __init__(
            self,
            *,
            tts: TTS,
            input_text: str,
            opts: _TTSOptions,
            conn_options: APIConnectOptions,
    ) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._opts = opts

    async def _run(self) -> None:
        request_id = utils.shortuuid()
        try:
            generator = self._tts._pipeline(
                self._input_text, voice=self._opts.voice, 
                speed=1, split_pattern=r'\n+'
            )
            for i, (_, _, audio) in enumerate(generator):
                # Convert float32 to int16 directly from tensor
                audio_int16 = (audio * 32767).to(torch.int16)

                audio_frame = rtc.AudioFrame(
                    data=audio_int16.numpy().tobytes(),
                    sample_rate=self._tts.sample_rate,
                    num_channels=1,
                    samples_per_channel=len(audio),
                )
                self._event_ch.send_nowait(
                    tts.SynthesizedAudio(
                        request_id=request_id,
                        frame=audio_frame,
                    )
                )
        except RuntimeError as e:
            if "timeout" in str(e).lower():
                raise APITimeoutError() from e
        except Exception as e:
            logger.error("Kokoro TTS synthesis failed", exc_info=e)
            raise APIConnectionError() from e