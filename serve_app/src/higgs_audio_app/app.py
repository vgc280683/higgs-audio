import copy
import io
import os
import re
import time
from dataclasses import asdict
from typing import List
from typing import Optional

import langid
import numpy as np
import soundfile as sf
import streamlit as st
import torch
import tqdm
import yaml
from loguru import logger
from transformers import AutoConfig, AutoTokenizer
from transformers.cache_utils import StaticCache

from boson_multimodal.audio_processing.higgs_audio_tokenizer import load_higgs_audio_tokenizer
from boson_multimodal.data_collator.higgs_audio_collator import HiggsAudioSampleCollator
from boson_multimodal.data_types import Message, ChatMLSample, AudioContent, TextContent
from boson_multimodal.dataset.chatml_dataset import (
    ChatMLDatasetSample,
    prepare_chatml_sample,
)
from boson_multimodal.model.higgs_audio import HiggsAudioModel
from boson_multimodal.model.higgs_audio.utils import revert_delay_pattern


def concatenate_audio(audio_paths, output_path):
    """
    Une múltiples archivos de audio `.wav` (con el mismo sample rate) en un solo archivo.

    Args:
        audio_paths (list): Lista de rutas a archivos de audio `.wav`.
        output_path (str): Ruta de salida del archivo final concatenado.
    """
    audios = []
    sr_final = None

    for path in audio_paths:
        audio, sr = sf.read(path)
        if sr_final is None:
            sr_final = sr
        elif sr != sr_final:
            raise ValueError(f"Sample rate inconsistente: {path}")
        audios.append(audio)

    audio_concat = np.concatenate(audios, axis=0)
    sf.write(output_path, audio_concat, sr_final)


CURR_DIR = os.path.dirname(os.path.abspath(__file__))

AUDIO_PLACEHOLDER_TOKEN = "<|__AUDIO_PLACEHOLDER__|>"

MULTISPEAKER_DEFAULT_SYSTEM_MESSAGE = """You are an AI assistant designed to convert text into speech.
If the user's message includes a [SPEAKER*] tag, do not read out the tag and generate speech for the following text, using the specified voice.
If no speaker tag is present, select a suitable voice on your own."""


def concatenate_audio(audio_paths, output_path, pause_duration=0.5):
    """
    Une múltiples archivos de audio `.wav` con pausas entre ellos.
    """
    combined_audio = []
    sample_rate = None

    for i, path in enumerate(audio_paths):
        audio, sr = sf.read(path)
        if sample_rate is None:
            sample_rate = sr
        elif sr != sample_rate:
            raise ValueError(f"Incompatible sample rate en {path}")

        combined_audio.append(audio)

        if i < len(audio_paths) - 1 and pause_duration > 0:
            silence = np.zeros(int(sample_rate * pause_duration), dtype=audio.dtype)
            combined_audio.append(silence)

    full_audio = np.concatenate(combined_audio)
    sf.write(output_path, full_audio, sample_rate)


def dividir_texto_por_pausa(texto_largo, max_len=300):
    """
    Divide el texto por pausas naturales (puntos, signos de interrogación/exclamación, etc.)
    y garantiza que ninguna parte sea excesivamente larga.
    """
    # Separar por frases usando signos de puntuación como delimitadores
    frases = re.split(r'(?<=[.!?…])\s+', texto_largo.strip())

    # Agrupar frases para que no se pasen del tamaño máximo
    bloques = []
    bloque_actual = ""

    for frase in frases:
        if len(bloque_actual) + len(frase) <= max_len:
            bloque_actual += " " + frase if bloque_actual else frase
        else:
            bloques.append(bloque_actual.strip())
            bloque_actual = frase
    if bloque_actual:
        bloques.append(bloque_actual.strip())

    return bloques


def save_audio(audio, ruta_salida, samplerate=24000):
    """
    Guarda un array de audio como WAV en formato PCM 16 bits.
    """
    audio = np.array(audio).astype(np.float32)
    sf.write(ruta_salida, audio, samplerate)


def concatenar_audios_con_pausa(lista_rutas, pausa_segundos=0.5, samplerate=24000):
    """
    Concatena varios archivos WAV introduciendo una pausa de silencio entre ellos.
    """
    pausa = np.zeros(int(samplerate * pausa_segundos), dtype=np.float32)
    resultado = []

    for ruta in lista_rutas:
        audio, sr = sf.read(ruta)
        if sr != samplerate:
            raise ValueError(f"Sample rate de {ruta} es {sr}, se esperaba {samplerate}")
        resultado.append(audio)
        resultado.append(pausa.copy())

    return np.concatenate(resultado)


def generar_audio_higgs(
        text,
        voice_path,
        voice_name,
        preset,
        language,
        scene_prompt,
        model,
        temperature=1.0,
        top_k=50,
        top_p=0.95,
        ras_win_len=7,
        ras_win_max_num_repeat=2,
        seed=123
):
    # === Normalizar texto ===
    processed_text = normalize_chinese_punctuation(text)
    processed_text = processed_text.replace("(", " ").replace(")", " ")
    processed_text = processed_text.replace("°F", " degrees Fahrenheit").replace("°C", " degrees Celsius")
    processed_text = processed_text.strip()
    if not any(processed_text.endswith(c) for c in [".", "!", "?", ",", ";", '"', "'", "</SE_e>", "</SE>"]):
        processed_text += "."

    # === Preparar contexto para una sola voz ===
    speaker_tags = ["SPEAKER0"]  # Puedes hacer esto dinámico más adelante
    messages, audio_ids = prepare_generation_context(
        scene_prompt=scene_prompt,
        ref_audio=voice_name,  # Asume que el nombre coincide con un prompt en voice_prompts/
        ref_audio_in_system_message=True,
        audio_tokenizer=model._audio_tokenizer,
        speaker_tags=speaker_tags,
    )

    if not processed_text.strip():
        raise ValueError("❌ Texto procesado vacío en generación.")

    chunked_text = [f"SPEAKER0: {processed_text}"]  # ✔️
    st.info(f"{chunked_text}.")
    # NO AÑADIR este bloque a `messages`, porque ya lo hace el modelo internamente:
    # messages.append(Message(role="user", content=TextContent(...))) ❌

    concat_wav, sr, _ = model.generate(
        messages,  # generado por prepare_generation_context
        audio_ids,  # generado por prepare_generation_context
        chunked_text,  # ✔️ cadena plana con el texto del hablante
        None,
        temperature,
        top_k,
        top_p,
        ras_win_len,
        ras_win_max_num_repeat,
        seed,
    )
    return concat_wav, sr


def generar_narracion_completa_old(
        texto_largo,
        voice_path,
        voice_name,
        preset,
        language,
        scene_prompt,
        output_dir,
        model,
        pause_duration=0.5
):
    from datetime import datetime

    # === Crear subcarpeta con timestamp ===
    timestamp = datetime.now().strftime("%d%m%Y_%H%M%S")
    carpeta_salida = os.path.join(output_dir, timestamp)
    os.makedirs(carpeta_salida, exist_ok=True)
    # === Panel de logging lateral ===
    log_panel = st.sidebar.expander("📋 Log de generación en tiempo real", expanded=True)
    log_panel.markdown("Iniciando generación...")
    # === Dividir en frases ===
    bloques = dividir_texto_por_pausa(texto_largo)
    audios = []
    duraciones = []

    st.subheader("🧾 Frases detectadas:")
    st.markdown(f"Se han detectado **{len(bloques)} bloques** de texto para generar.")
    with st.expander("Ver bloques detectados"):
        for i, b in enumerate(bloques):
            st.write(f"**Bloque {i + 1}:** {b}")

    for i, bloque in enumerate(bloques):
        nombre_archivo = f"{timestamp}_frase{i + 1}.wav"
        ruta_salida = os.path.join(carpeta_salida, nombre_archivo)

        st.markdown(f"🔄 Generando bloque `{i + 1}/{len(bloques)}`:")
        status_text = st.empty()
        progress_bar = st.progress(i / len(bloques))

        t0 = time.time()

        # Generar audio con Higgs Audio
        audio, sr = generar_audio_higgs(
            text=bloque,
            voice_path=voice_path,
            voice_name=voice_name,
            preset=preset,
            language=language,
            scene_prompt=scene_prompt,
            model=model  # 👈 ahora sí se usa el que has pasado
        )

        save_audio(audio, ruta_salida, sr)

        t1 = time.time()
        duraciones.append(t1 - t0)
        log_panel.markdown(f"""
**🟢 Bloque {i + 1}/{len(bloques)}**
- ⏱️ Tiempo: `{t1 - t0:.2f} s`
- 🗣️ Texto: `{bloque[:100]}{"..." if len(bloque) > 100 else ""}`
""")
        status_text.markdown(f"✅ Bloque `{i + 1}` generado en **{t1 - t0:.2f} s**")
        progress_bar.progress((i + 1) / len(bloques))
        audios.append(ruta_salida)

    # === Concatenar audio final ===
    ruta_audio_final = os.path.join(carpeta_salida, f"{timestamp}_completo.wav")
    audio_final = concatenar_audios_con_pausa(audios, pausa_segundos=pause_duration)
    save_audio(audio_final, ruta_audio_final)

    import json

    log_info = {
        "timestamp": timestamp,
        "scene_prompt": scene_prompt,
        "language": language,
        "preset": preset,
        "voice_name": voice_name,
        "voice_path": voice_path,
        "output_dir": carpeta_salida,
        "num_bloques": len(bloques),
        "bloques": [
            {
                "bloque": i + 1,
                "texto": bloques[i],
                "duracion": duraciones[i],
                "archivo": os.path.basename(audios[i])
            }
            for i in range(len(bloques))
        ],
        "duracion_total": sum(duraciones)
    }

    ruta_log = os.path.join(carpeta_salida, f"{timestamp}_log.json")
    with open(ruta_log, "w", encoding="utf-8") as f:
        json.dump(log_info, f, ensure_ascii=False, indent=4)

    return {
        "ruta_audio_final": ruta_audio_final,
        "duraciones": duraciones,
        "archivos_individuales": audios,
        "carpeta_salida": carpeta_salida
    }

def generar_narracion_completa(
    texto_largo,
    voice_path,
    voice_name,
    preset,
    language,
    scene_prompt,
    output_dir,
    model,
    pause_duration=0.5,
    temperature=1.0,
    top_k=50,
    top_p=0.95,
    ras_win_len=7,
    ras_win_max_num_repeat=2,
    seed=123
):
    from datetime import datetime
    import json

    # === Crear subcarpeta con timestamp ===
    timestamp = datetime.now().strftime("%d%m%Y_%H%M%S")
    carpeta_salida = os.path.join(output_dir, timestamp)
    os.makedirs(carpeta_salida, exist_ok=True)

    # === Panel lateral de log ===
    log_panel = st.sidebar.expander("📋 Log de generación en tiempo real", expanded=True)
    log_panel.markdown("Iniciando generación...")

    # === Dividir en frases/bloques naturales ===
    bloques = dividir_texto_por_pausa(texto_largo)
    chunked_text = [f"SPEAKER0: {bloque}" for bloque in bloques]

    st.subheader("🧾 Frases detectadas:")
    st.markdown(f"Se han detectado **{len(chunked_text)} bloques** de texto para generar.")
    with st.expander("Ver bloques detectados"):
        for i, b in enumerate(chunked_text):
            st.write(f"**Bloque {i+1}:** {b}")

    # === Preparar contexto para generación coherente ===
    messages, audio_ids = prepare_generation_context(
        scene_prompt=scene_prompt,
        ref_audio=voice_name,
        ref_audio_in_system_message=True,
        audio_tokenizer=model._audio_tokenizer,
        speaker_tags=["SPEAKER0"],
    )

    if messages is None:
        st.error("❌ Error al preparar el contexto. Revisa el nombre del prompt de voz y los archivos en voice_examples.")
        return

    # === Generación única con todos los bloques ===
    st.markdown("🔊 **Generando audio completo...**")
    t0 = time.time()

    audio, sr, _ = model.generate(
        messages=messages,
        audio_ids=audio_ids,
        chunked_text=chunked_text,
        generation_chunk_buffer_size=None,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        ras_win_len=ras_win_len,
        ras_win_max_num_repeat=ras_win_max_num_repeat,
        seed=seed,
    )

    t1 = time.time()
    duracion_total = t1 - t0

    log_panel.markdown(f"🟢 Audio completo generado en **{duracion_total:.2f} s**")

    # === Guardar audio final ===
    ruta_audio_final = os.path.join(carpeta_salida, f"{timestamp}_completo.wav")
    save_audio(audio, ruta_audio_final, sr)

    # === Estimar duración por bloque (distribución proporcional simple) ===
    duracion_media = duracion_total / len(chunked_text)
    duraciones = [duracion_media] * len(chunked_text)

    # === Guardar log JSON ===
    log_info = {
        "timestamp": timestamp,
        "scene_prompt": scene_prompt,
        "language": language,
        "preset": preset,
        "voice_name": voice_name,
        "voice_path": voice_path,
        "output_dir": carpeta_salida,
        "num_bloques": len(chunked_text),
        "bloques": [
            {
                "bloque": i + 1,
                "texto": bloques[i],
                "duracion_estimada": duracion_media,
            }
            for i in range(len(bloques))
        ],
        "duracion_total": duracion_total
    }

    ruta_log = os.path.join(carpeta_salida, f"{timestamp}_log.json")
    with open(ruta_log, "w", encoding="utf-8") as f:
        json.dump(log_info, f, ensure_ascii=False, indent=4)

    return {
        "ruta_audio_final": ruta_audio_final,
        "duraciones": duraciones,
        "archivos_individuales": [],  # ahora no hay WAVs intermedios
        "carpeta_salida": carpeta_salida
    }

def normalize_chinese_punctuation(text):
    """
    Convert Chinese (full-width) punctuation marks to English (half-width) equivalents.
    """
    # Mapping of Chinese punctuation to English punctuation
    chinese_to_english_punct = {
        "，": ", ",  # comma
        "。": ".",  # period
        "：": ":",  # colon
        "；": ";",  # semicolon
        "？": "?",  # question mark
        "！": "!",  # exclamation mark
        "（": "(",  # left parenthesis
        "）": ")",  # right parenthesis
        "【": "[",  # left square bracket
        "】": "]",  # right square bracket
        "《": "<",  # left angle quote
        "》": ">",  # right angle quote
        "“": '"',  # left double quotation
        "”": '"',  # right double quotation
        "‘": "'",  # left single quotation
        "’": "'",  # right single quotation
        "、": ",",  # enumeration comma
        "—": "-",  # em dash
        "…": "...",  # ellipsis
        "·": ".",  # middle dot
        "「": '"',  # left corner bracket
        "」": '"',  # right corner bracket
        "『": '"',  # left double corner bracket
        "』": '"',  # right double corner bracket
    }

    # Replace each Chinese punctuation with its English counterpart
    for zh_punct, en_punct in chinese_to_english_punct.items():
        text = text.replace(zh_punct, en_punct)

    return text


def prepare_chunk_text(
        text, chunk_method: Optional[str] = None, chunk_max_word_num: int = 100, chunk_max_num_turns: int = 1
):
    """Chunk the text into smaller pieces. We will later feed the chunks one by one to the model.

    Parameters
    ----------
    text : str
        The text to be chunked.
    chunk_method : str, optional
        The method to use for chunking. Options are "speaker", "word", or None. By default, we won't use any chunking and
        will feed the whole text to the model.
    replace_speaker_tag_with_special_tags : bool, optional
        Whether to replace speaker tags with special tokens, by default False
        If the flag is set to True, we will replace [SPEAKER0] with <|speaker_id_start|>SPEAKER0<|speaker_id_end|>
    chunk_max_word_num : int, optional
        The maximum number of words for each chunk when "word" chunking method is used, by default 100
    chunk_max_num_turns : int, optional
        The maximum number of turns for each chunk when "speaker" chunking method is used,

    Returns
    -------
    List[str]
        The list of text chunks.

    """
    if chunk_method is None:
        return [text]
    elif chunk_method == "speaker":
        lines = text.split("\n")
        speaker_chunks = []
        speaker_utterance = ""
        for line in lines:
            line = line.strip()
            if line.startswith("[SPEAKER") or line.startswith("<|speaker_id_start|>"):
                if speaker_utterance:
                    speaker_chunks.append(speaker_utterance.strip())
                speaker_utterance = line
            else:
                if speaker_utterance:
                    speaker_utterance += "\n" + line
                else:
                    speaker_utterance = line
        if speaker_utterance:
            speaker_chunks.append(speaker_utterance.strip())
        if chunk_max_num_turns > 1:
            merged_chunks = []
            for i in range(0, len(speaker_chunks), chunk_max_num_turns):
                merged_chunk = "\n".join(speaker_chunks[i: i + chunk_max_num_turns])
                merged_chunks.append(merged_chunk)
            return merged_chunks
        return speaker_chunks
    elif chunk_method == "word":
        # TODO: We may improve the logic in the future
        # For long-form generation, we will first divide the text into multiple paragraphs by splitting with "\n\n"
        # After that, we will chunk each paragraph based on word count
        language = langid.classify(text)[0]
        paragraphs = text.split("\n\n")
        chunks = []
        for idx, paragraph in enumerate(paragraphs):
            words = paragraph.split(" ")
            for i in range(0, len(words), chunk_max_word_num):
                chunk = " ".join(words[i: i + chunk_max_word_num])
                chunks.append(chunk)
            chunks[-1] += "\n\n"
        return chunks
    else:
        raise ValueError(f"Unknown chunk method: {chunk_method}")


def _build_system_message_with_audio_prompt(system_message):
    contents = []

    while AUDIO_PLACEHOLDER_TOKEN in system_message:
        loc = system_message.find(AUDIO_PLACEHOLDER_TOKEN)
        contents.append(TextContent(system_message[:loc]))
        contents.append(AudioContent(audio_url=""))
        system_message = system_message[loc + len(AUDIO_PLACEHOLDER_TOKEN):]

    if len(system_message) > 0:
        contents.append(TextContent(system_message))
    ret = Message(
        role="system",
        content=contents,
    )
    return ret


class HiggsAudioModelClient:
    def __init__(
            self,
            model_path,
            audio_tokenizer,
            device=None,
            device_id=None,
            max_new_tokens=2048,
            kv_cache_lengths: List[int] = [1024, 4096, 8192],  # Multiple KV cache sizes,
            use_static_kv_cache=False,
    ):
        # Use explicit device if provided, otherwise try CUDA/MPS/CPU
        if device_id is not None:
            device = f"cuda:{device_id}"
            self._device = device
        else:
            if device is not None:
                self._device = device
            else:  # We get to choose the device
                # Prefer CUDA over MPS (Apple Silicon GPU) over CPU if available
                if torch.cuda.is_available():
                    self._device = "cuda:0"
                elif torch.backends.mps.is_available():
                    self._device = "mps"
                else:
                    self._device = "cpu"

        logger.info(f"Using device: {self._device}")
        if isinstance(audio_tokenizer, str):
            # For MPS, use CPU due to embedding operation limitations in quantization layers
            audio_tokenizer_device = "cpu" if self._device == "mps" else self._device
            self._audio_tokenizer = load_higgs_audio_tokenizer(audio_tokenizer, device=audio_tokenizer_device)
        else:
            self._audio_tokenizer = audio_tokenizer

        self._model = HiggsAudioModel.from_pretrained(
            model_path,
            device_map=self._device,
            torch_dtype=torch.bfloat16,
        )
        self._model.eval()
        self._kv_cache_lengths = kv_cache_lengths
        self._use_static_kv_cache = use_static_kv_cache

        self._tokenizer = AutoTokenizer.from_pretrained(model_path)
        self._config = AutoConfig.from_pretrained(model_path)
        self._max_new_tokens = max_new_tokens
        self._collator = HiggsAudioSampleCollator(
            whisper_processor=None,
            audio_in_token_id=self._config.audio_in_token_idx,
            audio_out_token_id=self._config.audio_out_token_idx,
            audio_stream_bos_id=self._config.audio_stream_bos_id,
            audio_stream_eos_id=self._config.audio_stream_eos_id,
            encode_whisper_embed=self._config.encode_whisper_embed,
            pad_token_id=self._config.pad_token_id,
            return_audio_in_tokens=self._config.encode_audio_in_tokens,
            use_delay_pattern=self._config.use_delay_pattern,
            round_to=1,
            audio_num_codebooks=self._config.audio_num_codebooks,
        )
        self.kv_caches = None
        if use_static_kv_cache:
            self._init_static_kv_cache()

    def _init_static_kv_cache(self):
        cache_config = copy.deepcopy(self._model.config.text_config)
        cache_config.num_hidden_layers = self._model.config.text_config.num_hidden_layers
        if self._model.config.audio_dual_ffn_layers:
            cache_config.num_hidden_layers += len(self._model.config.audio_dual_ffn_layers)
        # A list of KV caches for different lengths
        self.kv_caches = {
            length: StaticCache(
                config=cache_config,
                max_batch_size=1,
                max_cache_len=length,
                device=self._model.device,
                dtype=self._model.dtype,
            )
            for length in sorted(self._kv_cache_lengths)
        }
        # Capture CUDA graphs for each KV cache length
        if "cuda" in self._device:
            logger.info(f"Capturing CUDA graphs for each KV cache length")
            self._model.capture_model(self.kv_caches.values())

    def _prepare_kv_caches(self):
        for kv_cache in self.kv_caches.values():
            kv_cache.reset()

    @torch.inference_mode()
    def generate(
            self,
            messages,
            audio_ids,
            chunked_text,
            generation_chunk_buffer_size,
            temperature=1.0,
            top_k=50,
            top_p=0.95,
            ras_win_len=7,
            ras_win_max_num_repeat=2,
            seed=123,
            *args,
            **kwargs,
    ):
        logger.info(f"[DEBUG] Texto final que llega al modelo: {chunked_text}")

        if ras_win_len is not None and ras_win_len <= 0:
            ras_win_len = None
        sr = 24000
        audio_out_ids_l = []
        generated_audio_ids = []
        generation_messages = []
        for idx, chunk_text in tqdm.tqdm(
                enumerate(chunked_text), desc="Generating audio chunks", total=len(chunked_text)
        ):
            generation_messages.append(
                Message(
                    role="user",
                    content=chunk_text,
                )
            )
            chatml_sample = ChatMLSample(messages=messages + generation_messages)
            input_tokens, _, _, _ = prepare_chatml_sample(chatml_sample, self._tokenizer)
            postfix = self._tokenizer.encode(
                "<|start_header_id|>assistant<|end_header_id|>\n\n", add_special_tokens=False
            )
            input_tokens.extend(postfix)

            logger.info(f"========= Chunk {idx} Input =========")
            logger.info(self._tokenizer.decode(input_tokens))
            context_audio_ids = audio_ids + generated_audio_ids

            curr_sample = ChatMLDatasetSample(
                input_ids=torch.LongTensor(input_tokens),
                label_ids=None,
                audio_ids_concat=torch.concat([ele.cpu() for ele in context_audio_ids], dim=1)
                if context_audio_ids
                else None,
                audio_ids_start=torch.cumsum(
                    torch.tensor([0] + [ele.shape[1] for ele in context_audio_ids], dtype=torch.long), dim=0
                )
                if context_audio_ids
                else None,
                audio_waveforms_concat=None,
                audio_waveforms_start=None,
                audio_sample_rate=None,
                audio_speaker_indices=None,
            )

            batch_data = self._collator([curr_sample])
            batch = asdict(batch_data)
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.contiguous().to(self._device)

            if self._use_static_kv_cache:
                self._prepare_kv_caches()

            start_time = time.perf_counter()

            # Generate audio
            outputs = self._model.generate(
                **batch,
                max_new_tokens=self._max_new_tokens,
                use_cache=True,
                do_sample=True,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                past_key_values_buckets=self.kv_caches,
                ras_win_len=ras_win_len,
                ras_win_max_num_repeat=ras_win_max_num_repeat,
                stop_strings=["<|end_of_text|>", "<|eot_id|>"],
                tokenizer=self._tokenizer,
                seed=seed,
            )
            end_time = time.perf_counter()
            elapsed = end_time - start_time
            logger.info(f"⏱️ Chunk {idx} generated in {elapsed:.2f} seconds.")

            step_audio_out_ids_l = []
            for ele in outputs[1]:
                audio_out_ids = ele
                if self._config.use_delay_pattern:
                    audio_out_ids = revert_delay_pattern(audio_out_ids)
                step_audio_out_ids_l.append(audio_out_ids.clip(0, self._audio_tokenizer.codebook_size - 1)[:, 1:-1])
            audio_out_ids = torch.concat(step_audio_out_ids_l, dim=1)
            audio_out_ids_l.append(audio_out_ids)
            generated_audio_ids.append(audio_out_ids)

            generation_messages.append(
                Message(
                    role="assistant",
                    content=AudioContent(audio_url=""),
                )
            )
            if generation_chunk_buffer_size is not None and len(generated_audio_ids) > generation_chunk_buffer_size:
                generated_audio_ids = generated_audio_ids[-generation_chunk_buffer_size:]
                generation_messages = generation_messages[(-2 * generation_chunk_buffer_size):]

        logger.info(f"========= Final Text output =========")
        logger.info(self._tokenizer.decode(outputs[0][0]))
        concat_audio_out_ids = torch.concat(audio_out_ids_l, dim=1)

        # Fix MPS compatibility: detach and move to CPU before decoding
        if concat_audio_out_ids.device.type in ["mps", "cuda"]:
            concat_audio_out_ids_cpu = concat_audio_out_ids.detach().cpu()
        else:
            concat_audio_out_ids_cpu = concat_audio_out_ids

        concat_wv = self._audio_tokenizer.decode(concat_audio_out_ids_cpu.unsqueeze(0))[0, 0]
        text_result = self._tokenizer.decode(outputs[0][0])
        return concat_wv, sr, text_result


def prepare_generation_context(scene_prompt, ref_audio, ref_audio_in_system_message, audio_tokenizer, speaker_tags):
    """Prepare the context for generation.

    The context contains the system message, user message, assistant message, and audio prompt if any.
    """
    system_message = None
    messages = []
    audio_ids = []
    if ref_audio is not None:
        num_speakers = len(ref_audio.split(","))
        speaker_info_l = ref_audio.split(",")
        voice_profile = None
        if any([speaker_info.startswith("profile:") for speaker_info in ref_audio.split(",")]):
            ref_audio_in_system_message = True
        if ref_audio_in_system_message:
            speaker_desc = []
            for spk_id, character_name in enumerate(speaker_info_l):
                if character_name.startswith("profile:"):
                    if voice_profile is None:
                        # Assuming a profile.yaml exists in the same directory
                        try:
                            st.info(f"Cargando perfil de voz desde: {CURR_DIR}/voice_examples/profile.yaml")
                            with open(f"{CURR_DIR}/voice_examples/profile.yaml", "r", encoding="utf-8") as f:
                                voice_profile = yaml.safe_load(f)
                            character_desc = voice_profile["profiles"][character_name[len("profile:"):].strip()]
                            speaker_desc.append(f"SPEAKER{spk_id}: {character_desc}")
                        except FileNotFoundError:
                            st.error(f"Archivo de perfil no encontrado: {CURR_DIR}/voice_examples/profile.yaml")
                            return None, None
                        except KeyError:
                            st.error(
                                f"Perfil de voz '{character_name[len('profile:'):].strip()}' no encontrado en el archivo de perfil.")
                            return None, None
                else:
                    speaker_desc.append(f"SPEAKER{spk_id}: {AUDIO_PLACEHOLDER_TOKEN}")
            if scene_prompt:
                system_message = (
                        "Generate audio following instruction."
                        "\n\n"
                        f"<|scene_desc_start|>\n{scene_prompt}\n\n" + "\n".join(speaker_desc) + "\n<|scene_desc_end|>"
                )
            else:
                system_message = (
                        "Generate audio following instruction.\n\n"
                        + f"<|scene_desc_start|>\n"
                        + "\n".join(speaker_desc)
                        + "\n<|scene_desc_end|>"
                )
            system_message = _build_system_message_with_audio_prompt(system_message)
        else:
            if scene_prompt:
                system_message = Message(
                    role="system",
                    content=f"Generate audio following instruction.\n\n<|scene_desc_start|>\n{scene_prompt}\n<|scene_desc_end|>",
                )
        voice_profile = None
        for spk_id, character_name in enumerate(ref_audio.split(",")):
            if not character_name.startswith("profile:"):
                prompt_audio_path = os.path.join(f"{CURR_DIR}/voice_examples", f"{character_name}.wav")
                prompt_text_path = os.path.join(f"{CURR_DIR}/voice_examples", f"{character_name}.txt")
                if not os.path.exists(prompt_audio_path):
                    st.error(f"Archivo de audio de referencia '{prompt_audio_path}' no encontrado.")
                    return None, None
                if not os.path.exists(prompt_text_path):
                    st.error(f"Archivo de texto de referencia '{prompt_text_path}' no encontrado.")
                    return None, None
                with open(prompt_text_path, "r", encoding="utf-8") as f:
                    prompt_text = f.read().strip()
                audio_tokens = audio_tokenizer.encode(prompt_audio_path)
                audio_ids.append(audio_tokens)

                if not ref_audio_in_system_message:
                    messages.append(
                        Message(
                            role="user",
                            content=f"[SPEAKER{spk_id}] {prompt_text}" if num_speakers > 1 else prompt_text,
                        )
                    )
                    messages.append(
                        Message(
                            role="assistant",
                            content=AudioContent(
                                audio_url=prompt_audio_path,
                            ),
                        )
                    )
    else:
        if len(speaker_tags) > 1:
            # By default, we just alternate between male and female voices
            speaker_desc_l = []

            for idx, tag in enumerate(speaker_tags):
                if idx % 2 == 0:
                    speaker_desc = f"feminine"
                else:
                    speaker_desc = f"masculine"
                speaker_desc_l.append(f"{tag}: {speaker_desc}")

            speaker_desc = "\n".join(speaker_desc_l)
            scene_desc_l = []
            if scene_prompt:
                scene_desc_l.append(scene_prompt)
            scene_desc_l.append(speaker_desc)
            scene_desc = "\n\n".join(scene_desc_l)

            system_message = Message(
                role="system",
                content=f"{MULTISPEAKER_DEFAULT_SYSTEM_MESSAGE}\n\n<|scene_desc_start|>\n{scene_desc}\n<|scene_desc_end|>",
            )
        else:
            system_message_l = ["Generate audio following instruction."]
            if scene_prompt:
                system_message_l.append(f"<|scene_desc_start|>\n{scene_prompt}\n<|scene_desc_end|>")
            system_message = Message(
                role="system",
                content="\n\n".join(system_message_l),
            )
    if system_message:
        messages.insert(0, system_message)
    return messages, audio_ids


@st.cache_resource
def get_model_client(
        model_path,
        audio_tokenizer_path,
        device,
        use_static_kv_cache,
        max_new_tokens,
):
    # For MPS, use CPU for audio tokenizer due to embedding operation limitations
    audio_tokenizer_device = "cpu" if device == "mps" else device
    audio_tokenizer = load_higgs_audio_tokenizer(audio_tokenizer_path, device=audio_tokenizer_device)

    # Disable static KV cache on MPS since it relies on CUDA graphs
    if "mps" in device and use_static_kv_cache:
        use_static_kv_cache = False
        st.warning("El caché estático KV se ha desactivado para el dispositivo MPS.")

    device_id = int(device.split(":")[-1]) if "cuda" in device else None

    return HiggsAudioModelClient(
        model_path=model_path,
        audio_tokenizer=audio_tokenizer,
        device=device,
        device_id=device_id,
        max_new_tokens=max_new_tokens,
        use_static_kv_cache=use_static_kv_cache,
    )


def main():
    logger.info("Higgs Audio App arrancando…")
    print("Higgs Audio App lista. Implementa aquí tu lógica de arranque.")

    st.title("Generador de Audio HiggsAudio")

    st.markdown("Crea audio a partir de texto utilizando el modelo HiggsAudio.")

    # --- Configuración del Modelo y Dispositivo ---
    st.header("🎙️ Generar narración completa con Higgs Audio V2")

    col1, col2 = st.columns(2)
    with col1:
        model_path = st.text_input(
            "Ruta del Modelo",
            value="bosonai/higgs-audio-v2-generation-3B-base",
            help="Ruta del modelo HiggsAudio pre-entrenado."
        )
    with col2:
        device_option = st.selectbox(
            "Dispositivo",
            options=["auto", "cuda", "mps", "cpu"],
            index=0,
            help="Selecciona el dispositivo para la generación (CUDA, MPS o CPU)."
        )

    # Lógica para determinar el dispositivo final
    if device_option == "auto":
        if torch.cuda.is_available():
            device = "cuda:0"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = device_option

    st.info(f"Dispositivo de generación seleccionado: **{device}**")

    # --- Parámetros de Entrada ---
    st.header("Entradas de Texto")

    transcript = st.text_area(
        "Transcripción",
        value="""[SPEAKER0] Hello, I'm just calling to confirm our meeting for tomorrow.
[SPEAKER1] Hi there! Yes, that sounds perfect. What time were we thinking?
[SPEAKER0] I was hoping for around 10 AM.
[SPEAKER1] That works for me! See you then.
[SPEAKER0] Great, see you then.""",
        height=250,
        help="Introduce el texto que quieres convertir en audio. Usa etiquetas [SPEAKER*] para múltiples voces."
    )

    st.subheader("Prompts")
    col3, col4 = st.columns(2)
    with col3:
        scene_prompt = st.text_area(
            "Prompt de Escena",
            value="quiet indoor",
            help="Describe el ambiente de la escena (ej: 'quiet indoor', 'busy street').",
        )
    with col4:
        ref_audio = st.text_input(
            "Audio de Referencia",
            value="belinda,chadwick",
            help="Nombres de prompts de voz separados por comas. (ej: 'belinda,chadwick'). Asume que los archivos existen en la carpeta `voice_prompts`."
        )

    ref_audio_in_system_message = st.checkbox(
        "Incluir prompts de voz en el mensaje del sistema",
        value=True,
        help="Si se marca, la descripción del prompt de voz se incluirá en el mensaje del sistema."
    )

    # --- Opciones de Generación Avanzadas ---
    st.header("Opciones de Generación")

    with st.expander("Parámetros avanzados"):
        col5, col6 = st.columns(2)
        with col5:
            max_new_tokens = st.number_input(
                "Máx. Tokens Nuevos",
                min_value=128,
                max_value=4096,
                value=2048,
                help="Número máximo de tokens de audio a generar."
            )
            temperature = st.slider(
                "Temperatura",
                min_value=0.0, max_value=2.0, value=1.0, step=0.05,
                help="Modula las probabilidades de los tokens generados."
            )
            top_k = st.slider(
                "Top K",
                min_value=1, max_value=200, value=50, step=1,
                help="Número de tokens con mayor probabilidad para el filtrado Top-K."
            )
            top_p = st.slider(
                "Top P",
                min_value=0.0, max_value=1.0, value=0.95, step=0.05,
                help="Tokens más probables cuyas probabilidades suman 'top_p'."
            )

        with col6:
            chunk_method = st.selectbox(
                "Método de segmentación",
                options=[None, "speaker", "word"],
                index=1,
                format_func=lambda x: "None" if x is None else x,
                help="Método para dividir la transcripción en fragmentos."
            )
            chunk_max_word_num = st.number_input(
                "Máx. Palabras por Segmento",
                min_value=1, value=200, help="Solo para el método 'word'."
            )
            chunk_max_num_turns = st.number_input(
                "Máx. Turnos por Segmento",
                min_value=1, value=1, help="Solo para el método 'speaker'."
            )
            seed = st.number_input(
                "Semilla Aleatoria",
                value=123,
                help="Semilla para la generación reproducible."
            )
            use_static_kv_cache = st.checkbox(
                "Usar Caché KV Estática",
                value=True,
                help="Mejora la velocidad en GPUs NVIDIA (no funciona en MPS)."
            )

    # === Parámetros de entrada ===
    voice_path = st.text_input("📁 Carpeta de voz", value="voice_examples/socrates")
    voice_name = st.text_input("🧑 Nombre de la voz", value="socrates")
    preset = st.selectbox("🎛️ Preset", options=["narration", "conversational", "emotional"], index=0)
    language = st.selectbox("🌐 Idioma", options=["es", "en", "fr", "de"], index=0)
    scene_prompt = st.text_area("🎬 Prompt de escena", value="A thoughtful conversation with Socrates", height=100)

    texto_largo = st.text_area("📝 Texto completo a narrar", height=600)

    output_dir = "resultado_narracion"
    os.makedirs(output_dir, exist_ok=True)

    @st.cache_resource
    def load_model_client():
        with st.spinner("🔧 Cargando modelo Higgs Audio..."):
            return get_model_client(
                model_path=model_path,
                audio_tokenizer_path="bosonai/higgs-audio-v2-tokenizer",
                device=device,
                use_static_kv_cache=use_static_kv_cache,
                max_new_tokens=max_new_tokens
            )

    model_client = load_model_client()
    if st.button("🔁 Narrar todo el texto"):
        if not texto_largo.strip():
            st.warning("Introduce un texto para narrar.")
        else:
            with st.spinner("Generando narración completa..."):
                resultados = generar_narracion_completa(
                    texto_largo=texto_largo,
                    voice_path=voice_path,
                    voice_name=voice_name,
                    preset=preset,
                    language=language,
                    scene_prompt=scene_prompt,
                    output_dir=output_dir,
                    pause_duration=0.5,  # Pausa entre bloques
                    model=model_client
                )
            tabla_estado = st.empty()
            tabla_estado.dataframe({
                "Bloque": list(range(1, len(resultados["duraciones"]) + 1)),
                "Duración (s)": resultados["duraciones"]
            })
            st.success("✅ Narración completada.")
            st.audio(resultados["ruta_audio_final"], format="audio/wav")

            st.subheader("⏱️ Duraciones por bloque:")
            for i, dur in enumerate(resultados["duraciones"]):
                st.write(f"🔊 Bloque {i + 1}: {dur:.2f} segundos")
            duracion_total = sum(resultados["duraciones"])
            st.info(f"🎧 Duración total de la narración: {duracion_total:.2f} segundos")
            st.download_button("⬇️ Descargar audio final", data=open(resultados["ruta_audio_final"], "rb"),
                               file_name="narracion_completa.wav")
    if st.button("Generar Audio", use_container_width=True):
        if not transcript:
            st.error("Por favor, introduce una transcripción para generar audio.")
            return

        with st.spinner("Cargando modelo y generando audio..."):
            # Lógica principal del script original, adaptada

            model_client = get_model_client(
                model_path=model_path,
                audio_tokenizer_path="bosonai/higgs-audio-v2-tokenizer",
                device=device,
                use_static_kv_cache=use_static_kv_cache,
                max_new_tokens=max_new_tokens
            )

            pattern = re.compile(r"\[(SPEAKER\d+)\]")
            speaker_tags = sorted(set(pattern.findall(transcript)))

            # Normalización del texto
            processed_transcript = normalize_chinese_punctuation(transcript)
            processed_transcript = processed_transcript.replace("(", " ")
            processed_transcript = processed_transcript.replace(")", " ")
            processed_transcript = processed_transcript.replace("°F", " degrees Fahrenheit")
            processed_transcript = processed_transcript.replace("°C", " degrees Celsius")
            for tag, replacement in [
                ("[laugh]", "<SE>[Laughter]</SE>"),
                ("[humming start]", "<SE_s>[Humming]</SE_s>"),
                ("[humming end]", "<SE_e>[Humming]</SE_e>"),
                ("[music start]", "<SE_s>[Music]</SE_s>"),
                ("[music end]", "<SE_e>[Music]</SE_e>"),
                ("[music]", "<SE>[Music]</SE>"),
                ("[sing start]", "<SE_s>[Singing]</SE_s>"),
                ("[sing end]", "<SE_e>[Singing]</SE_e>"),
                ("[applause]", "<SE>[Applause]</SE>"),
                ("[cheering]", "<SE>[Cheering]</SE>"),
                ("[cough]", "<SE>[Cough]</SE>"),
            ]:
                processed_transcript = processed_transcript.replace(tag, replacement)
            lines = processed_transcript.split("\n")
            processed_transcript = "\n".join([" ".join(line.split()) for line in lines if line.strip()])
            processed_transcript = processed_transcript.strip()
            if not any([processed_transcript.endswith(c) for c in
                        [".", "!", "?", ",", ";", '"', "'", "</SE_e>", "</SE>"]]):
                processed_transcript += "."

            messages, audio_ids = prepare_generation_context(
                scene_prompt=scene_prompt,
                ref_audio=ref_audio if ref_audio else None,
                ref_audio_in_system_message=ref_audio_in_system_message,
                audio_tokenizer=model_client._audio_tokenizer,
                speaker_tags=speaker_tags,
            )

            if messages is None:  # Si hay un error en los prompts, el prepare_generation_context devuelve None
                st.error("Error en la preparación del contexto de generación. Revisa los prompts de voz y escena.")
                return

            chunked_text = prepare_chunk_text(
                processed_transcript,
                chunk_method=chunk_method,
                chunk_max_word_num=chunk_max_word_num,
                chunk_max_num_turns=chunk_max_num_turns,
            )

            concat_wv, sr, text_output = model_client.generate(
                messages=messages,
                audio_ids=audio_ids,
                chunked_text=chunked_text,
                generation_chunk_buffer_size=None,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                ras_win_len=7,
                ras_win_max_num_repeat=2,
                seed=seed,
            )

        st.success("¡Audio generado con éxito!")

        # Reproducir y descargar el audio
        buffer = io.BytesIO()
        sf.write(buffer, concat_wv, sr, format='WAV')
        st.audio(buffer.getvalue(), format='audio/wav', sample_rate=sr)
        st.download_button(
            label="Descargar audio (.wav)",
            data=buffer.getvalue(),
            file_name="generation.wav",
            mime="audio/wav"
        )

        # Mostrar la transcripción final del modelo
        st.subheader("Transcripción final del modelo")
        st.text_area("Texto de salida:", value=text_output, height=200, disabled=True)


if __name__ == "__main__":
    main()
