import click
import soundfile as sf
import langid
import jieba
import os
import re
import copy
import torchaudio
import tqdm
import yaml
import streamlit as st
import io
import json
import numpy as np
import torch
import time
from datetime import datetime
from typing import List, Optional
from loguru import logger
from transformers import AutoConfig, AutoTokenizer
from transformers.cache_utils import StaticCache

# Importaciones de boson_multimodal
# NOTA: Asegúrate de tener boson-multimodal instalado y configurado
from boson_multimodal.serve.serve_engine import HiggsAudioServeEngine, HiggsAudioResponse
from boson_multimodal.data_types import Message, ChatMLSample, AudioContent, TextContent
from boson_multimodal.model.higgs_audio import HiggsAudioConfig, HiggsAudioModel
from boson_multimodal.data_collator.higgs_audio_collator import HiggsAudioSampleCollator
from boson_multimodal.audio_processing.higgs_audio_tokenizer import load_higgs_audio_tokenizer
from boson_multimodal.dataset.chatml_dataset import (
    ChatMLDatasetSample,
    prepare_chatml_sample,
)
from boson_multimodal.model.higgs_audio.utils import revert_delay_pattern
from dataclasses import asdict

# Constantes
AUDIO_PLACEHOLDER_TOKEN = "<|__AUDIO_PLACEHOLDER__|>"
CURR_DIR = os.path.dirname(os.path.abspath(__file__))
# Asumiendo una tasa de muestreo estándar, pero podría ser un parámetro configurable.
DEFAULT_SAMPLE_RATE = 24000


# --- FUNCIONES AUXILIARES ---

def save_audio(audio: np.ndarray, output_path: str, samplerate: int = DEFAULT_SAMPLE_RATE):
    """
    Guarda un array de audio como un archivo WAV en formato PCM 16 bits.
    """
    audio = np.array(audio).astype(np.float32)
    sf.write(output_path, audio, samplerate)


def concatenate_audios_with_pause(audio_paths: List[str], pause_duration: float = 0.5, samplerate: int = DEFAULT_SAMPLE_RATE) -> np.ndarray:
    """
    Concatena varios archivos WAV de una lista, introduciendo una pausa de silencio entre ellos.
    
    Args:
        audio_paths (List[str]): Lista de rutas a archivos de audio .wav.
        pause_duration (float): Duración de la pausa en segundos.
        samplerate (int): Tasa de muestreo del audio.

    Returns:
        np.ndarray: Un array numpy con el audio concatenado.
    """
    if not audio_paths:
        return np.array([], dtype=np.float32)

    combined_audio = []
    
    # Crea un array de silencio una vez para reutilizarlo
    silence = np.zeros(int(samplerate * pause_duration), dtype=np.float32)

    for i, path in enumerate(audio_paths):
        audio, sr = sf.read(path)
        if sr != samplerate:
            raise ValueError(f"La tasa de muestreo de {path} es {sr}, se esperaba {samplerate}")
        
        combined_audio.append(audio)

        # Añade la pausa solo si no es el último archivo
        if i < len(audio_paths) - 1:
            combined_audio.append(silence)

    # Si hay audios, concatenarlos. Si la lista está vacía, devuelve un array vacío.
    if combined_audio:
        return np.concatenate(combined_audio, axis=0)
    else:
        return np.array([], dtype=np.float32)


def dividir_texto_por_pausa(texto_largo: str, max_len: int = 300) -> List[str]:
    """
    Divide un texto largo en bloques más pequeños basados en la puntuación,
    asegurando que cada bloque no exceda una longitud máxima.
    """
    frases = re.split(r'(?<=[.!?…])\s+', texto_largo.strip())
    bloques = []
    bloque_actual = ""
    for frase in frases:
        if len(bloque_actual) + len(frase) + 1 <= max_len:
            bloque_actual += " " + frase if bloque_actual else frase
        else:
            if bloque_actual:
                bloques.append(bloque_actual.strip())
            bloque_actual = frase
    if bloque_actual:
        bloques.append(bloque_actual.strip())
    return bloques


def normalize_chinese_punctuation(text: str) -> str:
    """
    Convierte la puntuación china a la equivalente en inglés.
    Esta función se mantiene por si la necesitas para un uso multilingüe.
    """
    chinese_to_english_punct = {
        "：": ":", "；": ";", "？": "?", "！": "!", "（": "(", "）": ")",
        "【": "[", "】": "]", "《": "<", "》": ">", "“": '"', "”": '"',
        "‘": "'", "’": "'", "、": ",", "—": "-", "…": "...", "·": ".",
        "「": '"', "」": '"', "『": '"', "』": '"', "，": ",", "。": "."
    }
    for zh_punct, en_punct in chinese_to_english_punct.items():
        text = text.replace(zh_punct, en_punct)
    return text


def _build_system_message_with_audio_prompt(system_message):
    contents = []
    while AUDIO_PLACEHOLDER_TOKEN in system_message:
        loc = system_message.find(AUDIO_PLACEHOLDER_TOKEN)
        contents.append(TextContent(system_message[:loc]))
        contents.append(AudioContent(audio_url=""))
        system_message = system_message[loc + len(AUDIO_PLACEHOLDER_TOKEN) :]
    if len(system_message) > 0:
        contents.append(TextContent(system_message))
    ret = Message(
        role="system",
        content=contents,
    )
    return ret


# --- CLASE CLIENTE DEL MODELO ---

class HiggsAudioModelClient:
    def __init__(self, model_path, audio_tokenizer, device=None, device_id=None, **kwargs):
        if device_id is not None:
            self._device = f"cuda:{device_id}"
        else:
            if device is not None:
                self._device = device
            else:
                if torch.cuda.is_available():
                    self._device = "cuda:0"
                elif torch.backends.mps.is_available():
                    self._device = "mps"
                else:
                    self._device = "cpu"

        logger.info(f"Using device: {self._device}")

        # Se asume que audio_tokenizer es una ruta o un objeto ya cargado
        if isinstance(audio_tokenizer, str):
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

        # Configuración del tokenizer y el collator
        self._tokenizer = AutoTokenizer.from_pretrained(model_path)
        self._config = AutoConfig.from_pretrained(model_path)
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
        # Puedes añadir aquí la lógica de KV cache si la necesitas, como en tu código original
        # self.kv_caches = None
        # ...

    @torch.inference_mode()
    def generate(self, messages: List[Message], audio_ids, chunked_text: List[str], **kwargs):
        """
        Genera audio a partir de un texto, dividiéndolo en chunks.
        """
        # La lógica de generación por chunks se ha movido aquí para mayor cohesión
        sr = DEFAULT_SAMPLE_RATE
        audio_out_ids_l = []
        generated_audio_ids = []
        generation_messages = []

        for idx, chunk_text in tqdm.tqdm(enumerate(chunked_text), desc="Generating audio chunks", total=len(chunked_text)):
            generation_messages.append(Message(role="user", content=chunk_text))
            chatml_sample = ChatMLSample(messages=messages + generation_messages)
            input_tokens, _, _, _ = prepare_chatml_sample(chatml_sample, self._tokenizer)
            postfix = self._tokenizer.encode("<|start_header_id|>assistant<|end_header_id|>\n\n", add_special_tokens=False)
            input_tokens.extend(postfix)
            
            context_audio_ids = audio_ids + generated_audio_ids
            curr_sample = ChatMLDatasetSample(
                input_ids=torch.LongTensor(input_tokens),
                audio_ids_concat=torch.concat([ele.cpu() for ele in context_audio_ids], dim=1) if context_audio_ids else None,
                audio_ids_start=torch.cumsum(torch.tensor([0] + [ele.shape[1] for ele in context_audio_ids], dtype=torch.long), dim=0) if context_audio_ids else None,
                audio_sample_rate=sr
            )
            batch_data = self._collator([curr_sample])
            batch = asdict(batch_data)
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.contiguous().to(self._device)

            outputs = self._model.generate(
                **batch,
                max_new_tokens=kwargs.get("max_new_tokens", 2048),
                use_cache=True,
                do_sample=True,
                temperature=kwargs.get("temperature", 1.0),
                top_k=kwargs.get("top_k", 50),
                top_p=kwargs.get("top_p", 0.95),
                ras_win_len=kwargs.get("ras_win_len", 7),
                ras_win_max_num_repeat=kwargs.get("ras_win_max_num_repeat", 2),
                stop_strings=["<|end_of_text|>", "<|eot_id|>"],
                tokenizer=self._tokenizer,
                seed=kwargs.get("seed", 123),
            )

            step_audio_out_ids_l = []
            for ele in outputs[1]:
                audio_out_ids = ele
                if self._config.use_delay_pattern:
                    audio_out_ids = revert_delay_pattern(audio_out_ids)
                step_audio_out_ids_l.append(audio_out_ids.clip(0, self._audio_tokenizer.codebook_size - 1)[:, 1:-1])
            
            audio_out_ids = torch.concat(step_audio_out_ids_l, dim=1)
            audio_out_ids_l.append(audio_out_ids)
            generated_audio_ids.append(audio_out_ids)
            generation_messages.append(Message(role="assistant", content=AudioContent(audio_url="")))

        concat_audio_out_ids = torch.concat(audio_out_ids_l, dim=1)
        if concat_audio_out_ids.device.type in ["mps", "cuda"]:
            concat_audio_out_ids_cpu = concat_audio_out_ids.detach().cpu()
        else:
            concat_audio_out_ids_cpu = concat_audio_out_ids
        
        concat_wv = self._audio_tokenizer.decode(concat_audio_out_ids_cpu.unsqueeze(0))[0, 0]
        text_result = self._tokenizer.decode(outputs[0][0])
        return concat_wv, sr, text_result


# --- LÓGICA DE GENERACIÓN DE AUDIO PRINCIPAL ---

def generar_audio_higgs_single(
    text: str,
    voice_path: str, # Este parámetro no parece usarse en tu lógica de generación
    voice_name: str, # Asumimos que es un nombre que el modelo entiende
    scene_prompt: str,
    model: HiggsAudioModelClient,
    temperature: float = 1.0,
    top_k: int = 50,
    top_p: float = 0.95,
    ras_win_len: int = 7,
    ras_win_max_num_repeat: int = 2,
    seed: int = 123
):
    """
    Función para generar audio para un solo bloque de texto.
    """
    # === Normalizar texto ===
    processed_text = text.replace("(", " ").replace(")", " ")
    processed_text = processed_text.replace("°F", " degrees Fahrenheit").replace("°C", " degrees Celsius")
    processed_text = processed_text.strip()
    if not any(processed_text.endswith(c) for c in [".", "!", "?", ",", ";", '"', "'", "</SE_e>", "</SE>"]):
        processed_text += "."

    # Este es el código que necesitas para definir 'prepare_generation_context'
    # Esta función debería crear los mensajes iniciales para el modelo.
    # Como no la proporcionaste, estoy usando una implementación de ejemplo.
    def prepare_generation_context(scene_prompt, ref_audio, ref_audio_in_system_message, audio_tokenizer, speaker_tags):
        messages = []
        audio_ids = []
        if ref_audio_in_system_message:
            system_message = f"You are an AI assistant designed to convert text into speech. The user's audio input is a voice reference. Your task is to generate audio that sounds like the user's voice for a given text. The audio context is as follows: {scene_prompt}. The reference voice is {ref_audio}. {AUDIO_PLACEHOLDER_TOKEN}"
            messages.append(_build_system_message_with_audio_prompt(system_message))
        
        # Aquí deberías cargar el audio de referencia y tokenizarlo
        # Esto es un placeholder, ya que el código de carga del audio real no está.
        # Por ahora, solo devolvemos listas vacías.
        # Puedes añadir tu propia lógica aquí.
        # e.g., audio_data, sr = torchaudio.load(voice_path)
        # audio_ids = audio_tokenizer.tokenize(audio_data, sr)
        
        return messages, audio_ids


    # === Preparar contexto para una sola voz ===
    speaker_tags = ["SPEAKER0"]
    messages, audio_ids = prepare_generation_context(
        scene_prompt=scene_prompt,
        ref_audio=voice_name,
        ref_audio_in_system_message=True,
        audio_tokenizer=model._audio_tokenizer,
        speaker_tags=speaker_tags,
    )
    
    if not processed_text.strip():
        raise ValueError("❌ Texto procesado vacío en generación.")
    
    chunked_text = [f"SPEAKER0: {processed_text}"]
    st.info(f"Texto del chunk: {chunked_text}")    

    concat_wav, sr, _ = model.generate(
        messages=messages,
        audio_ids=audio_ids,
        chunked_text=chunked_text,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        ras_win_len=ras_win_len,
        ras_win_max_num_repeat=ras_win_max_num_repeat,
        seed=seed,
    )
    return concat_wav, sr


def generar_narracion_completa(
    texto_largo: str,
    voice_path: str,
    voice_name: str,
    preset: str,
    language: str,
    scene_prompt: str,
    output_dir: str,
    model: HiggsAudioModelClient,
    pause_duration: float = 0.5
):
    """
    Función principal para generar una narración completa a partir de un texto largo.
    """
    from datetime import datetime
    
    timestamp = datetime.now().strftime("%d%m%Y_%H%M%S")
    carpeta_salida = os.path.join(output_dir, timestamp)
    os.makedirs(carpeta_salida, exist_ok=True)
    
    log_panel = st.sidebar.expander("📋 Log de generación en tiempo real", expanded=True)
    log_panel.markdown("Iniciando generación...")
    
    bloques = dividir_texto_por_pausa(texto_largo)
    audios = []
    duraciones = []

    st.subheader("🧾 Frases detectadas:")
    st.markdown(f"Se han detectado **{len(bloques)} bloques** de texto para generar.")
    with st.expander("Ver bloques detectados"):
        for i, b in enumerate(bloques):
            st.write(f"**Bloque {i+1}:** {b}")

    for i, bloque in enumerate(bloques):
        nombre_archivo = f"{timestamp}_frase{i+1}.wav"
        ruta_salida = os.path.join(carpeta_salida, nombre_archivo)
        
        st.markdown(f"🔄 Generando bloque `{i+1}/{len(bloques)}`:")
        status_text = st.empty()
        progress_bar = st.progress(i / len(bloques))
        
        t0 = time.time()
        audio, sr = generar_audio_higgs_single(
            text=bloque,
            voice_path=voice_path,
            voice_name=voice_name,
            scene_prompt=scene_prompt,
            model=model,
            language=language, # Este parámetro no se usa dentro de la función
            preset=preset # Este parámetro no se usa dentro de la función
        )
        save_audio(audio, ruta_salida, sr)
        
        t1 = time.time()
        duraciones.append(t1 - t0)
        log_panel.markdown(f"""
**🟢 Bloque {i+1}/{len(bloques)}**
- ⏱️ Tiempo: `{t1 - t0:.2f} s`
- 🗣️ Texto: `{bloque[:100]}{"..." if len(bloque) > 100 else ""}`
""")
        status_text.markdown(f"✅ Bloque `{i+1}` generado en **{t1 - t0:.2f} s**")
        progress_bar.progress((i + 1) / len(bloques))
        audios.append(ruta_salida)

    ruta_audio_final = os.path.join(carpeta_salida, f"{timestamp}_completo.wav")
    audio_final = concatenate_audios_with_pause(audios, pause_duration=pause_duration)
    save_audio(audio_final, ruta_audio_final)
    
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


# --- ESTRUCTURA DE LA APLICACIÓN STREAMLIT ---

# Esto es un esqueleto de cómo usarías las funciones anteriores
# en una aplicación de Streamlit.
def main():
    st.title("🗣️ Generación de Voz con Higgs-audio")
    
    st.sidebar.header("Configuración del Modelo")
    model_path = st.sidebar.text_input("Ruta del Modelo", "boson/Higgs-audio-3B")
    audio_tokenizer_path = st.sidebar.text_input("Ruta del Tokenizer de Audio", "boson/Higgs-audio-tokenizer")
    
    # Intenta cargar el modelo solo una vez
    @st.cache_resource
    def get_model_client(model_path, audio_tokenizer_path):
        try:
            return HiggsAudioModelClient(model_path, audio_tokenizer_path)
        except Exception as e:
            st.error(f"Error al cargar el modelo: {e}")
            return None
            
    model_client = get_model_client(model_path, audio_tokenizer_path)
    
    if model_client is None:
        return

    st.header("Generación de Narración")
    texto_largo = st.text_area("Introduce el texto para generar el audio:", height=200)
    
    # Estos parámetros se pasarían a las funciones
    scene_prompt = st.text_input("Prompt de la escena:", "Un hombre hablando en un entorno tranquilo.")
    voice_name = st.text_input("Nombre de la voz:", "SPEAKER0")
    output_dir = "output_audios"
    
    if st.button("Generar Audio"):
        if not texto_largo:
            st.warning("Por favor, introduce un texto para generar el audio.")
            return

        with st.spinner("Generando audio..."):
            try:
                resultados = generar_narracion_completa(
                    texto_largo=texto_largo,
                    voice_path=None, # Este parámetro se podría quitar o usar para cargar un audio de referencia
                    voice_name=voice_name,
                    preset="default", # No se usa, podría ser un parámetro de entrada
                    language="es", # No se usa, podría ser un parámetro de entrada
                    scene_prompt=scene_prompt,
                    output_dir=output_dir,
                    model=model_client
                )
                
                st.success("¡Audio generado con éxito!")
                st.audio(resultados['ruta_audio_final'], format='audio/wav')
                
                st.download_button(
                    label="Descargar Audio Completo",
                    data=open(resultados['ruta_audio_final'], 'rb').read(),
                    file_name=os.path.basename(resultados['ruta_audio_final']),
                    mime="audio/wav"
                )

                st.subheader("Archivos Individuales")
                for path in resultados['archivos_individuales']:
                    st.audio(path, format='audio/wav', caption=os.path.basename(path))

            except Exception as e:
                st.error(f"Se ha producido un error durante la generación: {e}")

if __name__ == "__main__":
    main()

