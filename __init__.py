from typing import Optional, Tuple
from numpy import ndarray

from TTS.inference import Synthesizer
from chatbot.chatbot import ChatBot

synthesizer = Synthesizer("./TTS/models/Tacotron2/ckpt_300000", "./TTS/models/hifigan/")
chatbot = ChatBot([], tokenizer_path="./chatbot/tokenizer", embedding_model_path="./chatbot/models/embedding_model_state.pt")

def synthesize(text: str,
               wav_path: Optional[str] = "./res/res_wav.wav",
               plot_path: Optional[str] = "./res/res_plot.png",
               play_audio: bool = False) -> Tuple[ndarray, int, float]:
    """
    synthesize audio from input text.\n
    if input text includes non-korean(likes number, english, etc)text, that text will be change to korean or disappear.

    max auido length is defined by `n_frames_per_step * time(len(mel_output)) >= max_decoder_ratio * input_length(alignment.shape[1])`.\n
    most of the time, the audio having that length is having wrong sound at last(case of 'Warning: Reached max decoder steps.').\n
    and it means gate of tacotron 2 couldn't predict the end of sounds.\n
    in this case, you can try giving short length text or something difference text.

    :param text: text for synthesize.
    :param wav_path: path for saving result audio in wav form. has to end with ".wav". if None, do NOT save result audio.
    :param plot_path: path for saving result audio plot in png form. has to end with ".png". if None, do NOT save plot of result audio.
    :param play_audio: condition for playing audio in present audio output device.

    :return: audio(ndarray(Time, ), audio data), sampling_rate(int, audio sampling rate), duration(float, synthesize duration(sec))
    """
    audio, sr, duration = synthesizer.synthesize(text)
    if wav_path:
        synthesizer.save_wave(wav_path, audio)
    if plot_path:
        synthesizer.save_plot(plot_path)
    if play_audio:
        synthesizer.play_audio(audio)
    return audio, sr, duration


if __name__ == '__main__':
    synthesize("에이아이 서비스의 티티에스 모델입니다.")
