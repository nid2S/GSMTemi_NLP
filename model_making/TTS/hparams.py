import os
import torch

class hparams:
    ############################
    # train hyper parameter    #
    ############################
    seed = 7777
    # train
    epochs = 500
    batch_size = 16
    iters_per_checkpoint = 10000
    prioritize_loss = False

    # Model
    model_type = 'multi-speaker'  # [single, multi-speaker]
    speaker_embedding_size = 16
    pad_token_idx = 0
    num_speakers = 1

    # device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_workers = torch.cuda.device_count() if torch.cuda.is_available() else os.cpu_count() - 1

    # path
    initial_data_greedy = True
    initial_phase_step = 8000  # main greey factor가 적용되기 까지의 step
    main_data_greedy_factor = 0  # main data에 적용될 가중치
    main_data = ['']  # 가중치가 적용될 main data

    training_files = '../../data/TTS/train.txt'
    validation_files = '../../data/TTS/val.txt'
    model_output_path = '../../models/TTS/Tacotron2/ckpt'
    logging_dir = '../../models/TTS/Tacotron2/log'
    last_ckpt = f"{model_output_path}/checkpoint_{max(int(ckpt.split('_')[1]) for ckpt in os.listdir(model_output_path))}" \
        if os.listdir(model_output_path) else ""

    # layer params
    adam_beta1 = 0.9
    adam_beta2 = 0.999
    embedding_size = 512
    dropout_prob = 0.5
    attention_size = 128
    attention_dim = 256

    # Learning rate schedule
    tacotron_decay_learning_rate = True    # boolean, determines if the learning rate will follow an exponential decay
    tacotron_start_decay = 40000           # Step at which learning decay starts
    tacotron_decay_steps = 18000           # Determines the learning rate decay slope (UNDER TEST)
    tacotron_decay_rate = 0.5              # learning rate decay rate (UNDER TEST)
    tacotron_initial_learning_rate = 1e-3  # starting learning rate
    tacotron_final_learning_rate = 1e-4    # minimal learning rate

    ############################
    # tacotron hyper parameter #
    ############################
    cleaners = 'korean_cleaners'  # 'korean_cleaners' or 'english_cleaners'
    skip_path_filter = False      # npz파일에서 불필요한 것을 거르는 작업을 할지 말지 결정. receptive_field 보다 짧은 data를 걸러야 하기 때문에 해 줘야 한다.
    use_lws = False
    # Audio
    sample_rate = 22050
    fft_size = 2048     # n_fft(frame_shift_ms). 주로 1024로 되어있는데, tacotron에서 2048사용
    hop_size = 300      # frame_shift_ms = 12.5ms  | shift can be specified by either hop_size(우선) or fft_size
    win_size = 1200     # 50ms
    num_mels = 80
    num_freq = fft_size // 2 + 1
    frame_shift_ms = hop_size * 1000.0 / sample_rate  # hop_size=  sample_rate *  frame_shift_ms / 1000
    frame_length_ms = win_size * 1000.0 / sample_rate
    # Spectrogram Pre-Emphasis
    # Lfilter: Reduce spectrogram noise and helps model certitude levels. Also allows for better G&L phase reconstruction
    preemphasize = True  # whether to apply filter
    preemphasis = 0.97
    min_level_db = -100
    ref_level_db = 20
    max_abs_value = 4.           # max absolute value of data. If symmetric, data will be [-max, max] else [0, max] (Must not be too big to avoid gradient explosion, not too small for fast convergence)
    symmetric_mels = True        # Whether to scale the data to be symmetric around 0. (Also multiplies the output range by 2, faster and cleaner convergence)
    signal_normalization = True  # Whether to normalize mel spectrograms to some predefined range (following below parameters)
    allow_clipping_in_normalization = True  # Only relevant if mel_normalization = True

    # rescaling
    rescaling = True
    rescaling_max = 0.999
    
    # M-AILABS (and other datasets) trim params (there parameters are usually correct for any data, but definitely must be tuned for specific speakers)
    trim_silence = True  # Whether to clip silence in Audio (at beginning and end of audio only, not the middle)
    trim_fft_size = 512
    trim_hop_size = 128
    trim_top_db = 23

    # For cases of OOM (Not really recommended, only use if facing unsolvable OOM errors, also consider clipping your samples to smaller chunks)
    clip_mels_length = True
    # Only relevant when clip_mels_length = True, please only use after trying output_per_steps=3 and still getting OOM errors.
    max_mel_frames = 1000
    l2_regularization_strength = 0  # Coefficient in the L2 regularization.
    sample_size = 9000              # Concatenate and cut audio samples to this many samples
    silence_threshold = 0           # Volume threshold below which to trim the start and the end from the training set samples. e.g. 2
    filter_width = 3
    gc_channels = 32                # global_condition_vector의 차원. 이것읖 지정함으로써, global conditioning을 모델에 반영하라는 의미가 된다.

    input_type = "raw"              # 'mulaw-quantize', 'mulaw', 'raw',   mulaw, raw 2가지는 scalar input
    scalar_input = True             # input_type과 맞아야 함.
    dilations = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    residual_channels = 128
    dilation_channels = 256
    quantize_channels = 256
    quantization_channels = 256
    out_channels = 30  # discretized_mix_logistic_loss를 적용하기 때문에, 3의 배수
    skip_channels = 128
    use_biases = True
    upsample_type = 'SubPixel'     # 'SubPixel', None
    upsample_factor = [12, 25]     # np.prod(upsample_factor) must equal to hop_size

    # Encoder
    enc_conv_num_layers = 3
    enc_conv_kernel_size = 5
    enc_conv_channels = 512
    tacotron_zoneout_rate = 0.1
    encoder_lstm_units = 256

    # Attention mechanism
    smoothing = False          # Whether to smooth the attention normalization function
    attention_filters = 32     # number of attention convolution filters
    attention_kernel = (31, )  # kernel size of attention convolution
    cumulative_weights = True  # Whether to cumulate (sum) all previous attention weights or simply feed previous weights (Recommended: True)

    # Attention synthesis constraints
    # "Monotonic" constraint forces the model to only look at the forwards attention_win_size steps.
    # "Window" allows the model to look at attention_win_size neighbors, both forward and backward steps.
    synthesis_constraint = False          # Whether to use attention windows constraints in synthesis only (Useful for long utterances synthesis)
    synthesis_constraint_type = 'window'  # can be in ('window', 'monotonic').
    attention_win_size = 7                # Side of the window. Current step does not count. If mode is window and attention_win_size is not pair, the 1 extra is provided to backward part of the window.

    # Loss params
    mask_encoder = True  # whether to mask encoder padding while computing location sensitive attention. Set to True for better prosody but slower convergence.

    # Decoder
    prenet_layers = [256, 256]     # number of layers and number of units of prenet
    dec_prenet_sizes = [256, 256]  # number of layers and number of units of prenet
    decoder_lstm_units = 1024      # number of decoder lstm units on each layer

    # Residual postnet
    postnet_num_layers = 5       # number of postnet convolutional layers
    postnet_kernel_size = (5, )  # size of postnet convolution filters for each layer
    postnet_channels = 512       # number of postnet convolution filters for each layer

    # for linear mel spectrogrma
    post_bank_size = 8
    post_bank_channel_size = 128
    post_maxpool_width = 2
    post_highway_depth = 4
    post_rnn_size = 128
    post_proj_sizes = [256, 80]  # num_mels = 80
    post_proj_width = 3

    tacotron_reg_weight = 1e-6  # regularization weight (for L2 regularization)
    inference_prenet_dropout = True

    # Eval
    min_tokens = 30      # originally 50, 30 is good for korean,  text를 token으로 쪼갰을 때, 최소 길이 이상되어야 train에 사용
    min_n_frame = 30*5   # min_n_frame = reduction_factor * min_iters, reduction_factor와 곱해서 min_n_frame을 설정한다.
    max_n_frame = 200*5
    skip_inadequate = False
 
    griffin_lim_iters = 60
    power = 1.5

    ############################
    # wavenet training hp      #
    ############################
    wavenet_batch_size = 2  # 16--> OOM. wavenet은 batch_size가 고정되어야 한다.
    store_metadata = False
    num_steps = 1000000     # Number of training steps

    # Learning rate schedule
    wavenet_learning_rate = 1e-3  # wavenet initial learning rate
    wavenet_decay_rate = 0.5      # Only used with 'exponential' scheme. Defines the decay rate.
    wavenet_decay_steps = 300000  # Only used with 'exponential' scheme. Defines the decay steps.

    # Regularization parameters
    wavenet_clip_gradients = True  # Whether the clip the gradients during wavenet training.

    # residual 결과를 sum할 때,
    legacy = True  # Whether to use legacy mode: Multiply all skip outputs but the first one with sqrt(0.5) (True for more early training stability, especially for large models)

    # residual block내에서  x = (x + residual) * np.sqrt(0.5)
    residual_legacy = True  # Whether to scale residual blocks outputs by a factor of sqrt(0.5) (True for input variance preservation early in training and better overall stability)
    wavenet_dropout = 0.05
    optimizer = 'adam'
    momentum = 0.9       # 'Specify the momentum to be used by sgd or rmsprop optimizer. Ignored by the adam optimizer.
    max_checkpoints = 3  # 'Maximum amount of checkpoints that will be kept alive. Default: '


if hparams.use_lws:
    # Does not work if fft_size is not multiple of hop_size!!
    # sample size = 20480, hop_size=256=12.5ms. fft_size는 window_size를 결정하는데, 2048을 시간으로 환산하면 2048/20480 = 0.1초=100ms
    hparams.sample_rate = 20480
    
    # shift can be specified by either hop_size(우선) or frame_shift_ms
    hparams.hop_size = 256             # frame_shift_ms = 12.5ms
    hparams.frame_shift_ms = None      # hop_size = sample_rate * frame_shift_ms / 1000
    hparams.fft_size = 2048            # 주로 1024로 되어있는데, tacotron에서 2048사용 ==> output size = 1025
    hparams.win_size = None            # 256x4 --> 50ms

    # 미리 정의된 parameter들로 부터 consistant하게 정의해 준다.
    hparams.num_freq = int(hparams.fft_size/2 + 1)
    hparams.frame_shift_ms = hparams.hop_size * 1000.0 / hparams.sample_rate      # hop_size = sample_rate * frame_shift_ms / 1000
    hparams.frame_length_ms = hparams.win_size * 1000.0 / hparams.sample_rate

def hparams_debug_string():
    hp = ['  %s: %s' % (name, value) for name, value in hparams.__dict__.items() if "__" not in name]
    return 'Hyperparameters:\n' + '\n'.join(hp)
