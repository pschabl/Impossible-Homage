/*--------------------------------------------------------------------------

    Test program for synthesizing sounds in 'Impossible Homage'.

    This program does not play sounds, rather it generates some amount of
    samples (~1s) for all implemented wave forms and writes them to disk
    in WAVE format.

    Output: one WAVE file per wave form ('dsp_form.wav').

    Compile: gcc -Wall -O2 dsp.c -o dsp -lm

    Note: this code is written for Little Endian 64-bit Linux
          and may not run on other systems without modifications.

    References:

    Oscillator and Wavetable theory:

      https://en.wikibooks.org/wiki/Sound_Synthesis_Theory/Oscillators_and_Wavetables

    Jan Wilcek:

      https://thewolfsound.com/sound-synthesis/wavetable-synthesis-algorithm/
      https://thewolfsound.com/sound-synthesis/wavetable-synth-in-python/

      In-depth articles on the theory and implementation of wavetable synthesis,
      including how to fine-tune an oscillator via amplitude adjustment,
      index interpolation and 'Von Hann' window filtering.

------------------------------------------------------------------------------*/

#include <errno.h>
#include <fcntl.h>
#include <limits.h> // SHRT_MIN, SHRT_MAX
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static const char S_WAV[][8] = {
    [0] = "RIFF",
    [1] = "WAVE",
    [2] = "fmt ", // Observe the space.
    [3] = "data",
};

static const char S_WAVEFORM[][12] = {
    [0] = "sine",
    [1] = "sawtooth",
    [2] = "square",
    [3] = "triangle",
};

#define REG_PERM (S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH) // -rw-r--r--
#define WAV_PCM                                                                \
  16 /*! Value for the field subchunk_1_id (#of following bytes). */
#define WAV_FORMAT 1 /*! PCM */
#define ALSA_PERIOD_SIZE                                                       \
  64 // #of samples expected per write by the audio server
#define SOUND_FREQUENCY (432.0f) // Let's be frank.
#define SOUND_RATE 8000
#define PI (3.14159265358979323846f)  /*! π */
#define TAU (6.28318530717958647692f) /*! 2 π */
#define DECIBEL_FACTOR (20.0f)
#define WAVETABLE_SIZE 256 /*! The size of the sample lookup table. */
#define WAVETABLE_HANN_WINDOW                                                  \
  (4 * WAVETABLE_SIZE) /*! The size of the fade in/out window (in samples). */
#define WAVETABLE_GAIN_LIMIT                                                   \
  (DECIBEL_FACTOR) /*! Put a limit on the amplitude. */
#define WAVETABLE_DEFAULT_GAIN (-DECIBEL_FACTOR)
#define WAVETABLE_STEP (TAU / (float)WAVETABLE_SIZE) /*! Angle increment. */
#define clamp(lo, v, hi) (((v) < (lo)) ? (lo) : (((v) > (hi)) ? (hi) : (v)))
#define SOUND_MIX_LO SHRT_MIN
#define SOUND_MIX_HI SHRT_MAX

typedef unsigned char u8;
typedef signed short s16;
typedef unsigned short u16;
typedef unsigned int u32;

enum { OSC_Sine, OSC_Sawtooth, OSC_Square, OSC_Triangle, OSC_Max };

enum {
  Wavetable_Interrupt,
  Wavetable_Ended,
};

enum {
  WAV_Chunk_Id,
  WAV_Chunk_Format,
  WAV_Subchunk_1_Id,
  WAV_Subchunk_2_Id,
};

typedef struct { // WAVE file header.
  // Note: this representation only covers the one format, which the
  // application expects sound assets to be in. Some WAVE files might have more
  // subchunks, or data following the signal data, but those are not used here.
  char chunk_id[4]; // "RIFF" (0x52494646 big-endian form) = Resource
                    // Interchange File Format.
  u32 chunk_size;   // The size of the rest of the chunk, following this field
                    // (file size - 8).
  char format[4];   // "WAVE" (0x57415645 big-endian form)
  char subchunk_1_id[4]; // "fmt " (0x666d7420 big-endian form)
  u32 subchunk_1_size;   // The size of the rest of the subchunk, after this
                         // field. 16 for PCM.
  u16 audio_format;      // 1 = PCM
  u16 num_channels;      // 1 = mono, 2 = stereo.
  u32 sample_rate;       // E.g. 44100
  u32 byte_rate;         // = sample rate * num channels * bps / 8
  u16 block_align;       // = num channels * bps / 8
  u16 bits_per_sample;   // 8, 16, ...
  // Using PCM, so the fields 'extra param size' and 'extra params' don't exist.
  char subchunk_2_id[4]; // "data" (0x64617461 big-endian form)
  u32 subchunk_2_size; // Size of the sound data, = num samples * num channels *
                       // bps / 8
  u8 data[];           // raw sound data.
} __attribute__((__packed__)) WAV_Header; // 44 b

typedef struct {   // Represents a wave table for use with an oscillator.
  float hz;        /*! The frequency in Hertz. */
  float step_size; /*! The index (phase) advancement per sample. */
  float amplitude; /*! The amplitude of the signal. */
  float index;     /*! The current table lookup index. */
  u8 wave_form;    /*! The desired wave form (sine, square, etc.). */
  u8 flags;        /*! Hints for and from the audio server. */
  u16 window;      /*! The current position in the window (fade). */
  float *table;    /*! Lookup table of pre-computed wave samples. */
} Wavetable;

inline static float midi_to_frequency(int key) {
  // f = 440 * 2 ^ ( ( n - 69 ) / 12 )
  return SOUND_FREQUENCY * powf(2.0f, (float)(key - 69) * (1.0f / 12.0f));
}

inline static float remap(float x, float t1, float t2, float s1, float s2) {
  // Re-map x from old range [t1, t2] to the new range [s1, s2].
  return ((x - t1) / (t2 - t1)) * (s2 - s1) + s1;
}

inline static float dsp_gain_to_amplitude(float gain) {
  float clamped;

  clamped = clamp(-WAVETABLE_GAIN_LIMIT, gain, WAVETABLE_GAIN_LIMIT);
  return powf(10.0f, clamped / DECIBEL_FACTOR);
}

inline static void dsp_init_wave_struct(Wavetable *wave, int wave_form,
                                        int midi_key, float gain) {
  // Initialize the 'wave part' of the struct.
  float hz, cycles_per_sample;

  hz = midi_to_frequency(midi_key);
  cycles_per_sample = hz / SOUND_RATE;

  wave->hz = hz;
  wave->step_size = (float)WAVETABLE_SIZE * cycles_per_sample;
  wave->amplitude = dsp_gain_to_amplitude(gain);
  wave->index = 0.0f;
  wave->wave_form = wave_form;
  wave->flags = 0;
  wave->window = 0;
}

inline static void dsp_init_wave_table(Wavetable *wave) {
  // Initialize the wave form lookup table.
  int i, wave_form;
  float result, phase, table_step;

  wave_form = wave->wave_form;
  result = phase = 0.0f;
  table_step = WAVETABLE_STEP; // 2π over WAVETABLE_SIZE

  for (i = 0; i < WAVETABLE_SIZE; i++) {
    if (OSC_Sine == wave_form) {
      result = sinf(phase);
    } else if (OSC_Square == wave_form) {
      result = sinf(phase);

      if (result < 0.0f) {
        result = -1.0f;
      } else {
        result = 1.0f;
      }
    } else if (OSC_Sawtooth == wave_form) {
      result = fmodf((phase + PI) / PI, 2.0f) - 1.0f;
    } else if (OSC_Triangle == wave_form) {
      result = asinf(sinf(phase));
    }
    wave->table[i] = result;
    phase += table_step;
  }
}

inline static void dsp_wave_table(Wavetable *wave, int wave_form, int midi_key,
                                  float gain) {
  // Initialize a wave table for a given wave form.
  dsp_init_wave_struct(wave, wave_form, midi_key, gain);
  dsp_init_wave_table(wave);
}

inline static float dsp_lerp(float *table, float index) {
  // Linear interpolation between two neighbouring samples.
  int i1, i2;
  float weight;

  i1 = (int)floorf(index);
  i2 = (i1 + 1) % WAVETABLE_SIZE;
  weight = index - (float)i1;
  return table[i1] * (1.0f - weight) + table[i2] * weight;
}

inline static float von_hann(int size, int win_index) {
  // 'Von Hann' window function, aka 'raised cosine filter'.
  float mapped;

  mapped = remap((float)win_index, 0.0f, (float)size, 0.0f, PI);
  return 0.5f * (1.0f - cosf(mapped));
}

inline static int dsp_oscillate(float *dst, Wavetable *wave, int sample_count) {
  // Generate a batch of samples for a given wave form.
  // Start with a fade in filter and end with a fade out filter.
  // Returns zero after completion of the fade out phase.
  int fade_out, window, left;
  float *ptr, *end, amplitude, index, rem, value, step_size;
  const int window_size = WAVETABLE_HANN_WINDOW;

  window = wave->window;
  fade_out = (wave->flags & (1 << Wavetable_Interrupt));

  if (fade_out && (window >= 2 * window_size)) {
    wave->flags |= (1 << Wavetable_Ended);
    return 0;
  }
  ptr = dst;
  end = dst + sample_count;
  step_size = wave->step_size;
  index = wave->index;
  amplitude = wave->amplitude;

  while (ptr < end) {
    rem = fmodf(index, WAVETABLE_SIZE);
    index = rem + step_size;
    value = dsp_lerp(wave->table, rem);

    if (fade_out) {
      left = window_size - 1 - window++ % window_size;
      value *= von_hann(window_size - 1, left);
    } else if (window < window_size) { // Fade in.
      value *= von_hann(window_size - 1, window++);
    }
    *ptr++ = amplitude * value;
  }
  wave->index = index;
  wave->window = window;
  return 1;
}

static int pcm_to_wav(const char *fname, const s16 *samples, int size,
                      int num_channels, int bps, u32 rate) {
  // Write a WAVE file from the given data.
  WAV_Header wav;
  int fd, result, block_align, written, chunk_size;

  chunk_size = sizeof(WAV_Header) + size - 8;
  memcpy(wav.chunk_id, S_WAV[WAV_Chunk_Id], 4);
  wav.chunk_size = chunk_size;
  memcpy(wav.format, S_WAV[WAV_Chunk_Format], 4);
  memcpy(wav.subchunk_1_id, S_WAV[WAV_Subchunk_1_Id], 4);
  wav.subchunk_1_size = WAV_PCM;
  wav.audio_format = WAV_FORMAT;
  wav.num_channels = num_channels;
  wav.sample_rate = rate;
  block_align = num_channels * bps / 8;
  wav.block_align = block_align;
  wav.byte_rate = block_align * rate;
  wav.bits_per_sample = bps;
  memcpy(wav.subchunk_2_id, S_WAV[WAV_Subchunk_2_Id], 4);
  wav.subchunk_2_size = size;

  fd = open(fname, O_RDWR | O_CREAT, REG_PERM);

  if (-1 == fd) {
    fprintf(stderr, "file open error with pcm_to_wav: %s\n", strerror(errno));
    result = -1;
    goto lbl_end;
  }
  written = write(fd, &wav, sizeof(WAV_Header));

  if (sizeof(WAV_Header) != written) {
    fprintf(stderr, "write error with pcm_to_wav: %s\n", strerror(errno));
    result = -1;
    goto lbl_end;
  }
  written = write(fd, samples, size);

  if (size != written) {
    fprintf(stderr, "write error with pcm_to_wav: %s\n", strerror(errno));
    result = -1;
  }

lbl_end:

  close(fd);
  return result;
}

inline static void samples_to_s16(s16 *dst, const float *src, int count) {
  // Convert floating point to signed 16-bit samples.
  int i;

  const float lo = (float)SOUND_MIX_LO;
  const float hi = (float)SOUND_MIX_HI;

  for (i = 0; i < count; i++) {
    dst[i] = (s16)floorf(remap(src[i], -1.0f, 1.0f, lo, hi));
  }
}

int main(void) {
  char buffer[64];
  int i, count, block_size, mem_size;
  s16 *samples, *ptr;
  float output[ALSA_PERIOD_SIZE];
  Wavetable wt;

  wt.table = malloc(WAVETABLE_SIZE * sizeof(float));

  if (!wt.table) {
    fprintf(stderr, "Out of memory!\n");
    exit(EXIT_FAILURE);
  }
  block_size = 1024;
  mem_size = ALSA_PERIOD_SIZE * block_size;
  samples = malloc(mem_size * sizeof(s16));

  if (!samples) {
    fprintf(stderr, "Out of memory!\n");
    exit(EXIT_FAILURE);
  }

  for (i = 0; i < OSC_Max; i++) {
    ptr = samples;
    count = 0;
    dsp_wave_table(&wt, i, 69, -20.0f);

    while (dsp_oscillate(output, &wt, ALSA_PERIOD_SIZE)) {
      samples_to_s16(ptr, output, ALSA_PERIOD_SIZE);

      if (count > (8 * block_size)) { // ~1s of sound.
        wt.flags |= (1 << Wavetable_Interrupt);
      }
      count += ALSA_PERIOD_SIZE;
      ptr += ALSA_PERIOD_SIZE;
    }
    sprintf(buffer, "dsp_%s.wav", S_WAVEFORM[wt.wave_form]);
    pcm_to_wav(buffer, samples, count * sizeof(s16), 1, 16, SOUND_RATE);
  }
  free(wt.table);
  free(samples);
  return EXIT_SUCCESS;
}
