import numpy as np
from .constants import *
from mido import Message, MidiFile, MidiTrack
import soundfile


def extract_notes_np(onsets, frames, velocity,
                     onset_threshold=0.5, frame_threshold=0.5, onset_threshold_vec=None):
    if onset_threshold_vec is not None:
        onsets = (onsets > np.array(onset_threshold_vec)).astype(np.uint8)
    else:
        onsets = (onsets > onset_threshold).astype(np.uint8)

    frames = (frames > frame_threshold).astype(np.uint8)
    onset_diff = np.concatenate([onsets[:1, :], onsets[1:, :] - onsets[:-1, :]], axis=0) == 1

    if onsets.shape[-1] != frames.shape[-1]:
        num_instruments = onsets.shape[1] / frames.shape[1]
        assert num_instruments.is_integer()
        num_instruments = int(num_instruments)
        frames = np.tile(frames, (1, num_instruments))

    pitches = []
    intervals = []
    velocities = []
    instruments = []

    for nonzero in np.transpose(np.nonzero(onset_diff)):
        frame = nonzero[0].item()
        pitch = nonzero[1].item()

        onset = frame
        offset = frame
        velocity_samples = []

        while onsets[offset, pitch] or frames[offset, pitch]:
            if onsets[offset, pitch]:
                velocity_samples.append(velocity[offset, pitch])
            offset += 1
            if offset == onsets.shape[0]:
                break

        if offset > onset:
            pitch, instrument = pitch % N_KEYS, pitch // N_KEYS

            pitches.append(pitch)
            intervals.append([onset, offset])
            velocities.append(np.mean(velocity_samples) if len(velocity_samples) > 0 else 0)
            instruments.append(instrument)
    return np.array(pitches), np.array(intervals), np.array(velocities), np.array(instruments)


def midi_to_hz(m):
    return 440. * (2. ** ((m - 69.) / 12.))


def hz_to_midi(h):
    return 12. * np.log2(h / (440.)) + 69.


def append_track_multi(file, pitches, intervals, velocities, ins, single_ins=False):
    track = MidiTrack()
    file.tracks.append(track)
    chan = len(file.tracks) - 1
    if chan >= DRUM_CHANNEL:
        chan += 1
    if chan > 15:
        print(f"invalid chan {chan}")
        chan = 15
    track.append(Message('program_change', channel=chan, program=ins if not single_ins else 0, time=0))

    ticks_per_second = file.ticks_per_beat * 2.0

    events = []
    for i in range(len(pitches)):
        events.append(dict(type='on', pitch=pitches[i], time=intervals[i][0], velocity=velocities[i]))
        events.append(dict(type='off', pitch=pitches[i], time=intervals[i][1], velocity=velocities[i]))
    events.sort(key=lambda row: row['time'])

    last_tick = 0
    for event in events:
        current_tick = int(event['time'] * ticks_per_second)
        velocity = int(event['velocity'] * 127)
        if velocity > 127:
            velocity = 127
        pitch = int(round(hz_to_midi(event['pitch'])))
        track.append(Message('note_' + event['type'], channel=chan, note=pitch, velocity=velocity, time=current_tick - last_tick))
        last_tick = current_tick


def append_track(file, pitches, intervals, velocities):
    track = MidiTrack()
    file.tracks.append(track)
    ticks_per_second = file.ticks_per_beat * 2.0

    events = []
    for i in range(len(pitches)):
        events.append(dict(type='on', pitch=pitches[i], time=intervals[i][0], velocity=velocities[i]))
        events.append(dict(type='off', pitch=pitches[i], time=intervals[i][1], velocity=velocities[i]))
    events.sort(key=lambda row: row['time'])

    last_tick = 0
    for event in events:
        current_tick = int(event['time'] * ticks_per_second)
        velocity = int(event['velocity'] * 127)
        if velocity > 127:
            velocity = 127
        pitch = int(round(hz_to_midi(event['pitch'])))
        try:
            track.append(Message('note_' + event['type'], note=pitch, velocity=velocity, time=current_tick - last_tick))
        except Exception as e:
            print('Err Message', 'note_' + event['type'], pitch, velocity, current_tick - last_tick)
            track.append(Message('note_' + event['type'], note=pitch, velocity=max(0, velocity), time=current_tick - last_tick))
            if velocity >= 0:
                raise e
        last_tick = current_tick


def save_midi(path, pitches, intervals, velocities, insts=None):
    """
    Save extracted notes as a MIDI file
    Parameters
    ----------
    path: the path to save the MIDI file
    pitches: np.ndarray of bin_indices
    intervals: list of (onset_index, offset_index)
    velocities: list of velocity values
    """
    file = MidiFile()
    if isinstance(pitches, list):
        for p, i, v, ins in zip(pitches, intervals, velocities, insts):
            append_track_multi(file, p, i, v, ins)
    else:
        append_track(file, pitches, intervals, velocities)
    file.save(path)


def frames2midi(save_path, onsets, frames, vels,
                onset_threshold=0.5, frame_threshold=0.5, scaling=HOP_LENGTH / SAMPLE_RATE,
                inst_mapping=None, onset_threshold_vec=None):
    p_est, i_est, v_est, inst_est = extract_notes_np(onsets, frames, vels,
                                        onset_threshold, frame_threshold, onset_threshold_vec=onset_threshold_vec)
    i_est = (i_est * scaling).reshape(-1, 2)

    p_est = np.array([midi_to_hz(MIN_MIDI + midi) for midi in p_est])

    inst_set = set(inst_est)
    inst_set = sorted(list(inst_set))

    p_est_lst = {}
    i_est_lst = {}
    v_est_lst = {}
    assert len(p_est) == len(i_est) == len(v_est) == len(inst_est)
    for p, i, v, ins in zip(p_est, i_est, v_est, inst_est):
        if ins in p_est_lst:
            p_est_lst[ins].append(p)
        else:
            p_est_lst[ins] = [p]
        if ins in i_est_lst:
            i_est_lst[ins].append(i)
        else:
            i_est_lst[ins] = [i]
        if ins in v_est_lst:
            v_est_lst[ins].append(v)
        else:
            v_est_lst[ins] = [v]
    for elem in [p_est_lst, i_est_lst, v_est_lst]:
        for k, v in elem.items():
            elem[k] = np.array(v)
    inst_set = [e for e in inst_set if e in p_est_lst]
    # inst_set = [INSTRUMENT_MAPPING[e] for e in inst_set if e in p_est_lst]
    p_est_lst = [p_est_lst[ins] for ins in inst_set if ins in p_est_lst]
    i_est_lst = [i_est_lst[ins] for ins in inst_set if ins in i_est_lst]
    v_est_lst = [v_est_lst[ins] for ins in inst_set if ins in v_est_lst]
    assert len(p_est_lst) == len(i_est_lst) == len(v_est_lst) == len(inst_set)
    inst_set = [inst_mapping[e] for e in inst_set]
    save_midi(save_path,
              p_est_lst, i_est_lst, v_est_lst,
              inst_set)


def load_audio(flac):
    audio, sr = soundfile.read(flac, dtype='int16')
    if len(audio.shape) == 2:
        audio = audio.astype(float).mean(axis=1)
    else:
        audio = audio.astype(float)
    audio = audio.astype(np.int16)
    print('audio len', len(audio))
    assert sr == SAMPLE_RATE
    audio = torch.ShortTensor(audio)
    return audio


def max_inst(probs, threshold_vec=None):
    if threshold_vec is None:
        threshold_vec = 0.5
    if probs.shape[-1] == N_KEYS or probs.shape[-1] == N_KEYS * 2:
        # there is only pitch
        return probs
    keys = MAX_MIDI - MIN_MIDI + 1
    instruments = probs.shape[1] // keys
    time = len(probs)
    probs = probs.reshape((time, instruments, keys))
    notes = probs.max(axis=1) >= threshold_vec
    max_instruments = np.argmax(probs[:, : -1, :], axis=1)
    res = np.zeros(probs.shape, dtype=np.uint8)
    for t, p in zip(*(notes.nonzero())):
        res[t, max_instruments[t, p], p] = 1
        res[t, -1, p] = 1
    return res.reshape((time, instruments * keys))