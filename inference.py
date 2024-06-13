import os
# from onsets_and_frames.utils import frames2midi, load_audio, max_inst
# from onsets_and_frames.constants import *
# from onsets_and_frames.mel import melspectrogram
from onsets_and_frames import *
import numpy as np


def inference_single_flac(transcriber, flac_path, inst_mapping, out_dir, modulated_transcriber=False, use_max_inst=True,
                          pitch_transcriber=None, mask=None, save_onsets_and_frames=False, onset_threshold_vec=None):
    audio = load_audio(flac_path)
    audio_inp = audio.float() / 32768.
    MAX_TIME = 5 * 60 * SAMPLE_RATE
    audio_inp_len = len(audio_inp)
    if audio_inp_len > MAX_TIME:
        n_segments = 3 if audio_inp_len > 2 * MAX_TIME else 2
        print('long audio, splitting to {} segments'.format(n_segments))
        seg_len = audio_inp_len // n_segments
        onsets_preds = []
        offset_preds = []
        frame_preds = []

        pitch_onsets_preds = []
        pitch_offset_preds = []
        pitch_frame_preds = []
        for i_s in range(n_segments):
            curr = audio_inp[i_s * seg_len: (i_s + 1) * seg_len].unsqueeze(0).cuda()
            curr_mel = melspectrogram(curr.reshape(-1, curr.shape[-1])[:, :-1]).transpose(-1, -2)
            curr_onset_pred, curr_offset_pred, _, curr_frame_pred, *_ = transcriber(curr_mel)

            if pitch_transcriber is not None:
                pitch_curr_onset_pred, pitch_curr_offset_pred, _, pitch_curr_frame_pred, *_ = pitch_transcriber(
                    curr_mel)
                pitch_onsets_preds.append(pitch_curr_onset_pred)
                print("pitch curr onset pred shape", pitch_curr_onset_pred.shape)
                print("pitch curr frame pred shape", pitch_curr_frame_pred.shape)

                pitch_offset_preds.append(pitch_curr_offset_pred)
                pitch_frame_preds.append(pitch_curr_frame_pred)
                # pitch_vel_preds.append(pitch_curr_velocity_pred)
            onsets_preds.append(curr_onset_pred)
            offset_preds.append(curr_offset_pred)
            frame_preds.append(curr_frame_pred)
            # vel_preds.append(curr_velocity_pred)
        onset_pred = torch.cat(onsets_preds, dim=1)
        offset_pred = torch.cat(offset_preds, dim=1)
        frame_pred = torch.cat(frame_preds, dim=1)
        # velocity_pred = torch.cat(vel_preds, dim=1)
        if pitch_transcriber is not None:
            pitch_onset_pred = torch.cat(pitch_onsets_preds, dim=1)
            pitch_offset_pred = torch.cat(pitch_offset_preds, dim=1)
            pitch_frame_pred = torch.cat(pitch_frame_preds, dim=1)
            # pitch_velocity_pred = torch.cat(pitch_vel_preds, dim=1)

    else:
        print("didn't have to split")
        audio_inp = audio_inp.unsqueeze(0).cuda()
        mel = melspectrogram(audio_inp.reshape(-1, audio_inp.shape[-1])[:, :-1]).transpose(-1, -2)
        onset_pred, offset_pred, _, frame_pred, *_ = transcriber(mel)
        if pitch_transcriber is not None:
            pitch_onset_pred, pitch_offset_pred, _, pitch_frame_pred, *_ = pitch_transcriber(mel)

    onset_pred = onset_pred.detach().squeeze().cpu()
    frame_pred = frame_pred.detach().squeeze().cpu()

    onset_pred_np = onset_pred.numpy()
    frame_pred_np = frame_pred.numpy()

    if mask is not None:
        mask_with_pitch = mask + [1]
        mask_list = [np.full((onset_pred_np.shape[0], N_KEYS), i) for i in mask_with_pitch]
        mask_array = np.hstack(mask_list)
        print("mask shape", mask_array.shape)
        print("mask array:", mask_array)
        assert onset_pred_np.shape == mask_array.shape
        onset_pred_np = mask_array * onset_pred_np

    if pitch_transcriber is not None:
        pitch_onset_pred = pitch_onset_pred.detach().squeeze().cpu()
        pitch_frame_pred = pitch_frame_pred.detach().squeeze().cpu()
        pitch_onset_pred_np = pitch_onset_pred.numpy()
        onset_pred_np[:, -88:] = pitch_onset_pred_np[:, -88:]
        frame_pred_np = pitch_frame_pred.numpy()

    save_path = os.path.join(out_dir, os.path.basename(flac_path).replace('.flac', '.mid'))

    inst_only = len(inst_mapping) * N_KEYS
    print("inst only")
    print("onset pred shape", onset_pred_np.shape)
    if use_max_inst and len(inst_mapping) > 1:
        print("used max inst !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        onset_pred_np = max_inst(onset_pred_np, threshold_vec=onset_threshold_vec)
        onset_threshold_vec = None
        onset_pred_np = np.maximum(onset_pred_np, max_inst(onset_pred_np))
    if len(inst_mapping) == 1:
        print("onset_pred_np_shape_before", onset_pred_np.shape)
        onset_pred_np = onset_pred_np[:, -88:]
        print("onset_pred_np_shape_after", onset_pred_np.shape)
    frames2midi(save_path,
                onset_pred_np[:, : inst_only], frame_pred_np[:, : inst_only],
                64. * onset_pred_np[:, : inst_only],
                inst_mapping=inst_mapping, onset_threshold_vec=onset_threshold_vec)

    if save_onsets_and_frames:
        onset_save_path = os.path.join(out_dir, os.path.basename(flac_path).replace('.flac', '_onset_pred.npy'))
        frame_save_path = os.path.join(out_dir, os.path.basename(flac_path).replace('.flac', '_frame_pred.npy'))
        np.save(onset_save_path, onset_pred_np)
        np.save(frame_save_path, frame_pred_np)

    print(f"saved midi to {save_path}")
    return save_path
