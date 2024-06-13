import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from .mel import melspectrogram
from .lstm import BiLSTM
from torch.nn import TransformerEncoderLayer, TransformerEncoder
from onsets_and_frames.constants import *


class ConvStack(nn.Module):
    def __init__(self, input_features, output_features):
        super().__init__()

        # input is batch_size * 1 channel * frames * input_features
        self.cnn = nn.Sequential(
            # layer 0
            nn.Conv2d(1, output_features // 16, (3, 3), padding=1),
            nn.BatchNorm2d(output_features // 16),
            nn.ReLU(),
            # layer 1
            nn.Conv2d(output_features // 16, output_features // 16, (3, 3), padding=1),
            nn.BatchNorm2d(output_features // 16),
            nn.ReLU(),
            # layer 2
            nn.MaxPool2d((1, 2)),
            nn.Dropout(0.25),
            nn.Conv2d(output_features // 16, output_features // 8, (3, 3), padding=1),
            nn.BatchNorm2d(output_features // 8),
            nn.ReLU(),
            # layer 3
            nn.MaxPool2d((1, 2)),
            nn.Dropout(0.25),
        )
        self.fc = nn.Sequential(
            nn.Linear((output_features // 8) * (input_features // 4), output_features),
            nn.Dropout(0.5)
        )

    def forward(self, mel):
        x = mel.view(mel.size(0), 1, mel.size(1), mel.size(2))
        x = self.cnn(x)
        x = x.transpose(1, 2).flatten(-2)
        x = self.fc(x)
        return x


class ModulatedOnsetStack(nn.Module):
    def __init__(self, input_features, output_features, model_size,
                 mlp_input_features):
        super().__init__()
        sequence_model = lambda input_size, output_size: BiLSTM(input_size, output_size // 2)
        self.model_size = model_size
        self.modulation1 = LinearModulation(model_size, mlp_input_features)
        self.modulation2 = LinearModulation(model_size, mlp_input_features)
        self.conv_stack = ConvStack(input_features, model_size)
        self.seq_model = sequence_model(model_size, model_size)
        self.fc = nn.Sequential(
            nn.Linear(model_size, output_features),
            nn.Sigmoid()
        )

    def forward(self, x, inst_id):
        x = self.conv_stack(x)
        x = self.modulation1(x, inst_id)
        x = self.seq_model(x)
        x = self.modulation2(x, inst_id)
        x = self.fc(x)
        return x


class LinearModulation(nn.Module):
    def __init__(self, input_features, mlp_input_features):
        nn.Module.__init__(self)
        self.mlp = torch.nn.Linear(mlp_input_features, 2 * input_features)

    def forward(self, x, inst_id):
        inst_id = self.mlp(inst_id)
        scale, shift = inst_id.chunk(chunks=2, dim=1)
        scale, shift = scale[:, None, :], shift[:, None, :]
        x = (1 + scale) * x + shift
        return x


class OnsetsAndFrames(nn.Module):
    def __init__(self, input_features, output_features, model_complexity=48,
                 onset_complexity=1,
                 n_instruments=13):
        nn.Module.__init__(self)
        model_size = model_complexity * 16
        sequence_model = lambda input_size, output_size: BiLSTM(input_size, output_size // 2)

        onset_model_size = int(onset_complexity * model_size)
        self.onset_stack = nn.Sequential(
            ConvStack(input_features, onset_model_size),
            sequence_model(onset_model_size, onset_model_size),
            nn.Linear(onset_model_size, output_features * n_instruments),
            nn.Sigmoid()
        )
        self.offset_stack = nn.Sequential(
            ConvStack(input_features, model_size),
            sequence_model(model_size, model_size),
            nn.Linear(model_size, output_features),
            nn.Sigmoid()
        )
        self.frame_stack = nn.Sequential(
            ConvStack(input_features, model_size),
            nn.Linear(model_size, output_features),
            nn.Sigmoid()
        )
        self.combined_stack = nn.Sequential(
            sequence_model(output_features * 3, model_size),
            nn.Linear(model_size, output_features),
            nn.Sigmoid()
        )
        self.velocity_stack = nn.Sequential(
            ConvStack(input_features, model_size),
            nn.Linear(model_size, output_features * n_instruments)
        )

    def forward(self, mel):
        onset_pred = self.onset_stack(mel)
        offset_pred = self.offset_stack(mel)
        activation_pred = self.frame_stack(mel)

        onset_detached = onset_pred.detach()
        shape = onset_detached.shape
        keys = MAX_MIDI - MIN_MIDI + 1
        new_shape = shape[: -1] + (shape[-1] // keys, keys)
        onset_detached = onset_detached.reshape(new_shape)
        onset_detached, _ = onset_detached.max(axis=-2)

        offset_detached = offset_pred.detach()

        combined_pred = torch.cat([onset_detached, offset_detached, activation_pred], dim=-1)
        frame_pred = self.combined_stack(combined_pred)
        velocity_pred = self.velocity_stack(mel)
        return onset_pred, offset_pred, activation_pred, frame_pred, velocity_pred

    def run_on_batch(self, batch, parallel_model=None, multi=False, positive_weight=2., inv_positive_weight=2.):
        audio_label = batch['audio']

        onset_label = batch['onset']
        offset_label = batch['offset']
        frame_label = batch['frame']
        if 'velocity' in batch:
            velocity_label = batch['velocity']
        mel = melspectrogram(audio_label.reshape(-1, audio_label.shape[-1])[:, :-1]).transpose(-1, -2)

        if not parallel_model:
            onset_pred, offset_pred, _, frame_pred, velocity_pred = self(mel)
        else:
            onset_pred, offset_pred, _, frame_pred, velocity_pred = parallel_model(mel)

        if multi:
            onset_pred = onset_pred[..., : N_KEYS]
            offset_pred = offset_pred[..., : N_KEYS]
            frame_pred = frame_pred[..., : N_KEYS]
            velocity_pred = velocity_pred[..., : N_KEYS]

        predictions = {
            'onset': onset_pred.reshape(*onset_label.shape),
            'offset': offset_pred.reshape(*offset_label.shape),
            'frame': frame_pred.reshape(*frame_label.shape),
            # 'velocity': velocity_pred.reshape(*velocity_label.shape)
        }

        if 'velocity' in batch:
            predictions['velocity'] = velocity_pred.reshape(*velocity_label.shape)

        losses = {
            'loss/onset': F.binary_cross_entropy(predictions['onset'], onset_label, reduction='none'),
            'loss/offset': F.binary_cross_entropy(predictions['offset'], offset_label, reduction='none'),
            'loss/frame': F.binary_cross_entropy(predictions['frame'], frame_label, reduction='none'),
            # 'loss/velocity': self.velocity_loss(predictions['velocity'], velocity_label, onset_label)
        }
        if 'velocity' in batch:
            losses['loss/velocity'] = self.velocity_loss(predictions['velocity'], velocity_label, onset_label)

        onset_mask = 1. * onset_label
        onset_mask[..., : -N_KEYS] *= (positive_weight - 1)
        onset_mask[..., -N_KEYS:] *= (inv_positive_weight - 1)
        onset_mask += 1
        if 'onset_mask' in batch:
            onset_mask = onset_mask * batch['onset_mask']

        offset_mask = 1. * offset_label
        offset_positive_weight = 2.
        offset_mask *= (offset_positive_weight - 1)
        offset_mask += 1.

        frame_mask = 1. * frame_label
        frame_positive_weight = 2.
        frame_mask *= (frame_positive_weight - 1)
        frame_mask += 1.

        for loss_key, mask in zip(['onset', 'offset', 'frame'], [onset_mask, offset_mask, frame_mask]):
            losses['loss/' + loss_key] = (mask * losses['loss/' + loss_key]).mean()

        return predictions, losses

    def velocity_loss(self, velocity_pred, velocity_label, onset_label):
        denominator = onset_label.sum()
        if denominator.item() == 0:
            return denominator
        else:
            return (onset_label * (velocity_label - velocity_pred) ** 2).sum() / denominator


class ModulatedOnsetsAndFrames(nn.Module):
    def __init__(self, input_features, output_features, model_complexity=48,
                 onset_complexity=1,
                 n_instruments=13):
        super().__init__()
        model_size = model_complexity * 16
        onset_model_size = int(onset_complexity * model_size)
        self.onset_stack = ModulatedOnsetStack(input_features, output_features, onset_model_size, onset_model_size)
        self.mlp = nn.Sequential(
            nn.Linear(n_instruments, onset_model_size),
            nn.LeakyReLU(),
            nn.Linear(onset_model_size, onset_model_size),
            nn.LeakyReLU(),
        )
        sequence_model = lambda input_size, output_size: BiLSTM(input_size, output_size // 2)
        self.offset_stack = nn.Sequential(
            ConvStack(input_features, model_size),
            sequence_model(model_size, model_size),
            nn.Linear(model_size, output_features),
            nn.Sigmoid()
        )
        self.frame_stack = nn.Sequential(
            ConvStack(input_features, model_size),
            nn.Linear(model_size, output_features),
            nn.Sigmoid()
        )
        self.combined_stack = nn.Sequential(
            sequence_model(output_features * 3, model_size),
            nn.Linear(model_size, output_features),
            nn.Sigmoid()
        )
        self.velocity_stack = nn.Sequential(
            ConvStack(input_features, model_size),
            nn.Linear(model_size, output_features * n_instruments)
        )

    def forward(self, mel, inst_id):
        inst_id = self.mlp(inst_id)
        onset_pred = self.onset_stack(mel, inst_id)
        offset_pred = self.offset_stack(mel)
        activation_pred = self.frame_stack(mel)

        onset_detached = onset_pred.detach()
        shape = onset_detached.shape
        keys = MAX_MIDI - MIN_MIDI + 1
        new_shape = shape[: -1] + (shape[-1] // keys, keys)
        onset_detached = onset_detached.reshape(new_shape)
        onset_detached, _ = onset_detached.max(axis=-2)

        offset_detached = offset_pred.detach()

        combined_pred = torch.cat([onset_detached, offset_detached, activation_pred], dim=-1)
        frame_pred = self.combined_stack(combined_pred)
        velocity_pred = self.velocity_stack(mel)
        return onset_pred, offset_pred, activation_pred, frame_pred, velocity_pred

    def run_on_batch(self, batch, parallel_model=None, multi=False, positive_weight=2., inv_positive_weight=2.):
        audio_label = batch['audio']
        onset_label = batch['onset']
        offset_label = batch['offset']
        frame_label = batch['frame']
        instruments_one_hot_tensor = batch['instruments_one_hots']

        if 'velocity' in batch:
            velocity_label = batch['velocity']
        mel = melspectrogram(audio_label.reshape(-1, audio_label.shape[-1])[:, :-1]).transpose(-1, -2)

        if not parallel_model:
            onset_pred, offset_pred, _, frame_pred, velocity_pred = self(mel, instruments_one_hot_tensor)
        else:
            onset_pred, offset_pred, _, frame_pred, velocity_pred = parallel_model(mel, instruments_one_hot_tensor)

        # if multi:
        #     onset_pred = onset_pred[..., : N_KEYS]
        #     offset_pred = offset_pred[..., : N_KEYS]
        #     frame_pred = frame_pred[..., : N_KEYS]
        #     velocity_pred = velocity_pred[..., : N_KEYS]

        predictions = {
            'onset': onset_pred,
            'offset': offset_pred,
            'frame': frame_pred,
            # 'velocity': velocity_pred.reshape(*velocity_label.shape)
        }
        if 'velocity' in batch:
            predictions['velocity'] = velocity_pred

        losses = {
            'loss/onset': F.binary_cross_entropy(predictions['onset'], onset_label, reduction='none'),
            'loss/offset': F.binary_cross_entropy(predictions['offset'], offset_label, reduction='none'),
            'loss/frame': F.binary_cross_entropy(predictions['frame'], frame_label, reduction='none'),
            # 'loss/velocity': self.velocity_loss(predictions['velocity'], velocity_label, onset_label)
        }
        # if 'velocity' in batch:
        #     losses['loss/velocity'] = self.velocity_loss(predictions['velocity'], velocity_label, onset_label)

        onset_mask = 1. * onset_label
        onset_mask *= (positive_weight - 1)
        # onset_mask[..., : -N_KEYS] *= (positive_weight - 1)
        # onset_mask[..., -N_KEYS:] *= (inv_positive_weight - 1)
        onset_mask += 1
        if 'onset_mask' in batch:
            onset_mask = onset_mask * batch['onset_mask']

        offset_mask = 1. * offset_label
        offset_positive_weight = 2.
        offset_mask *= (offset_positive_weight - 1)
        offset_mask += 1.

        frame_mask = 1. * frame_label
        frame_positive_weight = 2.
        frame_mask *= (frame_positive_weight - 1)
        frame_mask += 1.

        for loss_key, mask in zip(['onset', 'offset', 'frame'], [onset_mask, offset_mask, frame_mask]):
            losses['loss/' + loss_key] = (mask * losses['loss/' + loss_key]).mean()

        return predictions, losses


class ModulatedOnsetsAndFramesGroup(nn.Module):
    def __init__(self, input_features, output_features, model_complexity=48,
                 onset_complexity=1,
                 n_instruments=13, n_groups=1):
        super().__init__()
        model_size = model_complexity * 16
        onset_model_size = int(onset_complexity * model_size)
        self.onset_stack = ModulatedOnsetStack(input_features, output_features, onset_model_size, 2 * onset_model_size)
        self.mlp_inst = nn.Sequential(
            nn.Linear(n_instruments, onset_model_size),
            nn.LeakyReLU(),
            nn.Linear(onset_model_size, onset_model_size),
            nn.LeakyReLU(),
        )
        self.mlp_group = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(n_groups, onset_model_size),
            nn.LeakyReLU(),
            nn.Linear(onset_model_size, onset_model_size),
            nn.LeakyReLU(),
        )
        sequence_model = lambda input_size, output_size: BiLSTM(input_size, output_size // 2)
        self.offset_stack = nn.Sequential(
            ConvStack(input_features, model_size),
            sequence_model(model_size, model_size),
            nn.Linear(model_size, output_features),
            nn.Sigmoid()
        )
        self.frame_stack = nn.Sequential(
            ConvStack(input_features, model_size),
            nn.Linear(model_size, output_features),
            nn.Sigmoid()
        )
        self.combined_stack = nn.Sequential(
            sequence_model(output_features * 3, model_size),
            nn.Linear(model_size, output_features),
            nn.Sigmoid()
        )
        self.velocity_stack = nn.Sequential(
            ConvStack(input_features, model_size),
            nn.Linear(model_size, output_features * n_instruments)
        )

    def forward(self, mel, inst_id, group_id):
        inst_id = self.mlp_inst(inst_id)
        group_id = self.mlp_group(group_id)
        modulation_id = torch.cat((inst_id, group_id), dim=1)
        onset_pred = self.onset_stack(mel, modulation_id)
        offset_pred = self.offset_stack(mel)
        activation_pred = self.frame_stack(mel)

        onset_detached = onset_pred.detach()
        shape = onset_detached.shape
        keys = MAX_MIDI - MIN_MIDI + 1
        new_shape = shape[: -1] + (shape[-1] // keys, keys)
        onset_detached = onset_detached.reshape(new_shape)
        onset_detached, _ = onset_detached.max(axis=-2)

        offset_detached = offset_pred.detach()

        combined_pred = torch.cat([onset_detached, offset_detached, activation_pred], dim=-1)
        frame_pred = self.combined_stack(combined_pred)
        velocity_pred = self.velocity_stack(mel)
        return onset_pred, offset_pred, activation_pred, frame_pred, velocity_pred

    def run_on_batch(self, batch, parallel_model=None, multi=False, positive_weight=2., inv_positive_weight=2.):
        audio_label = batch['audio']
        onset_label = batch['onset']
        offset_label = batch['offset']
        frame_label = batch['frame']
        instruments_one_hot_tensor = batch['instruments_one_hots']
        group_one_hot_tensor = batch['group_one_hots']

        if 'velocity' in batch:
            velocity_label = batch['velocity']
        mel = melspectrogram(audio_label.reshape(-1, audio_label.shape[-1])[:, :-1]).transpose(-1, -2)

        if not parallel_model:
            onset_pred, offset_pred, _, frame_pred, velocity_pred = self(mel, instruments_one_hot_tensor,
                                                                         group_one_hot_tensor)
        else:
            onset_pred, offset_pred, _, frame_pred, velocity_pred = parallel_model(mel, instruments_one_hot_tensor,
                                                                                   group_one_hot_tensor)

        # if multi:
        #     onset_pred = onset_pred[..., : N_KEYS]
        #     offset_pred = offset_pred[..., : N_KEYS]
        #     frame_pred = frame_pred[..., : N_KEYS]
        #     velocity_pred = velocity_pred[..., : N_KEYS]

        predictions = {
            'onset': onset_pred,
            'offset': offset_pred,
            'frame': frame_pred,
            # 'velocity': velocity_pred.reshape(*velocity_label.shape)
        }
        if 'velocity' in batch:
            predictions['velocity'] = velocity_pred

        losses = {
            'loss/onset': F.binary_cross_entropy(predictions['onset'], onset_label, reduction='none'),
            'loss/offset': F.binary_cross_entropy(predictions['offset'], offset_label, reduction='none'),
            'loss/frame': F.binary_cross_entropy(predictions['frame'], frame_label, reduction='none'),
            # 'loss/velocity': self.velocity_loss(predictions['velocity'], velocity_label, onset_label)
        }
        # if 'velocity' in batch:
        #     losses['loss/velocity'] = self.velocity_loss(predictions['velocity'], velocity_label, onset_label)

        onset_mask = 1. * onset_label
        onset_mask *= (positive_weight - 1)
        # onset_mask[..., : -N_KEYS] *= (positive_weight - 1)
        # onset_mask[..., -N_KEYS:] *= (inv_positive_weight - 1)
        onset_mask += 1
        if 'onset_mask' in batch:
            onset_mask = onset_mask * batch['onset_mask']

        offset_mask = 1. * offset_label
        offset_positive_weight = 2.
        offset_mask *= (offset_positive_weight - 1)
        offset_mask += 1.

        frame_mask = 1. * frame_label
        frame_positive_weight = 2.
        frame_mask *= (frame_positive_weight - 1)
        frame_mask += 1.

        for loss_key, mask in zip(['onset', 'offset', 'frame'], [onset_mask, offset_mask, frame_mask]):
            losses['loss/' + loss_key] = (mask * losses['loss/' + loss_key]).mean()

        return predictions, losses


class OnsetsAndFrames2(nn.Module):
    def __init__(self, input_features, output_features, model_complexity=48, onset_complexity=1):
        super().__init__()
        model_size = model_complexity * 16
        sequence_model = lambda input_size, output_size: BiLSTM(input_size, output_size // 2)
        print('onset complexity:', onset_complexity)
        transformer_sequence_model = lambda output_size, n_layers, n_heads: \
            TransformerEncoder(TransformerEncoderLayer(d_model=output_size, nhead=n_heads),
                               num_layers=n_layers)

        # self.onset_stack = nn.Sequential(
        #     ConvStack(input_features, model_size),
        #     sequence_model(model_size, model_size),
        #     nn.Linear(model_size, output_features),
        #     nn.Sigmoid()
        # )
        onset_model_size = int(onset_complexity * model_size)
        self.onset_stack = nn.Sequential(
            ConvStack(input_features, onset_model_size),
            # transformer_sequence_model(model_size, 1, 8),
            sequence_model(onset_model_size, onset_model_size),
            nn.Linear(onset_model_size, output_features),
            nn.Sigmoid()
        )
        self.offset_stack = nn.Sequential(
            ConvStack(input_features, model_size),
            # transformer_sequence_model(model_size, 1, 8),
            sequence_model(model_size, model_size),
            nn.Linear(model_size, output_features),
            nn.Sigmoid()
        )
        self.frame_stack = nn.Sequential(
            ConvStack(input_features, model_size),
            nn.Linear(model_size, output_features),
            # nn.Linear(model_size, 88),
            nn.Sigmoid()
        )
        self.combined_stack = nn.Sequential(
            sequence_model(output_features * 3, model_size),
            nn.Linear(model_size, output_features),
            # sequence_model(output_features * 3 * len(INSTRUMENT_MAPPING),
            #                output_features * 3 * len(INSTRUMENT_MAPPING)),
            # nn.Linear(output_features * 3 * len(INSTRUMENT_MAPPING), output_features),
            # sequence_model(output_features * 3, output_features * 3),
            # nn.Linear(output_features * 3, output_features),

            ## sequence_model(output_features * 3, model_size),
            # sequence_model(88 * 3, model_size),
            # nn.Linear(model_size, output_features),
            nn.Sigmoid()
        )
        # self.combined_stack = nn.Sequential(
        #     sequence_model(output_features * 3, 48 * 16),
        #     nn.Linear(48 * 16, output_features),
        #     nn.Sigmoid()
        # )
        # self.combined_stack = nn.Sequential(
        #     nn.Linear(output_features * 3, model_size),
        #     transformer_sequence_model(model_size, 1, 8),
        #     nn.Linear(model_size, output_features),
        #     nn.Sigmoid()
        # )
        self.velocity_stack = nn.Sequential(
            ConvStack(input_features, model_size),
            nn.Linear(model_size, output_features)
        )

    def forward(self, mel):
        onset_pred = self.onset_stack(mel)
        offset_pred = self.offset_stack(mel)
        activation_pred = self.frame_stack(mel)

        # onset_detached = onset_pred.detach()
        # offset_detached = offset_pred.detach()
        # n_copies = len(INSTRUMENT_MAPPING)
        combined_pred = torch.cat([onset_pred.detach(), offset_pred.detach(), activation_pred], dim=-1)
        # combined_pred = torch.cat(n_copies * [onset_detached] + n_copies * [offset_detached] + n_copies * [activation_pred], dim=-1)
        frame_pred = self.combined_stack(combined_pred)
        velocity_pred = self.velocity_stack(mel)
        return onset_pred, offset_pred, activation_pred, frame_pred, velocity_pred

    def run_on_batch(self, batch, parallel_model=None, multi=False):
        audio_label = batch['audio']

        onset_label = batch['onset']
        offset_label = batch['offset']
        frame_label = batch['frame']
        if 'velocity' in batch:
            velocity_label = batch['velocity']
        # velocity_mask = batch['velocity_mask'].squeeze()

        # if onset_label.shape[-1] < MAX_MIDI - MIN_MIDI + 1:
        #     t = onset_label.shape[0]
        #     p = MAX_MIDI - MIN_MIDI + 1 - onset_label.shape[-1]
        #     onset_label = torch.cat((onset_label, torch.zeros((t, p), dtype=torch.uint8).cuda()), dim=1)
        #     offset_label = torch.cat((offset_label, torch.zeros((t, p), dtype=torch.uint8).cuda()), dim=1)
        #     frame_label = torch.cat((frame_label, torch.zeros((t, p), dtype=torch.uint8).cuda()), dim=1)
        #     velocity_label = torch.cat((velocity_label, torch.zeros((t, p), dtype=torch.uint8).cuda()), dim=1)

        mel = melspectrogram(audio_label.reshape(-1, audio_label.shape[-1])[:, :-1]).transpose(-1, -2)
        if not parallel_model:
            onset_pred, offset_pred, _, frame_pred, velocity_pred = self(mel)
        else:
            onset_pred, offset_pred, _, frame_pred, velocity_pred = parallel_model(mel)

        if multi:
            onset_pred = onset_pred[..., : N_KEYS]
            offset_pred = offset_pred[..., : N_KEYS]
            # b, t, p = frame_pred.reshape
            # frame_pred_reshaped = frame_pred.reshape((b, t, p // N_KEYS, N_KEYS))
            # frame_pred, _ = frame_pred_reshaped.max(dim=2)
            frame_pred = frame_pred[..., : N_KEYS]
            velocity_pred = velocity_pred[..., : N_KEYS]
        predictions = {
            'onset': onset_pred.reshape(*onset_label.shape),
            'offset': offset_pred.reshape(*offset_label.shape),
            'frame': frame_pred.reshape(*frame_label.shape),
            # 'velocity': velocity_pred.reshape(*velocity_label.shape)
        }
        if 'velocity' in batch:
            predictions['velocity'] = velocity_pred.reshape(*velocity_label.shape)

        if 'onsets_mask' not in batch:
            losses = {
                'loss/onset': F.binary_cross_entropy(predictions['onset'], onset_label),
                'loss/offset': F.binary_cross_entropy(predictions['offset'], offset_label),
                'loss/frame': F.binary_cross_entropy(predictions['frame'], frame_label),
                # 'loss/velocity': self.velocity_loss(predictions['velocity'], velocity_label, onset_label)
            }
            if 'velocity' in batch:
                losses['loss/velocity'] = self.velocity_loss(predictions['velocity'], velocity_label, onset_label)

        else:
            losses = {
                'loss/onset': F.binary_cross_entropy(predictions['onset'], onset_label, reduction='none'),
                'loss/offset': F.binary_cross_entropy(predictions['offset'], offset_label, reduction='none'),
                'loss/frame': F.binary_cross_entropy(predictions['frame'], frame_label, reduction='none'),
                # 'loss/velocity': self.velocity_loss(predictions['velocity'], velocity_label, onset_label)
            }
            if 'velocity' in batch:
                losses['loss/velocity'] = self.velocity_loss(predictions['velocity'], velocity_label, onset_label)

                # l1 = F.binary_cross_entropy(predictions['onset'], onset_label, reduction='none')
            # l1 = l1[..., 1: -1, :]
            # l2 = F.binary_cross_entropy(predictions['onset'][..., : -1, :], onset_label[..., 1:, :], reduction='none')
            # l2 = l2[..., : -1, :]
            # l3 = F.binary_cross_entropy(predictions['onset'][..., 1:, :], onset_label[..., : -1, :], reduction='none')
            # l3 = l3[..., 1:, :]
            # min_loss = torch.min(l1, l2)
            # # min_loss = torch.min(min_loss, l3)
            # losses['loss/onset'] = min_loss
            # onset_mask = batch['onsets_mask'][..., 1: -1, :]

            onset_mask = batch['onsets_mask']
            frame_mask = batch['frames_mask']
            offset_mask = batch['offsets_mask']

            # onset_mask = len(INSTRUMENT_MAPPING) * onset_label + (1 - onset_label)
            # frame_mask = len(INSTRUMENT_MAPPING) * frame_label + (1 - frame_label)
            # offset_mask = len(INSTRUMENT_MAPPING) * offset_label + (1 - offset_label)
            # print('onset shapes', onset_label.shape, onset_pred.shape, len(INSTRUMENT_MAPPING) * onset_label.mean())
            # print('frame shapes', frame_label.shape, frame_pred.shape, frame_label.mean())
            # print('offset shapes', offset_label.shape, offset_pred.shape, offset_label.mean())

            for loss_key, mask in zip(['onset', 'offset', 'frame'], [onset_mask, frame_mask, offset_mask]):
                losses['loss/' + loss_key] = (mask * losses['loss/' + loss_key]).mean()

                # losses['loss/' + loss_key] = (mask * losses_unreduced['loss/' + loss_key]).mean()

                # losses['loss/' + loss_key] = (predictions[loss_key] * losses['loss/' + loss_key]).mean()
            # del losses_unreduced['loss/' + loss_key]
            # torch.cuda.empty_cache()

        # losses = {}
        # losses['loss/velocity'] = self.velocity_loss(predictions['velocity'], velocity_label, onset_label)
        # for name, lab in zip(['onset', 'offset', 'frame'], [onset_label, offset_label, frame_label]):
        #     l1 = F.binary_cross_entropy(predictions[name], lab, reduction='none')
        #     l1 = l1[..., 1: -1, :]
        #     l2 = F.binary_cross_entropy(predictions[name][..., : -1, :], lab[..., 1:, :], reduction='none')
        #     l2 = l2[..., : -1, :]
        #     l3 = F.binary_cross_entropy(predictions[name][..., 1:, :], lab[..., : -1, :], reduction='none')
        #     l3 = l3[..., 1:, :]
        #     min_loss = torch.min(l1, l2)
        #     min_loss = torch.min(min_loss, l3)
        #     losses['loss/' + name] = min_loss.mean()

        # if self.n_instruments > 1:
        #     predictions_max = {k: v.max(dim=-1)[0] for k, v in predictions.items()}
        #     # for elem in [onset_label, offset_label, frame_label, velocity_label]:
        #     #     print('elem shape', elem.shape)
        #     losses_max = {
        #         'loss_max/onset': F.binary_cross_entropy(predictions_max['onset'], onset_label.max(dim=-1)[0]),
        #         'loss_max/offset': F.binary_cross_entropy(predictions_max['offset'], offset_label.max(dim=-1)[0]),
        #         'loss_max/frame': F.binary_cross_entropy(predictions_max['frame'], frame_label.max(dim=-1)[0]),
        #         'loss_max/velocity': self.velocity_loss(predictions_max['velocity'],
        #                                             velocity_label.max(dim=-1)[0], onset_label.max(dim=-1)[0])
        #     }
        #     losses_max.update({k: self.loss_weight * v for k, v in losses.items()})
        #     losses = losses_max

        return predictions, losses

    def velocity_loss(self, velocity_pred, velocity_label, onset_label):
        denominator = onset_label.sum()
        if denominator.item() == 0:
            return denominator
        else:
            return (onset_label * (velocity_label - velocity_pred) ** 2).sum() / denominator

    # def velocity_loss(self, velocity_pred, velocity_label, onset_label, mask=None):
    #     to_sum = onset_label
    #     if mask is not None:
    #         to_sum = to_sum[mask]
    #     denominator = to_sum.sum()
    #     if denominator.item() == 0:
    #         return denominator
    #     else:
    #         res = (onset_label * (velocity_label - velocity_pred) ** 2)
    #         if mask is not None:
    #             res = res[mask]
    #         return res.sum() / denominator


class OnsetsAndFramesMultiV2(OnsetsAndFrames2):
    def forward(self, mel):
        onset_pred = self.onset_stack(mel)
        offset_pred = self.offset_stack(mel)
        activation_pred = self.frame_stack(mel)

        onset_detached = onset_pred.detach()
        offset_detached = offset_pred.detach()

        combined_pred = torch.cat([onset_detached, offset_detached, activation_pred], dim=-1)
        frame_pred = self.combined_stack(combined_pred)
        velocity_pred = self.velocity_stack(mel)
        return onset_pred, offset_pred, activation_pred, frame_pred, velocity_pred


class OnsetsAndFramesMultiV3(OnsetsAndFrames2):
    def __init__(self, input_features, output_features, model_complexity=48,
                 onset_complexity=1,
                 n_instruments=15):
        nn.Module.__init__(self)
        model_size = model_complexity * 16
        sequence_model = lambda input_size, output_size: BiLSTM(input_size, output_size // 2)

        transformer_sequence_model = lambda output_size, n_layers, n_heads: \
            TransformerEncoder(TransformerEncoderLayer(d_model=output_size, nhead=n_heads),
                               num_layers=n_layers)

        onset_model_size = int(onset_complexity * model_size)
        print('onset model size', onset_model_size)
        print('onset output size', output_features * n_instruments)
        self.onset_stack = nn.Sequential(
            ConvStack(input_features, onset_model_size),
            # transformer_sequence_model(model_size, 1, 8),
            sequence_model(onset_model_size, onset_model_size),
            nn.Linear(onset_model_size, output_features * n_instruments),
            nn.Sigmoid()
        )
        # self.onset_stack = nn.Sequential(
        #     ConvStack(input_features, model_size),
        #     # transformer_sequence_model(model_size, 1, 8),
        #     sequence_model(model_size, model_size),
        #     nn.Linear(model_size, output_features * n_instruments),
        #     nn.Sigmoid()
        # )
        self.offset_stack = nn.Sequential(
            ConvStack(input_features, model_size),
            # transformer_sequence_model(model_size, 1, 8),
            sequence_model(model_size, model_size),
            nn.Linear(model_size, output_features),
            nn.Sigmoid()
        )
        self.frame_stack = nn.Sequential(
            ConvStack(input_features, model_size),
            nn.Linear(model_size, output_features),
            # nn.Linear(model_size, 88),
            nn.Sigmoid()
        )
        self.combined_stack = nn.Sequential(
            sequence_model(output_features * 3, model_size),
            nn.Linear(model_size, output_features),
            # sequence_model(output_features * 3, output_features * 3),
            # nn.Linear(output_features * 3, output_features),
            nn.Sigmoid()
        )

        self.velocity_stack = nn.Sequential(
            ConvStack(input_features, model_size),
            nn.Linear(model_size, output_features * n_instruments)
        )

    def forward(self, mel):
        onset_pred = self.onset_stack(mel)
        offset_pred = self.offset_stack(mel)
        activation_pred = self.frame_stack(mel)

        onset_detached = onset_pred.detach()
        shape = onset_detached.shape
        keys = MAX_MIDI - MIN_MIDI + 1
        new_shape = shape[: -1] + (shape[-1] // keys, keys)
        onset_detached = onset_detached.reshape(new_shape)
        onset_detached, _ = onset_detached.max(axis=-2)
        # onset_detached = onset_detached[..., -1]

        offset_detached = offset_pred.detach()

        combined_pred = torch.cat([onset_detached, offset_detached, activation_pred], dim=-1)
        frame_pred = self.combined_stack(combined_pred)
        velocity_pred = self.velocity_stack(mel)
        return onset_pred, offset_pred, activation_pred, frame_pred, velocity_pred


class OnsetsAndFramesMultiV12(OnsetsAndFramesMultiV3):
    def run_on_batch(self, batch, parallel_model=None, multi=False, positive_weight=10., inv_positive_weight=5.,
                     label_dropout=0.5, apply_vel_loss=True):
        audio_label = batch['audio']

        onset_label = batch['onset']
        offset_label = batch['offset']
        frame_label = batch['frame']
        if 'velocity' in batch:
            velocity_label = batch['velocity']
        # print('run', onset_label.shape)
        mel = melspectrogram(audio_label.reshape(-1, audio_label.shape[-1])[:, :-1]).transpose(-1, -2)
        # print('run mel', mel.shape)

        if not parallel_model:
            onset_pred, offset_pred, _, frame_pred, velocity_pred = self(mel)
        else:
            onset_pred, offset_pred, _, frame_pred, velocity_pred = parallel_model(mel)

        if multi:
            onset_pred = onset_pred[..., : N_KEYS]
            offset_pred = offset_pred[..., : N_KEYS]
            frame_pred = frame_pred[..., : N_KEYS]
            velocity_pred = velocity_pred[..., : N_KEYS]

        predictions = {
            'onset': onset_pred.reshape(*onset_label.shape),
            'offset': offset_pred.reshape(*offset_label.shape),
            'frame': frame_pred.reshape(*frame_label.shape),
            # 'velocity': velocity_pred.reshape(*velocity_label.shape)
        }
        if 'velocity' in batch:
            predictions['velocity'] = velocity_pred.reshape(*velocity_label.shape)

        # losses = {
        #     'loss/onset': F.binary_cross_entropy(predictions['onset'], onset_label),
        #     'loss/offset': F.binary_cross_entropy(predictions['offset'], offset_label),
        #     'loss/frame': F.binary_cross_entropy(predictions['frame'], frame_label),
        #     # 'loss/velocity': self.velocity_loss(predictions['velocity'], velocity_label, onset_label)
        # }
        # if 'velocity' in batch:
        #     losses['loss/velocity'] = self.velocity_loss(predictions['velocity'], velocity_label, onset_label)

        losses = {
            'loss/onset': F.binary_cross_entropy(predictions['onset'], onset_label, reduction='none'),
            'loss/offset': F.binary_cross_entropy(predictions['offset'], offset_label, reduction='mean'),
            'loss/frame': F.binary_cross_entropy(predictions['frame'], frame_label, reduction='mean'),
            # 'loss/velocity': self.velocity_loss(predictions['velocity'], velocity_label, onset_label)
        }
        if 'velocity' in batch and apply_vel_loss:
            losses['loss/velocity'] = self.velocity_loss(predictions['velocity'], velocity_label, onset_label)

        # inst_classes = 3.
        # positive_weight = (1 + inst_classes) / 2.
        # positive_weight *= 256 / HOP_LENGTH
        # positive_weight = inst_classes
        onset_mask = 1. * onset_label  # + 1.
        # positive_weight = 5.
        # positive_weight = 3.
        # positive_weight = 2.
        # positive_weight = 4.
        # positive_weight = 10. if HOP_LENGTH == 256 else 15.
        # positive_weight = 15. if HOP_LENGTH == 256 else 15.

        onset_mask[..., : -N_KEYS] *= (positive_weight - 1)
        # if HOP_LENGTH == 128:
        #     onset_mask[..., -N_KEYS:] *= 3.
        onset_mask[..., -N_KEYS:] *= (inv_positive_weight - 1)
        # onset_mask *= 2
        onset_mask += 1

        if label_dropout:
            # onset_mask *= (torch.randn(onset_mask.shape) >= 0).cuda()
            onset_mask *= (torch.rand(onset_mask.shape) >= label_dropout).cuda()

        # offset_mask = 1. * offset_label
        # offset_positive_weight = 10.
        # offset_mask *= (offset_positive_weight - 1)
        # offset_mask += 1.

        # frame_mask = 1. * frame_label
        # frame_positive_weight = 2.
        # frame_mask *= (frame_positive_weight - 1)
        # frame_mask += 1.

        # for loss_key, mask in zip(['onset', 'offset', 'frame'], [onset_mask, offset_mask, frame_mask]):
        #     losses['loss/' + loss_key] = (mask * losses['loss/' + loss_key]).mean()
        for loss_key, mask in zip(['onset'], [onset_mask]):
            losses['loss/' + loss_key] = (mask * losses['loss/' + loss_key]).mean()
        for loss_key in ['offset', 'frame']:
            print(loss_key, 'loss shape', losses['loss/' + loss_key].shape)
        #     losses['loss/' + loss_key] = (losses['loss/' + loss_key]).mean()

        # else:
        #     losses = {
        #         'loss/onset': F.binary_cross_entropy(predictions['onset'], onset_label, reduction='none'),
        #         'loss/offset': F.binary_cross_entropy(predictions['offset'], offset_label, reduction='none'),
        #         'loss/frame': F.binary_cross_entropy(predictions['frame'], frame_label, reduction='none'),
        #         # 'loss/velocity': self.velocity_loss(predictions['velocity'], velocity_label, onset_label)
        #     }
        #     if 'velocity' in batch:
        #         losses['loss/velocity'] = self.velocity_loss(predictions['velocity'], velocity_label, onset_label)
        #
        #     onset_mask = batch['onsets_mask']
        #     frame_mask = batch['frames_mask']
        #     offset_mask = batch['offsets_mask']
        #
        #     for loss_key, mask in zip(['onset', 'offset', 'frame'], [onset_mask, frame_mask, offset_mask]):
        #         losses['loss/' + loss_key] = (mask * losses['loss/' + loss_key]).mean()
        #
        #     for pred, label, mask in zip(['onset', 'frame', 'offset'],
        #                                  [onset_label, frame_label, offset_label],
        #                                  [onset_mask, frame_mask, offset_mask]):
        #         if pred == 'onset':
        #             compress_functions = [compress_tensor_across_octave,
        #                                  compress_tensor_across_octave_and_instrument,
        #                                  compress_tensor_across_instrument
        #                                  ]
        #         else:
        #             compress_functions = [compress_tensor_across_instrument]
        #
        #         for comress_func in compress_functions:
        #             inv_pred = comress_func(predictions[pred])
        #             inv_label = comress_func(label)
        #             inv_mask = comress_func(mask, apply_min=True)
        #             # print('mask size', inv_mask.sum() / inv_mask.nelement())
        #             octave_inv_loss = F.binary_cross_entropy(inv_pred, inv_label, reduction='none')
        #             losses['loss/' + pred] += (inv_mask * octave_inv_loss).mean()

        return predictions, losses


class OnsetsAndFrames320(OnsetsAndFramesMultiV3):
    def __init__(self, input_features, output_features, model_complexity=48,
                 onset_complexity=1,
                 n_instruments=15):
        nn.Module.__init__(self)
        model_size = model_complexity * 16
        sequence_model = lambda input_size, output_size: BiLSTM(input_size, output_size // 2)

        onset_model_size = int(onset_complexity * model_size)

        self.onset_stack = nn.Sequential(
            ConvStack(input_features, onset_model_size),
            # transformer_sequence_model(model_size, 1, 8),
            sequence_model(onset_model_size, onset_model_size),
            nn.Linear(onset_model_size, output_features * n_instruments),
            nn.Sigmoid()
        )

        self.offset_stack = nn.Sequential(
            ConvStack(input_features, model_size),
            # transformer_sequence_model(model_size, 1, 8),
            sequence_model(model_size, model_size),
            nn.Linear(model_size, output_features),
            nn.Sigmoid()
        )
        self.frame_stack = nn.Sequential(
            ConvStack(input_features, model_size),
            nn.Linear(model_size, output_features),
            # nn.Linear(model_size, 88),
            nn.Sigmoid()
        )
        self.combined_stack = nn.Sequential(
            sequence_model(output_features * 3, model_size),
            nn.Linear(model_size, output_features),
            # sequence_model(output_features * 3, output_features * 3),
            # nn.Linear(output_features * 3, output_features),
            nn.Sigmoid()
        )

    def forward(self, mel):
        onset_pred = self.onset_stack(mel)
        offset_pred = self.offset_stack(mel)
        activation_pred = self.frame_stack(mel)

        onset_detached = onset_pred.detach()
        shape = onset_detached.shape
        keys = MAX_MIDI - MIN_MIDI + 1
        new_shape = shape[: -1] + (shape[-1] // keys, keys)
        onset_detached = onset_detached.reshape(new_shape)
        onset_detached, _ = onset_detached.max(axis=-2)
        # onset_detached = onset_detached[..., -1]

        offset_detached = offset_pred.detach()

        combined_pred = torch.cat([onset_detached, offset_detached, activation_pred], dim=-1)
        frame_pred = self.combined_stack(combined_pred)
        return onset_pred, offset_pred, activation_pred, frame_pred

    def run_on_batch(self, batch, mel_spec, parallel_model=None, positive_weight=10., inv_positive_weight=5.,
                     label_dropout=0.15, frame_dropout=0.5):
        audio_label = batch['audio']
        onset_label = batch['onset']
        offset_label = batch['offset']
        frame_label = batch['frame']
        # print('frame percent', frame_label.sum() / torch.numel(frame_label), 'shape', frame_label.shape)
        # print('frame percent 1', (frame_label == 1).sum() / torch.numel(frame_label), 'shape', frame_label.shape)
        # print('frame percent 0', (frame_label == 0).sum() / torch.numel(frame_label), 'shape', frame_label.shape)

        mel = mel_spec(audio_label.reshape(-1, audio_label.shape[-1])[:, :-1]).transpose(-1, -2)
        # print('transcriber mel shape', mel.shape)

        if not parallel_model:
            onset_pred, offset_pred, _, frame_pred = self(mel)
        else:
            onset_pred, offset_pred, _, frame_pred = parallel_model(mel)

        # print('frame_pred percent', frame_pred.sum() / torch.numel(frame_pred), 'shape', frame_pred.shape)
        # print('frame_pred percent 1', (frame_pred == 1).sum() / torch.numel(frame_pred), 'shape', frame_pred.shape)
        # print('frame_pred percent 0', (frame_pred == 0).sum() / torch.numel(frame_pred), 'shape', frame_pred.shape)
        # print('frame shape', frame_pred.shape, frame_label.shape)

        predictions = {
            'onset': onset_pred.reshape(*onset_label.shape),
            'offset': offset_pred.reshape(*offset_label.shape),
            'frame': frame_pred.reshape(*frame_label.shape),
        }

        losses = {
            'loss/onset': F.binary_cross_entropy(predictions['onset'], onset_label, reduction='none'),
            'loss/offset': F.binary_cross_entropy(predictions['offset'], offset_label, reduction='mean'),
            # 'loss/frame': F.binary_cross_entropy(predictions['frame'], frame_label, reduction='mean'),
            'loss/frame': F.binary_cross_entropy(predictions['frame'], frame_label, reduction='none'),
        }

        onset_mask = 1. * onset_label  # + 1.
        onset_mask[..., : -N_KEYS] *= (positive_weight - 1)
        onset_mask[..., -N_KEYS:] *= (inv_positive_weight - 1)
        onset_mask += 1

        frame_mask = torch.ones_like(frame_label)

        if label_dropout:
            onset_mask *= (torch.rand(onset_mask.shape) >= label_dropout).cuda()
        if frame_dropout:
            frame_mask *= (torch.rand(frame_mask.shape) >= frame_dropout).cuda()

        for loss_key, mask in zip(['onset', 'frame'], [onset_mask, frame_mask]):
            losses['loss/' + loss_key] = (mask * losses['loss/' + loss_key]).mean()
        # for loss_key, mask in zip(['onset'], [onset_mask]):
        #     losses['loss/' + loss_key] = (mask * losses['loss/' + loss_key]).mean()

        return predictions, losses


def duplicate_linear(linear, n):
    A, b = linear.parameters()
    in_features, out_features = linear.in_features, linear.out_features
    layer_new = torch.nn.Linear(in_features, n * out_features)
    A_new, b_new = layer_new.parameters()
    A_new.requires_grad, b_new.requires_grad = False, False
    for j in range(n):
        A_new[j * out_features: (j + 1) * out_features, :] = A.detach().clone()
        b_new[j * out_features: (j + 1) * out_features] = b.detach().clone()
    A_new.requires_grad, b_new.requires_grad = True, True
    return layer_new


def duplicate_linear_pop(linear, n):
    A, b = linear.parameters()
    num_notes = N_KEYS
    assert A.shape[0] % num_notes == 0
    num_prev_inst = A.shape[0] // num_notes - 1
    A_pitch = A.detach()[-num_notes:, :]
    b_pitch = b.detach()[-num_notes:]
    in_features, out_features = linear.in_features, linear.out_features
    layer_new = torch.nn.Linear(in_features, n * num_notes)
    A_new, b_new = layer_new.parameters()
    A_new.requires_grad, b_new.requires_grad = False, False
    for j in range(n):
        if j < num_prev_inst:
            A_new[j * num_notes: (j + 1) * num_notes, :] = A.detach()[j * num_notes: (j + 1) * num_notes, :].clone()
            b_new[j * num_notes: (j + 1) * num_notes] = b.detach()[j * num_notes: (j + 1) * num_notes].clone()
        else:
            A_new[j * num_notes: (j + 1) * num_notes, :] = A_pitch.clone()
            b_new[j * num_notes: (j + 1) * num_notes] = b_pitch.detach().clone()
        # A_new[j * out_features: (j + 1) * out_features, :] = A.detach().clone()
        # b_new[j * out_features: (j + 1) * out_features] = b.detach().clone()
    A_new.requires_grad, b_new.requires_grad = True, True
    return layer_new


def load_weights(model, old_model, n_instruments):
    for i in range(len(model.onset_stack)):
        if i < len(model.onset_stack) - 2:
            model.onset_stack[i].load_state_dict(old_model.onset_stack[i].state_dict())
        elif i < len(model.onset_stack) - 1:
            linear = old_model.onset_stack[i]
            layer_new = duplicate_linear(linear, n_instruments)
            model.onset_stack[i].load_state_dict(layer_new.state_dict())

    for i in range(len(model.frame_stack)):
        if i < len(model.frame_stack) - 1:
            model.frame_stack[i].load_state_dict(old_model.frame_stack[i].state_dict())

    for i in range(len(model.combined_stack)):
        if i < len(model.combined_stack) - 1:
            model.combined_stack[i].load_state_dict(old_model.combined_stack[i].state_dict())

    for i in range(len(model.offset_stack)):
        if i < len(model.offset_stack) - 1:
            model.offset_stack[i].load_state_dict(old_model.offset_stack[i].state_dict())

    for i in range(len(model.velocity_stack)):
        if i < len(model.velocity_stack) - 1:
            model.velocity_stack[i].load_state_dict(old_model.velocity_stack[i].state_dict())
        elif i < len(model.velocity_stack):
            linear = old_model.velocity_stack[i]
            layer_new = duplicate_linear(linear, n_instruments)
            model.velocity_stack[i].load_state_dict(layer_new.state_dict())


def load_weights_pop(model, old_model, n_instruments):
    for i in range(len(model.onset_stack)):
        if i < len(model.onset_stack) - 2:
            model.onset_stack[i].load_state_dict(old_model.onset_stack[i].state_dict())
        elif i < len(model.onset_stack) - 1:
            linear = old_model.onset_stack[i]
            layer_new = duplicate_linear_pop(linear, n_instruments)
            model.onset_stack[i].load_state_dict(layer_new.state_dict())

    for i in range(len(model.frame_stack)):
        if i < len(model.frame_stack) - 1:
            model.frame_stack[i].load_state_dict(old_model.frame_stack[i].state_dict())

    for i in range(len(model.combined_stack)):
        if i < len(model.combined_stack) - 1:
            model.combined_stack[i].load_state_dict(old_model.combined_stack[i].state_dict())

    for i in range(len(model.offset_stack)):
        if i < len(model.offset_stack) - 1:
            model.offset_stack[i].load_state_dict(old_model.offset_stack[i].state_dict())

    for i in range(len(model.velocity_stack)):
        if i < len(model.velocity_stack) - 1:
            model.velocity_stack[i].load_state_dict(old_model.velocity_stack[i].state_dict())
        elif i < len(model.velocity_stack):
            linear = old_model.velocity_stack[i]
            layer_new = duplicate_linear_pop(linear, n_instruments)
            model.velocity_stack[i].load_state_dict(layer_new.state_dict())


def modulated_load_weights(model: ModulatedOnsetsAndFrames, old_model: OnsetsAndFrames, n_instruments):
    model.onset_stack.conv_stack.load_state_dict(old_model.onset_stack[0].state_dict())
    model.onset_stack.seq_model.load_state_dict(old_model.onset_stack[1].state_dict())
    model.onset_stack.fc[0].load_state_dict(old_model.onset_stack[2].state_dict())

    for i in range(len(model.frame_stack)):
        if i < len(model.frame_stack) - 1:
            model.frame_stack[i].load_state_dict(old_model.frame_stack[i].state_dict())

    for i in range(len(model.combined_stack)):
        if i < len(model.combined_stack) - 1:
            model.combined_stack[i].load_state_dict(old_model.combined_stack[i].state_dict())

    for i in range(len(model.offset_stack)):
        if i < len(model.offset_stack) - 1:
            model.offset_stack[i].load_state_dict(old_model.offset_stack[i].state_dict())

    for i in range(len(model.velocity_stack)):
        if i < len(model.velocity_stack) - 1:
            model.velocity_stack[i].load_state_dict(old_model.velocity_stack[i].state_dict())
        elif i < len(model.velocity_stack):
            linear = old_model.velocity_stack[i]
            layer_new = duplicate_linear(linear, n_instruments)
            model.velocity_stack[i].load_state_dict(layer_new.state_dict())