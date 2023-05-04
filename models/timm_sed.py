import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchlibrosa.stft import LogmelFilterBank, Spectrogram
from torchlibrosa.augmentation import SpecAugmentation


class AttBlockV2(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        activation="linear"
    ):
        super().__init__()
        
        self.activation = activation
        self.att = nn.Conv1d(
            in_channels=in_features,
            out_channels=out_features,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True
        )
        self.cla = nn.Conv1d(
            in_channels=in_features,
            out_channels=out_features,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )
        
        self.init_weights()
        
    def init_weights(self):
        self.init_layer(self.att)
        self.init_layer(self.cla)

    def init_layer(self, layer):
        nn.init.xavier_uniform_(layer.weight)

        if hasattr(layer, "bias"):
            if layer.bias is not None:
                layer.bias.data.fill_(0.)
        
    def forward(self, x):
        # x: (n_samples, n_in, n_time) (B, D, T)
        norm_att = torch.softmax(torch.tanh(self.att(x)), dim=-1)
        cla = self.nonlinear_transform(self.cla(x))
        x = torch.sum(norm_att * cla, dim=2)
        return x, norm_att, cla

    def nonlinear_transform(self, x):
        if self.activation == 'linear':
            return x
        elif self.activation == 'sigmoid':
            return torch.sigmoid(x)


# TimmSED model
class TimmSED(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # Spectrogram extractor
        self.spectrogram_extractor = Spectrogram(
            n_fft=cfg.n_fft, 
            hop_length=cfg.hop_length,
            win_length=cfg.n_fft, 
            window="hann", 
            center=True,
            pad_mode="reflect",
            freeze_parameters=True
        )

        # Logmel feature extractor
        self.logmel_extractor = LogmelFilterBank(
            sr=cfg.sample_rate, 
            n_fft=cfg.n_fft,
            n_mels=cfg.n_mels, 
            fmin=cfg.fmin, 
            fmax=cfg.fmax, 
            ref=1.0, 
            amin=1e-10, 
            top_db=None,
            freeze_parameters=True
        )

        # Spec augmenter
        self.spec_augmenter = SpecAugmentation(time_drop_width=64, time_stripes_num=2,
                                               freq_drop_width=8, freq_stripes_num=2)

        self.bn0 = nn.BatchNorm2d(cfg.n_mels)

        base_model = timm.create_model(
            cfg.base_model_name, 
            pretrained=cfg.pretrained, 
            in_chans=cfg.in_channels
        )
        layers = list(base_model.children())[:-2]
        self.encoder = nn.Sequential(*layers)

        if hasattr(base_model, "fc"):
            in_features = base_model.fc.in_features
        else:
            in_features = base_model.classifier.in_features
        self.fc1 = nn.Linear(in_features, in_features, bias=True)
        self.att_block = AttBlockV2(
            in_features, cfg.num_classes, activation="sigmoid")

        self.init_weight()

    def init_weight(self):
        self.init_layer(self.fc1)
        self.init_bn(self.bn0)

    def init_layer(self, layer):
        nn.init.xavier_uniform_(layer.weight)

        if hasattr(layer, "bias"):
            if layer.bias is not None:
                layer.bias.data.fill_(0.)

    def init_bn(self, bn):
        bn.bias.data.fill_(0.)
        bn.weight.data.fill_(1.0)

    def interpolate(self, x: torch.Tensor, ratio: int):
        """Interpolate data in time domain. This is used to compensate the
        resolution reduction in downsampling of a CNN.
        Args:
        x: (batch_size, time_steps, classes_num)
        ratio: int, ratio to interpolate
        Returns:
        upsampled: (batch_size, time_steps * ratio, classes_num)
        """
        (batch_size, time_steps, classes_num) = x.shape
        upsampled = x[:, :, None, :].repeat(1, 1, ratio, 1)
        upsampled = upsampled.reshape(batch_size, time_steps * ratio, classes_num)
        return upsampled


    def pad_framewise_output(self, framewise_output: torch.Tensor, frames_num: int):
        """Pad framewise_output to the same length as input frames. The pad value
        is the same as the value of the last frame.
        Args:
        framewise_output: (batch_size, frames_num, classes_num)
        frames_num: int, number of frames to pad
        Outputs:
        output: (batch_size, frames_num, classes_num)
        """
        output = F.interpolate(
            framewise_output.unsqueeze(1),
            size=(frames_num, framewise_output.size(2)),
            align_corners=True,
            mode="bilinear").squeeze(1)

        return output

    def forward(self, input):
        x = self.spectrogram_extractor(input)   # (batch_size, 1, time_steps, freq_bins)
        x = self.logmel_extractor(x)    # (batch_size, 1, time_steps, mel_bins)

        frames_num = x.shape[2]

        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)

        if self.training:
            x = self.spec_augmenter(x)

        x = x.transpose(2, 3)
        # (batch_size, channels, freq, frames)
        x = self.encoder(x)

        # (batch_size, channels, frames)
        x = torch.mean(x, dim=2)

        # channel smoothing
        x1 = F.max_pool1d(x, kernel_size=3, stride=1, padding=1)
        x2 = F.avg_pool1d(x, kernel_size=3, stride=1, padding=1)
        x = x1 + x2

        x = F.dropout(x, p=0.5, training=self.training)
        x = x.transpose(1, 2)
        x = F.relu_(self.fc1(x))
        x = x.transpose(1, 2)
        x = F.dropout(x, p=0.5, training=self.training)
        
        (clipwise_output, norm_att, segmentwise_output) = self.att_block(x)
        logit = torch.sum(norm_att * self.att_block.cla(x), dim=2)
        segmentwise_logit = self.att_block.cla(x).transpose(1, 2)
        segmentwise_output = segmentwise_output.transpose(1, 2)

        interpolate_ratio = frames_num // segmentwise_output.size(1)

        # Get framewise output
        framewise_output = self.interpolate(segmentwise_output,
                                       interpolate_ratio)
        framewise_output = self.pad_framewise_output(framewise_output, frames_num)

        framewise_logit = self.interpolate(segmentwise_logit, interpolate_ratio)
        framewise_logit = self.pad_framewise_output(framewise_logit, frames_num)

        output_dict = {
            "framewise_output": framewise_output,
            "segmentwise_output": segmentwise_output,
            "logit": logit,
            "framewise_logit": framewise_logit,
            "clipwise_output": clipwise_output
        }

        return output_dict