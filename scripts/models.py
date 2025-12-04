import torch.nn as nn

class CNN(nn.Module):

    def __init__(self, 
                 num_classes=8,
                 sr=44100, 
                 duration=3, 
                 clip_duration=None,
                 mel_bins=128, 
                 hop_length=512,
                 channel_size=16, 
                 kernel_size=3, 
                 num_conv_layers=4, 
                 p_dropout=0.2):

      super().__init__()

      self.sr = sr
      self.hop_length = hop_length
      # clip_duration overrides duration if provided to make switching between 3s and 30s explicit.
      self.duration = clip_duration if clip_duration is not None else duration
      self.time = int(self.sr * self.duration / self.hop_length)
      self.mel_bins = mel_bins
      self.num_classes = num_classes

      self.channel_size = channel_size
      self.kernel_size = kernel_size
      self.num_conv_layers = num_conv_layers
      self.p_dropout= p_dropout

      self.layers = [
          nn.Conv2d(1, self.channel_size, kernel_size=self.kernel_size, padding=1),
          nn.ReLU(),
          nn.MaxPool2d(kernel_size=2),
          nn.Dropout(self.p_dropout)
      ]

      for i in range(1, self.num_conv_layers):
        self.layers += [
            nn.Conv2d(self.channel_size * (2**(i-1)), self.channel_size * (2**i), kernel_size=self.kernel_size, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(self.p_dropout)
        ]

      self.ConvLayers = nn.Sequential(*self.layers)

      final_channels = self.channel_size * (2**(self.num_conv_layers-1))
      final_mel_bins = self.mel_bins // (2**self.num_conv_layers)
      final_time = self.time // (2**self.num_conv_layers)

      self.Classifier_Logits = nn.Sequential(
          nn.Linear(final_channels * final_mel_bins * final_time, 128),
          nn.ReLU(),
          nn.Linear(128,self.num_classes)
      )

    def forward(self, x): # x.shape : (B, 1, n_mels=128, time=258)
      bs = x.size(0)

      x = self.ConvLayers(x) # (B, final_channels, final_mel_bins, final_time)

      x_flatten = x.view(bs, -1) # x is flatten

      logits = self.Classifier_Logits(x_flatten)

      return logits
