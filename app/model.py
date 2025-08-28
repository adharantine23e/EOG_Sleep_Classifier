import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Any

class BiLSTM(nn.Module):
  def __init__(self, input_size: int, 
               hidden_size: int, num_layers: int, 
               bidirectional: bool, batch_first: bool, device: torch.device):
    super(BiLSTM, self).__init__()
    self.lstm = nn.LSTM(input_size = input_size,
                      hidden_size = hidden_size,
                      num_layers = num_layers,
                      bidirectional = bidirectional,
                      batch_first = batch_first)
    self.device = device

  def forward(self, x):
      h0 = torch.zeros((self.lstm.num_layers * (2 if self.lstm.bidirectional else 1),
                        x.size(0),
                        self.lstm.hidden_size)).to(self.device)
      c0 = torch.zeros((self.lstm.num_layers * (2 if self.lstm.bidirectional else 1),
                        x.size(0),
                        self.lstm.hidden_size)).to(self.device)

      output, _ = self.lstm(x, (h0, c0))
      if self.lstm.bidirectional == True :
        output_f = output[:, -1, :self.lstm.hidden_size]
        output_b = output[:, 0, :self.lstm.hidden_size]
        output = torch.cat((output_f, output_b), dim = 1)
      else:
        output = output[:, -1,:]
      return output

class PlainGRU(nn.Module):
  def __init__(self, input_size: int, 
               hidden_size: int, num_layers: int, 
               bidirectional: bool, batch_first: bool, device: torch.device):
    super(PlainGRU, self).__init__()
    self.gru = nn.GRU(input_size = input_size,
                      hidden_size = hidden_size,
                      num_layers = num_layers,
                      bidirectional = bidirectional,
                      batch_first = batch_first)
    self.device = device
  def init_hidden(self, x):
        h0 = torch.zeros((self.gru.num_layers * (2 if self.gru.bidirectional else 1),
                          x.size(0),
                          self.gru.hidden_size)).to(self.device) #Remember to fix thix back to cuda()
        return h0

  def forward(self, x):
      hidden = self.init_hidden(x)
      gru_output, hidden = self.gru(x, hidden)
      if self.gru.bidirectional == True :
        output_f = gru_output[:, -1, :self.gru.hidden_size]
        output_b = gru_output[:, 0, self.gru.hidden_size:]
        output = torch.cat((output_f, output_b), dim = 1)
      else:
        output = gru_output[:, -1,:]
      return output

class LSTMAttention(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int = 1):
        super(LSTMAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.attention = nn.Linear(hidden_dim, 1)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        attention_weights = torch.softmax(self.attention(lstm_out).squeeze(-1), dim=-1)
        context_vector = torch.sum(lstm_out * attention_weights.unsqueeze(-1), dim=1)
        out = self.fc(context_vector)
        return out

class SelfAttention(nn.Module):
  def __init__(self, feature_size: int):
    super(SelfAttention, self).__init__()
    self.feature_size = feature_size

    # Linear transformation for Q, K, V from the same source
    self.key = nn.Linear(feature_size, feature_size)
    self.query = nn.Linear(feature_size, feature_size)
    self.value = nn.Linear(feature_size, feature_size)
  def forward(self, x, mask= None):
    keys = self.key(x)
    queries = self.query(x)
    values = self.value(x)

    scores = torch.matmul(queries, keys.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.feature_size, dtype= torch.float32))

    if mask is not None:
      scores = scores.masked_fill(mask == 0, -1e9)

    attention_weights = F.softmax(scores, dim = -1)
    attention_output = torch.matmul(attention_weights, values)
    return attention_output, attention_weights

class Conv_layer(nn.Module):
  def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation):
    super(Conv_layer, self).__init__()
    self.Conv = nn.Conv1d(in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias = False)
    self.bn = nn.BatchNorm1d(num_features=out_channels)
    self.relu = nn.PReLU()
  def forward(self, x):
    x = self.Conv(x)
    x = self.bn(x)
    x = self.relu(x)
    return x

class Residual_Block(nn.Module):
  def __init__(self, in_channels, out_channels, kernel_size, stride =1, padding= "same"):
    super (Residual_Block, self).__init__()
    self.Conv1 = nn.Conv1d(in_channels=in_channels,
                           out_channels=out_channels,
                           kernel_size=kernel_size,
                           stride=stride,
                           padding = padding,
                           bias = False)
    self.bn1 = nn.BatchNorm1d(num_features=out_channels)
    self.relu =  nn.PReLU()
    self.Conv2 = nn.Conv1d(in_channels=out_channels,
                           out_channels=out_channels,
                           kernel_size=kernel_size,
                           stride=1,
                           padding = padding,
                           bias= False)
    self.bn2 = nn.BatchNorm1d(num_features=out_channels)
    if in_channels != out_channels:
      self.shortcut = nn.Sequential(


          nn.Conv1d(in_channels, out_channels, kernel_size = 1, stride = stride),
          nn.BatchNorm1d(out_channels),
          nn.ReLU(),
          nn.Conv1d(out_channels, out_channels, kernel_size = 3, stride = stride),
          nn.BatchNorm1d(out_channels),
          nn.ReLU(),
          nn.Conv1d(out_channels, out_channels, kernel_size = 1, stride = stride),
          nn.BatchNorm1d(out_channels),
      )
    else:
      self.shortcut = nn.Identity()
  def forward(self, x):
      shortcut = self.shortcut(x)
      x = self.Conv1(x)
      x = self.bn1(x)
      x = self.relu(x)
      x = self.Conv2(x)
      x = self.bn2(x)
      out = x + shortcut
      out = self.relu(out)
      return out

class SELayer(nn.Module):
    def __init__(self, channel, reduction: int = 16):
        super(SELayer, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction,bias=False),
            nn.PReLU(),
            nn.Linear(channel // reduction, channel,bias=False),
            nn.Tanh()
        )

        self.avg_pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        initital = x
        b, c, _ = initital.size()
        y = self.avg_pool(initital).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)




class Recombination_Calibration(nn.Module):
  def __init__(self, in_channels, out_channels, kernel_size, downsample, stride = 1, reduction = 16):
      super(Recombination_Calibration, self).__init__()
      self.Conv1= nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size =kernel_size, stride =1 , bias  =False)
      self.Conv2= nn.Conv1d(in_channels=out_channels,out_channels=out_channels,  kernel_size =kernel_size, stride =1, bias = False)
      self.bn1 = nn.BatchNorm1d(out_channels)
      self.bn2 = nn.BatchNorm1d(out_channels)
      self.relu =   nn.PReLU()
      self.se = SELayer(out_channels, reduction)
      self.downsample = downsample
      self.stride = stride
  def forward(self, x):
    residual = x
    out = self.Conv1(x)
    out = self.bn1(out)
    out = self.relu(out)
    out = self.Conv2(out)
    out = self.bn2(out)

    out = self.se(out)
    if self.downsample is not None:
            residual = self.downsample(x)
    out += residual
    out = self.relu(out)
    return out


class EOGClassifier(nn.Module):
    def __init__(self, num_class):
        super(EOGClassifier, self).__init__()
        self.sr = 100
        self.num_classes = num_class
        self.channel = 64
        self.kernel_mult_p1 = 0.3
        self.kernel_mult_p2 = 3
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.path1 = nn.Sequential(
            Conv_layer(in_channels = 1, out_channels = self.channel, kernel_size =int(self.kernel_mult_p1 * self.sr),
                       stride = 3, padding = "valid", dilation =1),
            nn.MaxPool1d(4, stride=2, padding= 0),
            nn.Dropout(0.5),
            Conv_layer(in_channels = self.channel, out_channels = self.channel*2, 
                       kernel_size =int(0.1 * self.kernel_mult_p1 * self.sr),stride = 1, padding = 'same', dilation =1),

            self.make_layer_for_residual(block = Residual_Block,inplanes = 2*self.channel, 
                                         outplanes =2*self.channel, kernel_size =int(0.1 * self.kernel_mult_p1 * self.sr) ,num_blocks = 2),
            nn.MaxPool1d(2, stride=2, padding =0),
            #nn.Flatten()
        )
        self.path2 = nn.Sequential(
            Conv_layer(in_channels = 1, out_channels = self.channel, kernel_size =int(self.kernel_mult_p2 * self.sr),
                       stride = int(0.1*self.sr), padding = 200, dilation =1),
            nn.MaxPool1d(3, stride=3, padding =0 ),
            nn.Dropout(0.5),
            Conv_layer(in_channels = self.channel, out_channels = self.channel*2, 
                       kernel_size =int(0.1 * self.kernel_mult_p2 * self.sr),stride = 1, padding = 'same', dilation =1),

            self.make_layer_for_residual(block = Residual_Block,inplanes = 2*self.channel, 
                                         outplanes =2*self.channel, kernel_size =int(0.1 * self.kernel_mult_p2 * self.sr) ,num_blocks = 2),
            nn.MaxPool1d(2, stride=2, padding =0),
            #nn.Flatten(),
        )
        self.shortcut = nn.Identity()
        self.FRC2 = self.make_layer_for_calibration(block = Recombination_Calibration, inplanes = 2*self.channel, outplanes =self.channel, kernel_size =1 ,num_blocks = 1)


        self.sequence_learning = BiLSTM(input_size = 298, hidden_size =128, num_layers = 1, bidirectional = True, batch_first = True, device = self.device)
        


        self.classification = nn.Sequential(

            nn.Dropout(0.5),
            nn.Linear(256, self.num_classes, bias = False))

    def make_layer_for_calibration(self, block,inplanes, outplanes, kernel_size ,num_blocks, stride =1):
      downsample = None
      if stride != 1 or inplanes > outplanes*num_blocks:
        downsample = nn.Sequential(
            nn.Conv1d(inplanes, outplanes, kernel_size = 1, stride = stride, bias = False),
            nn.BatchNorm1d(outplanes),
        )
      layers = []
      layers.append(block(inplanes, outplanes, kernel_size, downsample))

      for i in range(1, num_blocks):
        layers.append(block(outplanes, outplanes, kernel_size))
      return nn.Sequential(*layers)


    def make_layer_for_residual(self, block,inplanes, outplanes, kernel_size ,num_blocks, stride =1):
      layers = []
      layers.append(block(inplanes, outplanes, kernel_size))
      for i in range(1, num_blocks):
        layers.append(block(outplanes, outplanes, kernel_size))
      return nn.Sequential(*layers)

    def forward(self, x):
        x1 = self.path1(x)
        x2 = self.path2(x)

        # Concatenate outputs from path1 and path2
        x3 = torch.cat((x1, x2), dim=2)
        x3 = self.FRC2(x3)
        # Apply sequence learning
        output_sequence = self.sequence_learning(x3)

        output = self.classification(output_sequence)
        return output