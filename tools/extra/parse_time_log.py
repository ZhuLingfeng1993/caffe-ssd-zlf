import re
import argparse
from collections import OrderedDict

import matplotlib.pyplot as plt

def parse_log(logfile_path):
  """
  Parse log file
  """
  
  regex_layer_forward_time = re.compile(r'\b(\S+)\b\tforward: ([\.\deE+-]+) ms.')
  
  row_dict_list = []
  layer_name_list = []
  layer_forward_time_list = []
  
  with open(logfile_path) as f:
    for line in f:
      layer_forward_time_match = regex_layer_forward_time.search(line)
      if layer_forward_time_match:
        layer_name = layer_forward_time_match.group(1)
        layer_forward_time = float(layer_forward_time_match.group(2))
        layer_name_list.append(layer_name)
        layer_forward_time_list.append(layer_forward_time)
        #print(layer_name,layer_forward_time)
        row = OrderedDict([
          ('Layer', layer_name),
          ('ForwardTime',layer_forward_time)
        ])
        row_dict_list.append(row)
      else:
        #print('match failed!')
        pass
    
  return row_dict_list, layer_name_list, layer_forward_time_list

def plot_log_SSD(layer_name_list, layer_forward_time_list, base_net_name):
  """
  plot basenet layer and SSD layers time in two bar subplot 
  """
  base_net_layer_num = get_base_net_layer_num(base_net_name)
  total_layer_num = len(layer_name_list)
  max_layer_time = max(layer_forward_time_list)
  ssd_layers_num = total_layer_num - base_net_layer_num
  subplot_num = '2'
  if ssd_layers_num > 60:
    subplot_num = '3'
  
  fig=plt.figure(figsize=(22,13.5))#adjust figsiez if needed
  fig.subplots_adjust(wspace=0.5)
  if subplot_num == '3':
    fig.subplots_adjust(wspace=1)
  
  #plot base net time
  ax1 = fig.add_subplot(int('1'+subplot_num+'1'))
  plt.xlim(0,max_layer_time)
  plt.ylim(0,base_net_layer_num)
  plt.barh(range(base_net_layer_num), layer_forward_time_list[:base_net_layer_num])
  plt.yticks(range(base_net_layer_num), layer_name_list[:base_net_layer_num],  rotation=0)
  plt.title(base_net_name+':baseNet')
  plt.xlabel('time/ms')
  plt.ylabel('layer')
  
  #plot SSD layers time
  if subplot_num == '2':
    ax2 = fig.add_subplot(122)
    plt.xlim(0,max_layer_time)
    plt.ylim(0,ssd_layers_num)
    plt.barh(range(ssd_layers_num), layer_forward_time_list[base_net_layer_num:])
    plt.yticks(range(ssd_layers_num), layer_name_list[base_net_layer_num:],  rotation=0)
    plt.title(base_net_name+':SSD Layers')
    plt.xlabel('time/ms')
    plt.ylabel('layer')
  else:
    layer_num1 = ssd_layers_num/2
    layer_num2 = ssd_layers_num-layer_num1
    print(layer_num1,layer_num2)
    
    ax2 = fig.add_subplot(132)
    plt.xlim(0,max_layer_time)
    plt.ylim(0,layer_num1)
    layer_index_start = base_net_layer_num
    layer_index_end = base_net_layer_num + layer_num1
    plt.barh(range(layer_num1), layer_forward_time_list[layer_index_start:layer_index_end])
    plt.yticks(range(layer_num1), layer_name_list[layer_index_start:layer_index_end],  rotation=0)
    plt.title(base_net_name+':SSD Layers')
    plt.xlabel('time/ms')
    plt.ylabel('layer')
    
    ax3 = fig.add_subplot(133)
    plt.xlim(0,max_layer_time)
    plt.ylim(0,layer_num2)
    layer_index_start = base_net_layer_num+layer_num1
    plt.barh(range(layer_num2), layer_forward_time_list[layer_index_start:])
    plt.yticks(range(layer_num2), layer_name_list[layer_index_start:],  rotation=0)
    plt.title(base_net_name+':SSD Layers')
    plt.xlabel('time/ms')
    plt.ylabel('layer')
  
  plt.show()
  
def get_base_net_layer_num(base_net_name):
  if base_net_name == 'MobileNet':
    base_net_layer_num = 58
  elif base_net_name == 'VGGNet':
    base_net_layer_num = 39
  return base_net_layer_num

def print_net_time(time_forward_dict_list, layer_forward_time_list, base_net_name):
#  time_forward_dict_list.sort(key=lambda item: item['ForwardTime'], reverse=True)
  # print each layer forward time
  for idx,item in enumerate(time_forward_dict_list):
    print(idx, item['Layer'], item['ForwardTime'])
  
  base_net_layer_num = get_base_net_layer_num(base_net_name)
  base_net_time = sum(layer_forward_time_list[:base_net_layer_num])
  net_time = sum(layer_forward_time_list)
  print('net forward time = %f ms' % net_time)
  print('base net forward time = %f ms' % base_net_time)
  
  
def parse_args():
  description = ('Parse a Caffe timing log and plot bar figure')
  parser = argparse.ArgumentParser(description=description)
  
  parser.add_argument('logfile_path', help = 'Path to log file')
  parser.add_argument('base_net_name', help = 'base_net_name: VGGNet or MobileNet')
  
  args = parser.parse_args()
  return args
  
def main():
  args = parse_args()
  time_forward_dict_list, layer_name_list, layer_forward_time_list = parse_log(args.logfile_path)
  
  print_net_time(time_forward_dict_list, layer_forward_time_list, args.base_net_name)
  
  plot_log_SSD(layer_name_list, layer_forward_time_list, args.base_net_name)

  
if __name__ == '__main__':
  main()
