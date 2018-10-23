import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os

def plot_log(logfile_path):
#  print(logfile_path+".train")
  train_log = pd.read_csv(logfile_path+".train")
  test_log = pd.read_csv(logfile_path+".test")
  logfile_name = os.path.split(logfile_path)[-1]
  
  fig=plt.figure(figsize=(20, 12)) 

  ax1 = fig.add_subplot(211)
  ax1.set_title("{}\ntrain loss and test mAP".format(logfile_name))
  ax1.plot(train_log["NumIters"], train_log["mbox_loss"], alpha=0.5)
  #ax1.plot(train_log["NumIters"], train_log["LearningRate"], 'g')
  ax1.set_xlabel('iteration')
  ax1.set_ylabel('train loss')
  plt.grid(True)
  plt.legend(loc=2)

  ax2 = ax1.twinx()
  ax2.plot(test_log["NumIters"], test_log["detection_eval"], 'r')
  ax2.set_ylabel('test mAP')
  plt.legend(loc=0)

  ax3 = fig.add_subplot(212)
  ax3.set_title('train learn rate')
  ax3.plot(train_log["NumIters"], train_log["LearningRate"], 'g')
  ax3.set_xlabel('iteration')
  ax3.set_ylabel('learningRate')
  plt.grid(True)
  plt.legend(loc=0)

  picture_path = logfile_path.replace('.log','.png')
  print('saving file: {}'.format(picture_path))
  plt.savefig(picture_path)
  
  plt.show()
  
  

def parse_args():
  description='Plot caffe log.'

  parser = argparse.ArgumentParser(description)

  parser.add_argument('logfile_path',
                      help='Path to log file',
                      default = ' ')
  
  #logfile_pathdir = os.split()
  parser.add_argument('--picture_dir',
                      help = 'where to save caffe log plot picture')

  args = parser.parse_args()
  return args


def main():
    args = parse_args()
    print('plotting...')
    plot_log(args.logfile_path)


if __name__ == '__main__':
    main()
