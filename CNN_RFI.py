# pylint: disable=maybe-no-member
# from plot_model import plot_results
import torch
import torchvision
import matplotlib.pyplot as plt
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import time
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from astropy.io import fits
import numpy as np
import glob
import torch.utils.data as Data
from sklearn.model_selection import train_test_split
from glob import glob
from scipy.optimize import curve_fit, least_squares
from scipy.signal import medfilt
import scipy.constants as C
import warnings
import math
import itertools
import logging
import time 
import os 
import csv
from logging import handlers

class Logger(object):
    level_relations = {
        'debug':logging.DEBUG,
        'info':logging.INFO,
        'warning':logging.WARNING,
        'error':logging.ERROR,
        'crit':logging.CRITICAL
    }

    def __init__(self,filename,level='info'):
        self.logger = logging.getLogger(filename)
        format_str = logging.Formatter()
        self.logger.setLevel(self.level_relations.get(level))
        sh.setFormatter(format_str) 
        th = logging.FileHandler(filename=filename, mode='w', encoding='utf-8')
        self.logger.addHandler(sh) 
        self.logger.addHandler(th)
        

def save_module(cnn_model, epoch):
    if not os.path.exists('./'+os.path.basename(__file__).replace('.py', '')):
        os.makedirs('./'+os.path.basename(__file__).replace('.py', ''))
    torch.save(cnn_model, './'+os.path.basename(__file__).replace('.py', '/') + str(epoch) + '_cnn_model.pt')

def data_write_csv(file_name, datas, mode): 
    writer = csv.writer(file_csv)
    writer.writerow(datas)

def data_read_csv(file_name):
    data = open('./' + file_name)
    data = csv.reader(data)
    allend = []
    for row in data:
        allend.append(row)
    allend = np.array(allend)
    return allend

def get_variable(x):
    x = Variable(x)
    return x

def onehot(classes):
    ''' Encodes a list of descriptive labels as one hot vectors '''
    label_encoder = LabelEncoder()
    int_encoded = label_encoder.fit_transform(classes)
    labels = label_encoder.inverse_transform(np.arange(np.amax(int_encoded) + 1))
    onehot_encoder = OneHotEncoder(sparse=False)
    int_encoded = int_encoded.reshape(len(int_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(int_encoded)
    return onehot_encoded

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        m.weight.data.uniform_(0.0, 1.0)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        m.weight.data.uniform_(0.0, 1.0)
        m.bias.data.fill_(0)

def confusion_matrix(preds, labels, conf_matrix):
    
    for p, t in zip(preds, labels):
        conf_matrix[p, t] += 1
    return conf_matrix
def read_data():
    print("Reading in training data...")
    flux = []
    scls = []
    sdir_data = '/mnt/storage-ssd/sunhaomin/work/test_data/test_100_119/no_rfi/*.npy'
    for imageFile in glob(sdir_data):
        im = np.load(imageFile)
        flux.append(im)
        scls.append(0)
    add_flux_data = '/mnt/storage-ssd/sunhaomin/work/test_data/test_100_119/rfi/*.npy'
    for imageFile in glob(add_flux_data):
        add_im = np.load(imageFile)
        flux.append(add_im)
        scls.append(1)

    flux = np.array(flux)
    scls = onehot(scls)
    scls = np.array(scls)
    flux = flux.reshape(-1, 1, 1, 2)
    print('scls_train_shape:', scls.shape)
    # Reading in testing data:
    print("Reading in testing data...")
    flux_test = []
    scls_test = []
    data_test_path = '/mnt/storage-data/sunhaomin/other/lofar_data/'
    sdir_data_test = os.listdir(data_test_path)
    sdir_data_test.sort(key=lambda x: int(x.split('.')[0]))

    for imageFile in sdir_data_test:
        im = np.load(os.path.join(data_test_path, imageFile))
        flux_test.append(im)

    flux_test = np.array(flux_test)
    flux_test = flux_test.reshape(-1, 1, 1, 2)
    print('flux_test_shape:', flux_test.shape)
    sdir_label_test = '/mnt/storage-ssd/sunhaomin/lofar_flag.npy'
    for imageFile in glob(sdir_label_test):
        lb = np.load(imageFile)
        scls_test = np.array(lb)

    scls_test = np.array(scls_test)
    print('test_data_rfi_sum:', np.sum(scls_test))
    scls_test = scls_test.reshape(-1)
    scls_test = onehot(scls_test)
    print('scls_test:', scls_test.shape)
    print(np.sum(scls_test))


    flux, fluxTE, scls, sclsTE = train_test_split(flux, scls, test_size=0.01)
    Xtrain1 = torch.from_numpy(flux)  
    Xtest1 = torch.from_numpy(flux_test)
    ytrain1 = torch.from_numpy(scls)
    ytest1 = torch.from_numpy(scls_test)
    torch_dataset_train = Data.TensorDataset(Xtrain1, ytrain1)
    torch_dataset_test = Data.TensorDataset(Xtest1, ytest1)
    print(scls.shape[0],scls_test[0])
    data_loader_train = torch.utils.data.DataLoader(dataset=torch_dataset_train, batch_size=batch_size, shuffle=True)
    data_loader_test = torch.utils.data.DataLoader(dataset=torch_dataset_test, batch_size=scls_test.shape[0], shuffle=False)
    return data_loader_train, data_loader_test, scls.shape[0], scls_test.shape[0]
def plot_confusion_matrix(cm, classes, title, normalize=False, cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, n0.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)


    plt.axis("equal")
  
    ax = plt.gca()  
    left, right = plt.xlim()  
    ax.spines['left'].set_position(('data', left))
    ax.spines['right'].set_position(('data', right))
    for edge_i in ['top', 'bottom', 'right', 'left']:
        ax.spines[edge_i].set_edgecolor("white")
   

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        num = '{:.2f}'.format(cm[i, j]) if normalize else int(cm[i, j])
        plt.text(j, i, num,
                 verticalalignment='center',
                 horizontalalignment="center",
                 color="white" if num > thresh else "black")
    plt.tight_layout()
    plt.ylabel('Predicted label')
    plt.xlabel('True label')
    plt.show()
    plt.close()

class CNN_Model(torch.nn.Module):
    def __init__(self):
        super(CNN_Model, self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 64, kernel_size=(1, 2), padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(stride=1, kernel_size=(1, 1)))
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 64, kernel_size=(1, 2), stride=1, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(stride=1, kernel_size=(1,2)))
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(stride=1, kernel_size=(1,2)))
        self.conv4 = torch.nn.Sequential(
            torch.nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(stride=1,kernel_size=2))
        self.conv5 = torch.nn.Sequential(
            torch.nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(1024),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(stride=2,kernel_size=3))
        self.conv6 = torch.nn.Sequential(
            torch.nn.Conv2d(1024, 1024, kernel_size=1, stride=1, padding=1),
            torch.nn.BatchNorm2d(1024),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(stride=2,kernel_size=3))
        self.dense = torch.nn.Sequential(
            torch.nn.Linear(128 * 4 * 1, 1024),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0),
            torch.nn.Linear(1024, num_class),
            torch.nn.Softmax())
  
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(-1, 128 * 4 * 1)
        x = self.dense(x)
        return x

def run_CNN_module(num_class, num_epochs, batch_size, learning_rate, train, test, clsTR, clsTE):
    
    cnn_model = CNN_Model()
    loss_func = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(cnn_model.parameters(), lr=learning_rate, momentum=0.9)
    ctest = []
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        running_correct = 0.0
        log.logger.info("Epoch  {}/{}".format(epoch, num_epochs))


        for data in train:
            optimizer.zero_grad()
            X_train, y_train = data
            X_train, y_train = get_variable(X_train), get_variable(y_train)
            optimizer.zero_grad()
            X_train = torch.tensor(X_train, dtype=torch.float32)
            outputs = cnn_model(X_train)
            _, pred = torch.max(outputs.data, 1)
            
            y_train = torch.max(y_train, 1)[1]


            loss = loss_func(outputs, y_train)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            running_correct += torch.sum(pred == y_train.data)
        
        
        testing_correct = 0.0
        class_correct = list(0. for i in range(num_class))
        class_total = list(0. for i in range(num_class))

        for data in test:
            X_test, y_test = data
            X_test, y_test = get_variable(X_test), get_variable(y_test)
            X_test = torch.tensor(X_test, dtype=torch.float32)
            outputs = cnn_model(X_test)
            _, pred = torch.max(outputs.data, 1) 
            y_test = torch.max(y_test, 1)[1]
            a = np.array(y_test)
            b = np.array(pred)
            c = np.array(_)
         
            data_write_csv(os.path.basename(__file__).replace('.py','.csv'), a, 'a')
            data_write_csv(os.path.basename(__file__).replace('.py','.csv'), b, 'a')

            conf_matrix = torch.zeros(2, 2)
            conf_matrix = confusion_matrix(pred, y_test, conf_matrix)
            log.logger.info(conf_matrix)
            s = a - b
            p = np.mean(s)
            q = np.std(s)
            log.logger.info('means:%.3f' % (p))
            log.logger.info('std:%.3f' % (q))
            c = (pred == y_test).squeeze()
            testing_correct += torch.sum(pred == y_test.data)
            classes = ['0', '1']

            for i in range(len(y_test)):
                label = y_test[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
        print(time.time()-bt)
        all_pre = testing_correct / clsTE
        ctest.append(all_pre)

        all_recall = testing_correct / clsTE
        all_f1 = 2 * all_pre * all_recall / (all_pre + all_recall)
        all_recall = np.array(all_recall)
        all_f1 = np.array(all_f1)

        log.logger.info('all_recall:%.3f' % (all_recall))
        log.logger.info('all_f1:%.3f' % (all_f1))
        log.logger.info("Loss is :{:.4f},Train Accuracy is:{:.4f}%,Test Accuracy is:{:.4f}%".format(
            running_loss / clsTR, 100 * running_correct / clsTR, 100 * all_pre))

        save_module(cnn_model, epoch)
    ctest = np.array(ctest)
    premax = np.max(ctest)
    premaxdex = np.argmax(ctest)
    log.logger.info('The highest value of pred is :%.5f,which index is :%d' % (premax, premaxdex))

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    begin_time = time.time()
    num_class = 2
    num_epochs = 100
    batch_value = 0
    batch_size = 16
    learning_rate = 0.001
    log = Logger(os.path.basename(__file__).replace('.py','.log'), level='info')
    file_csv = open(os.path.basename(__file__).replace('.py','.csv'), mode='w')


    train, test, clsTR, clsTE = read_data()
    run_CNN_module(num_class, num_epochs, batch_size, learning_rate, train, test, clsTR, clsTE)

    # cnn_model=torch.load('data/cnn_model.pt')
    # cnn_model.eval()
    end_time=time.time()
    print(begin_time-end_time)
   # print(cnn_model)
