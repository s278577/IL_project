import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

parameters = {
    'NUM_CLASSES': 100,
    'BATCH_SIZE' : 128,
    'CLASSES_BATCH' : 10,
    'STEPDOWN_FACTOR' : 5,
    'LR' : 2,
    'MOMENTUM' : 0.9,
    'WEIGHT_DECAY' : 1e-5,
    'NUM_EPOCHS' :70,
    'MILESTONES' : [49,63],
    'GAMMA':0.1
}

def plot_ft(new_acc_train, new_acc_test, new_loss_train, new_loss_test, all_acc, args):
    num_epochs = len(new_acc_train[0])
    x = np.linspace(1, num_epochs, num_epochs)

    for i, (acc_train, acc_test, loss_train, loss_test) in enumerate(zip(new_acc_train, new_acc_test, new_loss_train, new_loss_test)):

        title = 'Accuracy dataset # %d - BATCH_SIZE= %d LR= %f  EPOCHS= %d  GAMMA= %f' \
                % (i + 1, args['BATCH_SIZE'], args['LR'], args['NUM_EPOCHS'], args['GAMMA'])
        title2 = 'Loss dataset # %d - BATCH_SIZE= %d LR= %f  EPOCHS= %d  GAMMA= %f' \
                 % (i + 1, args['BATCH_SIZE'], args['LR'], args['NUM_EPOCHS'], args['GAMMA'])

        plt.plot(x, acc_train, color='green')
        plt.plot(x, acc_test, color='darkorange')
        plt.title(title)
        plt.xticks(np.arange(1, num_epochs, 4))
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend(['Train accuracy', 'Test accuracy'], loc='best')
        plt.show()

        plt.plot(x, loss_train, color='green')
        plt.plot(x, loss_test, color='darkorange')
        plt.title(title2)
        plt.xticks(np.arange(1, num_epochs, 4))
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(['Train loss', 'Test loss'], loc='best')
        plt.show()
    x=[1,2,3,4,5,6,7,8,9,10]
    plt.plot(x, np.array(all_acc)[:,0], color='lightseagreen')
    plt.title('%s incremental learning accuracy' % (args['name']))
    plt.xticks(np.arange(1, len(all_acc), 1))
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')
    plt.legend(['Test accuracy'], loc='best')
    plt.show()

    csv_name = '{}_batch-size-{}_lr-{}_epochs-{}_gamma-{}' .format(args['name'], args['BATCH_SIZE'], args['LR'], args['NUM_EPOCHS'], args['GAMMA']) 
    pd.DataFrame(all_acc).to_csv('./Results/{}.csv' .format (csv_name))

    print('Accuracy last test', new_acc_test[-1])
    
def plot_loss(new_loss_train,new_loss_test,num_epochs,args):
    for i,(loss_train, loss_test) in enumerate(zip(new_loss_train,new_loss_test)):
        title2 = 'Loss dataset # %d - BATCH_SIZE= %d LR= %f  EPOCHS= %d  GAMMA= %f' \
                 % (i + 1, args['BATCH_SIZE'], args['LR'], args['NUM_EPOCHS'], args['GAMMA'])
        plt.plot(x, loss_train, color='green')
        plt.plot(x, loss_test, color='darkorange')
        plt.title(title2)
        plt.xticks(np.arange(1, num_epochs, 4))
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(['Train loss', 'Test loss'], loc='best')
        plt.show()
    #csv_name = '{}_batch-size-{}_lr-{}_epochs-{}_gamma-{}' .format(args['name'], args['BATCH_SIZE'], args['LR'], args['NUM_EPOCHS'], args['GAMMA']) 
    #pd.DataFrame(all_acc).to_csv('./Results/{}.csv' .format (csv_name))

def plot_loss(loss_list,num_epochs,args):
    x = np.linspace(1, num_epochs, num_epochs)
    title2 = 'Loss dataset # %d - BATCH_SIZE= %d LR= %f  EPOCHS= %d  GAMMA= %f' \
                 % (i + 1, args['BATCH_SIZE'], args['LR'], args['NUM_EPOCHS'], args['GAMMA'])
    for i,loss in enumerate(loss_list):
        plt.plot(x, loss, color='green')
        plt.title(title)
        plt.xticks(np.arange(1, num_epochs, 4))
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.show()