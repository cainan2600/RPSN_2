import matplotlib.pyplot as plt
import matplotlib
import os

matplotlib.use('Agg')

def plot_IK_solution(checkpoint_dir, start_epoch, epochs, num_train, num_incorrect_test, num_correct_test):

    draw_epochs = list(range(start_epoch, start_epoch + epochs))

    plt.figure()
    plt.plot(draw_epochs, num_incorrect_test, 'r-', label='Incorrect-No solutions')
    plt.plot(draw_epochs, num_correct_test, 'b-', label='Correct-IK have solutions')

    plt.annotate('{} data sets'.format(num_train), xy=(0.4, 0.5), xycoords='axes fraction', fontsize=12,
                color='gray', horizontalalignment='center', verticalalignment='center')
    # if epoch == 400:
    #     plt.annotate(str(num_correct_test[399]), xy=(draw_epochs[399], num_correct_test[399]),
    #                 xytext=(draw_epochs[399] - 0.1, num_correct_test[399] + 0.8),
    #                 fontsize=8)

    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.title('Testing Process ')
    plt.legend()

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    file_path = os.path.join(checkpoint_dir, 'Testing Process.png')
    plt.savefig(file_path)

    # plt.show()

def plot_train(checkpoint_dir, start_epoch, epochs, num_train, numError1, numError2, numNOError1, numNOError2):
    draw_epochs = list(range(start_epoch, start_epoch + epochs))
    plt.figure()

    plt.plot(draw_epochs, numError1, 'r-', label='illroot')
    plt.plot(draw_epochs, numError2, 'g-', label='outdom')
    plt.plot(draw_epochs, numNOError1, 'b-', label='illsolu')
    plt.plot(draw_epochs, numNOError2, 'b-', linewidth=3, label='idesolu')

    plt.annotate('{} data sets'.format(num_train), xy=(0.4, 0.5), xycoords='axes fraction', fontsize=12,
                 color='gray', horizontalalignment='center', verticalalignment='center')
    # if epoch == 400:
    #     plt.annotate(str(numNOError2[399]), xy=(draw_epochs[399], numNOError2[399]),
    #                  xytext=(draw_epochs[399] - 0.1, numNOError2[399] + 0.8),
    #                  fontsize=8)

    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.title('Training Process')
    plt.legend()

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    file_path = os.path.join(checkpoint_dir, 'Training Process.png')
    plt.savefig(file_path)

    # plt.show()