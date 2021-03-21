
import matplotlib.pyplot as plt
# %matplotlib inline

COLOR_LIST = ['r', 'g', 'b', 'k', 'p']


def plot_loss(history, key_list=None, need_save=False, save_path=None):
    if not key_list:
        print('no data need to plot')
        return

    no_data_key = []
    for key in key_list:
        if key not in history or not history.get(key):
            no_data_key.append(key)
    if not no_data_key:
        print("key: {} not in history, will filter".format(no_data_key))
    new_key_list = list(filter(lambda x: x not in no_data_key, key_list))
    if not new_key_list:
        print("no index need to plot")
        return
    x_index = range(len(history.get(new_key_list[0])))
    plt.figure()

    for idx, key in enumerate(new_key_list):
        plt.plot(x_index, history.get(key), COLOR_LIST[idx % len(COLOR_LIST)], label=key)
    plt.grid(True)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(loc="upper right")

    if need_save and save_path:
        plt.savefig(save_path)
    else:
        plt.show()
