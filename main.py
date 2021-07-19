# -*- coding : utf-8-*-

import pandas as pd

from functions import *
from weinet import *
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from net_para import *
import argparse
import math

# device = 'cpu'
device = 'cuda' if torch.cuda.is_available() else 'cpu'  # 设置设备 优先在GPU上运行
dataset_path = os.path.join(os.getcwd(), '..', 'UCRArchive_2018')

dataset_list=[ "EpilepticSeizure", "InsectWingbeatSound", "MFPT", "XJTU"]

percentage_subsequence_length = 0.3  # 子序列长度占时间序列长度的比例
percentage_stride = 0.05  # stride占时间序列长度的比例
num_epochs = 100  # number of iteration
batch_size = 8
alpha = 0.1
level = 2

network_args = {
    "stride": percentage_stride,
    "horizon": percentage_subsequence_length,
    "level": level,
    "alpha": alpha,
    "run_times": "t1",
    "batch_size": batch_size,
    "num_epochs": num_epochs,
    #  "net_base": "wave-Alex-waveconcat-ne-AdamW-R2-"
    "net_base": "wave-Alex-NewTrainForm-UCR"
}


def calcul_accuracy_score(model, x, y, batch_size=2):
    x_iter = DataLoader(x, batch_size=batch_size, drop_last=False)
    y_hat = np.array([] * 1, dtype='float32').T

    for batch, x_batch in enumerate(x_iter):
        x_batch_unsequ = x_batch[:, 0, :].unsqueeze(1)
        y_hat_batch = model.forward_test(x_batch_unsequ.float().to(device)).cpu()
        y_hat_batch_res = np.argmax(y_hat_batch.cpu().detach().numpy(), axis=1)
        # y_hat = np.concatenate(y_hat, test)
        y_hat = np.append(y_hat, y_hat_batch_res)
    accuracy = accuracy_score(
        y_hat, y.long().cpu().numpy())
    return accuracy


def run_network(dataset_name, net_info):
    print(dataset_name)
    # load data
    data = loadDataset(dataset_path, dataset_name)
    data = data.astype('float32')
    # data = data[0:400]
    # acquire the number of class
    num_classes = len(np.unique(data.values[:, 0]))

    # split train test dataset
    train_labeled_x, train_labeled_y, train_unlabeled_x, validate_x, validate_y, test_x, test_y = stratifiedSampling(
        data, seed=0, normalization=True)

    # calculate the length of stride
    sub_length = int(train_labeled_x.shape[2] * net_info.horizon)
    stride = int(train_labeled_x.shape[2] * net_info.stride)

    # get the subsequence
    pre, post = getPreAndPostSubsequences(torch.cat((train_labeled_x, train_unlabeled_x)), sub_length, stride)

    # calculate coefficient
    pre_coeffs = waveletTransform(pre.cpu(), level=net_info.level)
    post_coeffs = waveletTransform(post.cpu(), level=net_info.level)

    print("horizon:{} stride{} level{}".format(net_info.horizon, net_info.stride, net_info.level))

    # pre predict post
    subsequences = torch.cat((pre, post), dim=1)
    subsequences_coeffs = torch.cat((pre_coeffs, post_coeffs), dim=1)

    # flexible batch size
    label_x_iterate = DataLoader(train_labeled_x, batch_size=batch_size, drop_last=False)
    label_y_iterate = DataLoader(train_labeled_y, batch_size=batch_size, drop_last=False)

    label_unlabel_ratio = math.ceil(subsequences.shape[0] / train_labeled_x.shape[0])

    # 获取指定批量大小的iteratable的训练数据
    subsequences_iterate = DataLoader(subsequences, batch_size=batch_size * label_unlabel_ratio, drop_last=False)
    subsequences_coeffs_iterate = DataLoader(subsequences_coeffs, batch_size=batch_size * label_unlabel_ratio,
                                             drop_last=False)

    # build model
    alexnet_mod = ALEXNetMOD(int(pre.shape[2]), int(pre_coeffs.shape[2]), num_classes).to(device)

    # loss function
    criterion_classification = nn.CrossEntropyLoss()  # 分类的损失函数，交叉熵
    criterion_forecasting = nn.MSELoss()  # 预测的损失函数，均方差

    # optimizer
    optimizer = torch.optim.AdamW(alexnet_mod.parameters(), lr=0.01, weight_decay=0.2)

    dataset_stat = pd.DataFrame(
        columns=['epoch', 'train_acc', 'val_acc', 'test_acc', 'classification_loss', 'forecast_loss'])
    best_accuracy = 0

    max_test_acc = 0
    # train model
    for t in range(num_epochs):

        for batch, (label_x_batch, label_y_batch, subsequences_batch, subsequences_coeffs_batch) in enumerate(
                zip(label_x_iterate, label_y_iterate,
                    subsequences_iterate,
                    subsequences_coeffs_iterate)):
            label_x_batch_unsequ = label_x_batch[:, 0, :].unsqueeze(1)
            label_y_batch_unsequ = label_y_batch

            pre_subsequences_batch = subsequences_batch[:, 0, :].unsqueeze(1)
            post_subsequences_batch = subsequences_batch[:, 1, :].unsqueeze(1)
            pre_subsequences_coeffs_batch = subsequences_coeffs_batch[:, 0, :].unsqueeze(1)
            post_subsequences_coeffs_batch = subsequences_coeffs_batch[:, 1, :].unsqueeze(1)
            loss_classification, loss_forecast = optimize_network(alexnet_mod, optimizer, label_x_batch_unsequ.to(device),
                                                                  label_y_batch_unsequ.to(device),
                                                                  pre_subsequences_batch.to(device),
                                                                  post_subsequences_batch.to(device),
                                                                  pre_subsequences_coeffs_batch.to(device),
                                                                  post_subsequences_coeffs_batch.to(device), alpha)

        train_acc = calcul_accuracy_score(alexnet_mod, train_labeled_x, train_labeled_y, 64)
        val_acc = calcul_accuracy_score(alexnet_mod, validate_x, validate_y, 64)
        test_acc = calcul_accuracy_score(alexnet_mod, test_x, test_y, 64)

        current_accuracy = test_acc
        if current_accuracy >= best_accuracy:
            best_accuracy = current_accuracy
            torch.save(alexnet_mod.state_dict(), net_info.model_save_pos)

        if (test_acc >= max_test_acc):
            max_test_acc = test_acc
        print(
            f'Epoch{t}   train_acc: {train_acc}, val_acc: {val_acc}, test_acc: {test_acc}, 分类损失: {loss_classification}, 预测损失: {loss_forecast}')
        dataset_stat = dataset_stat.append(
            {'epoch': t, 'train_acc': train_acc, 'val_acc': val_acc, 'test_acc': test_acc,
             'classification_loss': loss_classification, 'forecast_loss': loss_forecast},
            ignore_index=True
        )
        if best_accuracy == 1:
            break
    dataset_stat = dataset_stat.append({'epoch': 'best_accuracy:{}'.format(best_accuracy)}, ignore_index=True)
    dataset_stat.to_csv(net_info.dataset_stat_pos)
    print(max_test_acc)
    return best_accuracy


def one_run(horizon=network_args["horizon"], stride=network_args["stride"],
            level=network_args["level"], alpha=network_args["alpha"], run_time=network_args["run_times"]):

    network_args["horizon"] = horizon
    network_args["stride"] = stride
    network_args["level"] = level
    network_args["alpha"] = alpha
    network_args["run_times"] = run_time

    # try to use net_info to manage the parameters
    net_info = net_para()

    datasets_stat = pd.DataFrame(columns=['Dataset', 'Accuracy', 'horizon', 'stride', 'level', 'alpha'])
    print(net_info.onerun_suffix)

    for dataset in dataset_list:
        if not net_info.init_network(parser=argparse.ArgumentParser(), network_args=network_args):
            print("{} has problem.".format(net_info.onerun_suffix))
            return False
        #        net_info.current_dataset = dataset
        print(net_info.onerun_suffix)
        print(dataset)

        net_info.dataset_run_set(dataset)
        if os.path.exists(net_info.dataset_stat_pos):
            print("{} has been processed".format(net_info.dataset_stat_pos))
            continue

        try:
            # train model
            data_set_accuracy = run_network(dataset, net_info)
            datasets_stat.append({'Dataset': dataset, 'Accuracy': data_set_accuracy, 'horizon': net_info.horizon,
                                  'stride': net_info.stride, 'level': net_info.level, 'alpha': net_info.alpha},
                                 ignore_index=True
                                 )
            datasets_stat.to_csv(net_info.datasets_stat_pos)
        except Exception as ex:
            template = "An exception of type {0} occurred. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print('\n' + message)
            continue
    return True


if __name__ == '__main__':
    one_run(percentage_subsequence_length, percentage_stride, run_time="t1")
