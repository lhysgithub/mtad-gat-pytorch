import json
from datetime import datetime
import torch.nn as nn
import random

from args import get_parser
from utils import *
from mtad_gat import MTAD_GAT
from prediction import Predictor
from training import Trainer

def filter_input(sellected,x_train):
    temp_x_train = []
    for i in range(len(sellected)):
        xi = x_train[:,sellected[i]].reshape(-1, 1)
        if i == 0:
            temp_x_train = xi
        else:
            temp_x_train = np.concatenate((temp_x_train, xi), axis=1)
    return temp_x_train


def filter_input_by_bool(selected,x_train):
    temp_x_train = []
    first = 0
    for i in range(len(selected)):
        xi = x_train[:,i].reshape(-1, 1)
        if selected[i] == True:
            if first == 0:
                temp_x_train = xi
                first = 1
            else:
                temp_x_train = np.concatenate((temp_x_train, xi), axis=1)
    return temp_x_train


def list2bin(l,max_dim):
    bin = []
    for i in range(max_dim):
        if i in l:
            bin.append("1")
        else:
            bin.append("0")
    return "".join(bin)


def bool_list2bin(l):
    bin = []
    for i in l:
        if i:
            bin.append("1")
        else:
            bin.append("0")
    return "".join(bin)


def bin2list(bin):
    l = []
    for i in range(len(bin)):
        if bin[i] == "1":
            l.append(i)
    return l


def get_f1(file_name):
    with open(file_name) as f:
        summary = json.load(f)
        f1 = summary["bf_result"]["f1"]
    return f1


def split_val_set(x_test,y_test,val_ratio=0.05):
    dataset_len = int(len(x_test))
    val_use_len = int(dataset_len * val_ratio)
    index_list = []
    lens_list = []
    find = 0
    count = 0
    for i in range(len(y_test)):
        if int(y_test[i]) == 1:
            index_list.append(i)
            find = 1
            count += 1
        elif find == 1:
            find = 0
            lens_list.append(count)
            count = 0
    index = random.choice(index_list)
    # i = np.argmax(lens_list)
    # index = index_list[i]
    start = 0
    end = 0
    if index < val_use_len/2:
        start = 0
        end = val_use_len
    elif dataset_len - index < val_use_len/2:
        start = dataset_len - val_use_len
        end = dataset_len
    else:
        start = index - val_use_len/2
        end = index + val_use_len/2
    start = int(start)
    end = int(end)
    x_val = x_test[start:end]
    y_val = y_test[start:end]
    new_x_test = np.concatenate((x_test[:start],x_test[end:]))
    new_y_test = np.concatenate((y_test[:start],y_test[end:]))
    return x_val,y_val,new_x_test,new_y_test


def main(feature_numbers,sellected,dataset,group_index,index):

    id = datetime.now().strftime("%d%m%Y_%H%M%S")

    parser = get_parser()
    args = parser.parse_args()

    # dataset = args.dataset
    window_size = args.lookback
    spec_res = args.spec_res
    normalize = args.normalize
    n_epochs = args.epochs
    batch_size = args.bs
    init_lr = args.init_lr
    val_split = args.val_split
    shuffle_dataset = args.shuffle_dataset
    use_cuda = args.use_cuda
    print_every = args.print_every
    log_tensorboard = args.log_tensorboard
    # group_index = args.group[0]
    # index = args.group[2:]
    args.sellected = sellected
    args_summary = str(args.__dict__)
    print(args_summary)

    if dataset == 'SMD':
        output_path = f'output/SMD/{args.group}'
        (x_train, _), (x_test, y_test) = get_data(f"machine-{group_index}-{index}", normalize=normalize)
    elif dataset in ['MSL', 'SMAP']:
        output_path = f'output/{dataset}'
        (x_train, _), (x_test, y_test) = get_data(dataset, normalize=normalize)
    else:
        raise Exception(f'Dataset "{dataset}" not available.')

    log_dir = f'{output_path}/logs'
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    # save_path = f"{output_path}/20221117_smd"
    # save_path = f"{output_path}/{id}"

    # todo add filter
    bin_str = list2bin(sellected,feature_numbers)
    save_path = f"{output_path}/search/20221117_smd_{bin_str}"
    temp_x_train = filter_input(sellected,x_train)
    temp_x_test = filter_input(sellected,x_test)
    x_train = torch.from_numpy(temp_x_train).float()
    x_test = torch.from_numpy(temp_x_test).float()
    # todo clip verification set
    # x_val,y_val,x_test,y_test = split_val_set(x_test,y_test)
    # old
    # x_train = torch.from_numpy(x_train).float()
    # x_test = torch.from_numpy(x_test).float()
    n_features = x_train.shape[1]

    target_dims = get_target_dims(dataset)
    if target_dims is None:
        out_dim = n_features
        print(f"Will forecast and reconstruct all {n_features} input features")
    elif type(target_dims) == int:
        print(f"Will forecast and reconstruct input feature: {target_dims}")
        out_dim = 1
    else:
        print(f"Will forecast and reconstruct input features: {target_dims}")
        out_dim = len(target_dims)

    train_dataset = SlidingWindowDataset(x_train, window_size, target_dims)
    test_dataset = SlidingWindowDataset(x_test, window_size, target_dims)

    train_loader, val_loader, test_loader = create_data_loaders(
        train_dataset, batch_size, val_split, shuffle_dataset, test_dataset=test_dataset
    )

    model = MTAD_GAT(
        n_features,
        window_size,
        out_dim,
        kernel_size=args.kernel_size,
        use_gatv2=args.use_gatv2,
        feat_gat_embed_dim=args.feat_gat_embed_dim,
        time_gat_embed_dim=args.time_gat_embed_dim,
        gru_n_layers=args.gru_n_layers,
        gru_hid_dim=args.gru_hid_dim,
        forecast_n_layers=args.fc_n_layers,
        forecast_hid_dim=args.fc_hid_dim,
        recon_n_layers=args.recon_n_layers,
        recon_hid_dim=args.recon_hid_dim,
        dropout=args.dropout,
        alpha=args.alpha
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=args.init_lr)
    forecast_criterion = nn.MSELoss()
    recon_criterion = nn.MSELoss()

    trainer = Trainer(
        model,
        optimizer,
        window_size,
        n_features,
        target_dims,
        n_epochs,
        batch_size,
        init_lr,
        forecast_criterion,
        recon_criterion,
        use_cuda,
        save_path,
        log_dir,
        print_every,
        log_tensorboard,
        args_summary
    )

    trainer.fit(train_loader, val_loader)

    plot_losses(trainer.losses, save_path=save_path, plot=False)

    # Check test loss
    test_loss = trainer.evaluate(test_loader)
    print(f"Test forecast loss: {test_loss[0]:.5f}")
    print(f"Test reconstruction loss: {test_loss[1]:.5f}")
    print(f"Test total loss: {test_loss[2]:.5f}")

    # Some suggestions for POT args
    level_q_dict = {
        "SMAP": (0.90, 0.005),
        "MSL": (0.90, 0.001),
        "SMD-1": (0.9950, 0.001),
        "SMD-2": (0.9925, 0.001),
        "SMD-3": (0.9999, 0.001)
    }
    key = "SMD-" + args.group[0] if args.dataset == "SMD" else args.dataset
    level, q = level_q_dict[key]
    if args.level is not None:
        level = args.level
    if args.q is not None:
        q = args.q

    # Some suggestions for Epsilon args
    reg_level_dict = {"SMAP": 0, "MSL": 0, "SMD-1": 1, "SMD-2": 1, "SMD-3": 1}
    key = "SMD-" + args.group[0] if dataset == "SMD" else dataset
    reg_level = reg_level_dict[key]

    trainer.load(f"{save_path}/model.pt")
    prediction_args = {
        'dataset': dataset,
        "target_dims": target_dims,
        'scale_scores': args.scale_scores,
        "level": level,
        "q": q,
        'dynamic_pot': args.dynamic_pot,
        "use_mov_av": args.use_mov_av,
        "gamma": args.gamma,
        "reg_level": reg_level,
        "save_path": save_path,
    }
    best_model = trainer.model
    predictor = Predictor(
        best_model,
        window_size,
        n_features,
        prediction_args,
    )

    label = y_test[window_size:] if y_test is not None else None
    predictor.predict_anomalies(x_train, x_test, label)

    # Save config
    args_path = f"{save_path}/config.txt"
    with open(args_path, "w") as f:
        json.dump(args.__dict__, f, indent=2)

    file_name = save_path+"/"+"summary.txt"
    return get_f1(file_name)


if __name__ == '__main__':
    feature_numbers = 38
    # selected = range(feature_numbers)  # todo add ga
    # main(feature_numbers,selected)
    temp = []
    for j in range(feature_numbers):
        best_f1 = 0
        best_id = -1
        for i in range(feature_numbers):
            if i in temp:
                continue
            selected = [i] + temp
            f1 = main(feature_numbers,selected,"SMD","1","4")
            if best_f1 < f1:
                best_f1 = f1
                best_id = i
        temp = temp + [best_id]