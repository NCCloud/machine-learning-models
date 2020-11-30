# ML Subscriptions NN
# Trains Server Implementation

from trains import Task, Logger
from trains.storage.manager import StorageManager
from trains.storage.util import get_config_object_matcher
from trains.backend_config.config import Config
from trains.backend_config.defs import LOCAL_CONFIG_FILES
from trains.backend_api import Session


from nn.subs import SubsDS
from nn.nets import SubsNN, SubsNN2, FFNN
from nn.fernet import decrypt, FERNET_KEY
from tempfile import gettempdir

import torch
import torch.nn as nn
import argparse
import sys
import os
from minio import Minio
from minio.error import (ResponseError, AccessDenied)

# Sealed keys
KEY_ID = "gAAAAABfwRbhbvqaFGHhV91TfPPUuUgNbpOCJOk5owAUC1jc-Kljb1nLQJxyVwfyuOETRi2Ge6ZCefY6aLfRyjALF4ZcKlZapqbDzxRaiRj4ICVGRzCMDK0="
KEY_SECRET = "gAAAAABfwRbhvaJ1wG-Jw4dMu9xFTdWH4Wi_wXvjxWUIQ5M6yaB--ca_GY9-o7EO8em2wddDM-weaafcSB4zURrd0ohUAxnASOofXMNrmW2wfNTD__9mfJUzLr7cGyn7XLw7gEBVNzbC"


def get_minio_client():
    k_id = decrypt(KEY_ID.encode(), FERNET_KEY).decode()
    k_secret = decrypt(KEY_SECRET.encode(), FERNET_KEY).decode()

    return Minio(
        "s3.namecheapcloud.net",
        access_key=os.environ.get("AWS_ACCESS_KEY_ID", k_id),
        secret_key=os.environ.get("AWS_ACCESS_KEY_SECRET", k_secret),
        secure=True
    )


def get_file(client: Minio, key: str, dest_folder: str) -> bool:
    """Locally download key from s3
    This method aims to fix errors configuring the trains StorageManager
    Args:
        client (Minio): Minio client
        key (str): path to file in bucket
        dest_folder (str): folder path without filename

    Returns:
        bool: If the file was downloaded
    """
    try:
        client.fget_object(
            "trains",
            key,
            f"{dest_folder}/{os.path.basename(key)}"
        )
        return True
    except ResponseError as err:
        print(err)
        return False
    except AccessDenied as err:
        print(f"AccessDenied: {err} -- check s3 credentials")
        return False


def ensure_input(input_files: list, local_dir: str) -> None:
    """Ensure inputs
    Manages the download of input files for the ML Subscriptions
    TODO:
    Load filenames from input params

    Args:
        input_files (list): Input filename list
        local_dir (str): Local destination folder
    """

    minioClient = get_minio_client()
    
    for file in input_files:
        if get_file(minioClient, 'ml/tests/' + file, local_dir):
            if os.path.isfile(
                os.path.join(local_dir, file)
            ):
                print(f"File {file} succesfully downloaded")
            else:
                print(f"Can't get input file {file}")
                sys.exit(1)
        else:
            print(f"Can't get input file {file}")
            sys.exit(1)


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    loss_func = nn.MSELoss()
    epoch_loss = 0
    for batch_idx, batch in enumerate(train_loader):
        data = batch['subs']
        target = batch['duration']

        X = data.to(device)
        Y = target.to(device)

        optimizer.zero_grad()
        output = model(X)
        loss = loss_func(output, Y)
        epoch_loss += loss
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            Logger.current_logger().report_scalar(
                "train", "loss", iteration=(epoch * len(train_loader) + batch_idx), value=loss.item())
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\Acc Loss: {:.6f}'.format(
                epoch, batch_idx * len(X), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item(), epoch_loss))


def test(args, model, device, test_loader, epoch):
    model.eval()
    test_loss = 0
    correct = 0
    loss_func = nn.MSELoss()
    with torch.no_grad():
        for _, batch in enumerate(test_loader):
            data = batch['subs']
            target = batch['duration']

            data, target = data.to(device), target.to(device)
            output = model(data)

            # Loss function
            # test_loss += loss_func(output, target, reduction='sum').item()  # sum up batch loss
            # This loss function has no reduction attribute
            test_loss += loss_func(output, target).item()  # sum up batch loss

            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    Logger.current_logger().report_scalar(
        "test", "loss", iteration=epoch, value=test_loss)
    Logger.current_logger().report_scalar(
        "test", "accuracy", iteration=epoch, value=(correct / len(test_loader.dataset)))
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def prepare_parser(parser):
    """Wrapper to configure training arguments
    It loads arguments into input parser, which is modified in place

    Args:
        parser (argparse.ArgumentParser): a parser to add args to
    """
    parser.add_argument(
        '--nn',
        nargs='+',
        type=int,
        default=(7, 1, 7),
        metavar='',
        help='Network size: [N M T O]. Where: N input size(features), \
            M hidden layers, T hidden layer size, O output size (default: 7 1 7 1)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=64,
        metavar='N',
        help='input batch size for training (default: 64)'
    )
    parser.add_argument(
        '--test-batch-size',
        type=int,
        default=1000,
        metavar='N',
        help='input batch size for testing (default: 1000)'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=10,
        metavar='N',
        help='number of epochs to train (default: 10)'
    )
    parser.add_argument(
        '--lr',
        type=float, 
        default=0.01,
        metavar='LR',
        help='learning rate (default: 0.01)'
    )
    parser.add_argument(
        '--momentum',
        type=float,
        default=0.5,
        metavar='M',
        help='SGD momentum (default: 0.5)'
    )
    parser.add_argument(
        '--no-cuda',
        action='store_true',
        default=False,
        help='disables CUDA training'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=1,
        metavar='S',
        help='random seed (default: 1)'
    )
    parser.add_argument(
        '--log-interval',
        type=int,
        default=10,
        metavar='N',
        help='how many batches to wait before logging training status'
    )
    parser.add_argument(
        '--save-model',
        action='store_true',
        default=True,
        help='For Saving the current Model'
    )


def main():
    """Allegro trains main
    """

    model_snapshots_path = '/tmp/trains'
    if not os.path.exists(model_snapshots_path):
        os.makedirs(model_snapshots_path)
    
    project_name = 'ML-Subscriptions'
    input_files = [
        'subs_dss_0.1_sorted_norm.csv',
        'subs_dss_0.1_sorted_norm_test.csv'
    ]
    task_name = 'v0.1.1'
    out_name = 'ml-subs'
    
    # Prepare training settings parser
    parser = argparse.ArgumentParser(description=project_name)
    prepare_parser(parser)
    args = parser.parse_args()
    print(args)

    # Prepare task
    task = Task.init(
        project_name=project_name,
        task_name=task_name,
        output_uri=model_snapshots_path
    )

    task.set_parameters_as_dict(vars(args))
    task.execute_remotely(queue_name="default")

    # Getting the config from agent
    session = Session()
    print(session.config.__dict__)

    use_cuda = not args.no_cuda and torch.cuda.is_available()

    print(f"Using Cuda: {use_cuda}")

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}

    # This is the default way to do it using trains configuration
    # not working till keys are not set in `trains.conf`
    # sm = StorageManager()
    # sm.get_local_copy(remote_url="s3://trains/ml/tests/subs_dss_0.1_sorted_norm.csv")

    # Instead, `ensure_input` does its job
    print("Ensure s3 files locally")
    ensure_input(input_files, model_snapshots_path)

    print("Loading dss")
    train_file = os.path.join(model_snapshots_path, input_files[0])
    train_ds = SubsDS(train_file)

    test_file = os.path.join(model_snapshots_path, input_files[1])
    test_ds = SubsDS(test_file)

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        **kwargs)
    
    test_loader = torch.utils.data.DataLoader(
        test_ds,
        batch_size=args.test_batch_size,
        shuffle=True,
        **kwargs)

    model = FFNN(*args.nn).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    out_name = f"{out_name}-{model.print()}"

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(args, model, device, test_loader, epoch)

    if (args.save_model):
        torch.save(model.state_dict(), os.path.join(gettempdir(), out_name))
    

if __name__ == '__main__':
    main()
