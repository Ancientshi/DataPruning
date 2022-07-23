import argparse
import math

_GLOBAL_ARGS = None


def get_args():
    global _GLOBAL_ARGS
    if _GLOBAL_ARGS is None:
        _GLOBAL_ARGS = parse_args()

    return _GLOBAL_ARGS


def str2bool(v):
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Unsupported value encountered.")


def parse_args(ignore_unknown_args=False):
    parser = argparse.ArgumentParser(description='LIE Training')
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    # parser.add_argument('--save-path', default='''/workspace/DataPruning/TrainPhase_dataPruning_bot5000/''', type=str, help='model save path')
    #parser.add_argument('--save-path', default='''/workspace/DataPruning/TrainPhase/''', type=str, help='model save path')
    parser.add_argument('--save-path', default='''/workspace/DataPruning/TrainPhase_randomPruning_5000/''', type=str, help='model save path')
    parser.add_argument('--net-name', default='''MyNet''', type=str, help='used net name')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--weight-decay', default=0.003, type=float, help='weight_decay')
    parser.add_argument('--train-batch-size', default=200, type=int, help='train batch size')
    parser.add_argument('--test-batch-size', default=1000, type=int, help='test batch size')
    parser.add_argument('--epoch', default=20, type=int, help='train epoch')
    parser.add_argument('--resume', '-r', action='store_true',
                        help='resume from checkpoint')                  
    args = parser.parse_args()
    return args


def _print_args(args):
    print("------------------------ arguments ------------------------", flush=True)
    str_list = []
    for arg in vars(args):
        dots = "." * (48 - len(arg))
        str_list.append("  {} {} {}".format(arg, dots, getattr(args, arg)))
    for arg in sorted(str_list, key=lambda x: x.lower()):
        print(arg, flush=True)
    print("-------------------- end of arguments ---------------------", flush=True)


if __name__ == "__main__":
    get_args()