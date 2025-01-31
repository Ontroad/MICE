import argparse

import omnisafe
from omnisafe.utils.tools import custom_cfgs_to_dict, update_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--algo',
        type=str,
        metavar='ALGO',
        default='MICE', 
        help='algorithm to train',
        choices=omnisafe.ALGORITHMS['all'],
    )
    parser.add_argument(
        '--env-id',
        type=str,
        metavar='ENV',
        default='SafetyPointGoal1-v0',
        help='the name of test environment',
    )
    parser.add_argument(
        '--parallel',
        default=1,
        type=int,
        metavar='N',
        help='number of paralleled progress for calculations.',
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda:0',
        metavar='DEVICES',
        help='device to use for training',
    )
    parser.add_argument(
        '--vector-env-nums',
        type=int,
        default=1, 
        metavar='VECTOR-ENV',
        help='number of vector envs to use for training',
    )
    parser.add_argument(
        '--torch-threads',
        type=int,
        default=16,
        metavar='THREADS',
        help='number of threads to use for torch',
    )
    parser.add_argument(
        '--log-dir',
        type=str,
        default='./',  
        metavar='LOG-DIR',
        help='save logger path',
    )
    args, unparsed_args = parser.parse_known_args()
    keys = [k[2:] for k in unparsed_args[0::2]]
    values = list(unparsed_args[1::2])
    unparsed_args = dict(zip(keys, values))

    var_args = vars(args)
    custom_cfgs = {
        'algo_cfgs': {
            'steps_per_epoch': var_args['steps_per_epoch'],
        },
        'logger_cfgs': {
            'log_dir': var_args['log_dir'],
        },
    }
    train_cfgs = {
        'device': var_args['device'],
        'torch_threads': var_args['torch_threads'],
        'vector_env_nums': var_args['vector_env_nums'],
        'parallel': var_args['parallel'],
        'total_steps': var_args['total_steps'],
    }

    for k, v in unparsed_args.items():
        update_dict(custom_cfgs, custom_cfgs_to_dict(k, v))

    agent = omnisafe.Agent(
        args.algo,
        args.env_id,
        train_terminal_cfgs=train_cfgs,
        custom_cfgs=custom_cfgs,
    )
    agent.learn()
