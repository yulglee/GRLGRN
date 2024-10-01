import torch
def parser_add_main_args(parser):
    # contra_loss-parameter
    parser.add_argument('--tau', type=float, default=0.07,
                        help='temperture of loss')
    parser.add_argument('--alpha_loss', type=float, default=0.08)
    parser.add_argument('--beta_loss', type=float, default=0.8)
    parser.add_argument('--gamma_loss', type=float, default=1)
    # downstream_task_parameter
    parser.add_argument('--device', type=str,
                        default=torch.device('cuda:' + str(0) if torch.cuda.is_available() else 'cpu'),
                        help='device')
    parser.add_argument('--epoch', type=int, default=50, help='epoch')
    parser.add_argument('--output_dim', type=int, default=256, help='GTN_output_dim')
    parser.add_argument('--lr', type=float, default=0.0002, help='learning_rate')
    parser.add_argument('--num_implicit_links', type=int, default=8, help='the channel of implicit_adj')
    parser.add_argument('--seed', type=int, default=3047, help='seed')
def get_train_parameter(name:str):
    if name == 'STRING Dataset':
        lr = 0.0002 
        pre_loss_cof = 1
        contra_loss_cof = 0
        use_contra_learning = False
        epoch = 20
    if name == 'Specific Dataset': 
        lr = 0.0002
        pre_loss_cof = 1
        contra_loss_cof = 0
        epoch = 50
        use_contra_learning = False
    if name == 'Non-Specific Dataset':
        lr = 0.0002
        pre_loss_cof = 0.2
        contra_loss_cof = 0.8
        epoch = 45
        use_contra_learning = True
    return lr, pre_loss_cof, contra_loss_cof, use_contra_learning, epoch
