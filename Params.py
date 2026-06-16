import argparse

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
def ParseArgs():
    parser = argparse.ArgumentParser(description='Model Params')
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--lr1', default=1e-4, type=float, help='learning rate for denoise model')
    parser.add_argument('--batch', default=2048, type=int, help='batch size')
    parser.add_argument('--tstBat', default=256, type=int, help='number of users in a testing batch')
    parser.add_argument('--reg', default=0.001, type=float, help='weight decay regularizer') 
    parser.add_argument('--ssl_reg_uu_ii', default=0.15, type=float, help='uni_uu_ii regularizer')
    parser.add_argument('--ssl_reg_ui', default=0.2, type=float, help='uni_ui regularizer')
    parser.add_argument('--temperature', default=5, type=float, help='uni_temperature')
    parser.add_argument('--cl_temperature', default=5, type=float, help='cl temperature')
    parser.add_argument('--temperature1', default=0.5, type=float, help='u-i_uni_temperature')
    parser.add_argument('--epoch', default=100, type=int, help='number of epochs')
    parser.add_argument('--decay', default=0, type=float, help='weight decay rate')
    parser.add_argument('--latdim', default=300, type=int, help='embedding size')
    parser.add_argument('--mask_r', default=1, type=float, help='mask ratio')
    parser.add_argument('--lp', default=0, type=float, help='mask ratio low bound')
    parser.add_argument('--gcn_layer0', default=2, type=int, help='number of gcn layers')

    parser.add_argument('--noise_scale', default=0.01, type=float, help='the scale of noise')
    parser.add_argument('--noise_min', default=0.005, type=float, help='lower bounds of the added noises')
    parser.add_argument('--noise_max', default=0.01, type=float, help='upper bounds of the added noises')
    parser.add_argument('--time_step', default=80, type=int, help='diffusion step')
    parser.add_argument('--sample_step', default=0, type=int, help='p_sample: denoise step')
    parser.add_argument('--noiseDirection', default=True, type=str2bool, help='noiseDirection')
    parser.add_argument('--sampleNoise', default=False, type=str2bool, help='sample noise')
    parser.add_argument('--elbo', default=0.01, type=float, help='diffusion regularizer')


    parser.add_argument('--mlp_dims', default='[600]', type=str, help='denoise model hidden dim')
    parser.add_argument('--emb_size', default=10, type=float, help='time emb size')
    parser.add_argument('--actFunc', default='tanh', type=str, help='activation function')
    parser.add_argument('--norm', default=False, type=str2bool, help='denoise normalize')
    parser.add_argument('--dropout', default=0.2, type=float, help='dropout')
    parser.add_argument('--dropout1', default=0, type=float, help='dropout')
    parser.add_argument('--residual', default=False, type=str2bool, help='residual net in denoise model')
    parser.add_argument('--sample_methon', default="importance", type=str, help='assign different weight to different timestep or not')
    parser.add_argument('--scale', default=0.2, type=float, help='weight of uncondition to condition')



    parser.add_argument('--load_model', default=None, help='model name to load')
    parser.add_argument('--data', default='ml-1m', type=str, help='name of dataset') # ml-1m yelp douban
    parser.add_argument('--tstEpoch', default=1, type=int, help='number of epoch to test while training')
    parser.add_argument('--gpu', default='0', type=str, help='indicates which gpu to use')
    parser.add_argument('--seed', default='3407', type=int, help='model seed')

    parser.add_argument('--mask_mode', default='rw_rl', type=str,help='mask mode: random, degree, rw, rw_rl')
    parser.add_argument('--rw_mask_ratio', default=0.2, type=float,help='target ratio of masked edges')
    parser.add_argument('--rw_walk_length', default=6, type=int,help='length of each random walk')
    parser.add_argument('--rw_num_walks', default=4, type=int,help='number of walks per seed')
    parser.add_argument('--rw_restart_prob', default=0.15, type=float,help='restart probability in random walk')
    parser.add_argument('--rw_seed_ratio', default=0.1, type=float,help='ratio of seed nodes')
    parser.add_argument('--rw_bias_alpha', default=0.0, type=float,help='degree bias strength, >0 high-degree, <0 low-degree')
    # rl
    parser.add_argument('--use_rl_mask', default=True, type=str2bool,help='whether to use RL to control rw mask')
    parser.add_argument('--rl_lr', default=1e-3, type=float,help='policy learning rate')
    parser.add_argument('--rl_gamma', default=0.9, type=float,help='reward discount')
    parser.add_argument('--rl_entropy_reg', default=1e-3, type=float,help='entropy regularization')
    parser.add_argument('--rl_warmup', default=5, type=int,help='epochs before enabling RL updates')
    parser.add_argument('--rl_reward_scale', default=1.0, type=float,help='reward scale')
    parser.add_argument('--diffusion_contrast_step', default=10, type=int,help='number of denoising steps for contrastive view')
    parser.add_argument('--contrast_loss_weight', default=0.05, type=float,help='weight of contrastive loss between diffused view and masked view')
    parser.add_argument('--use_rl_diffusion', default=True, type=str2bool,help='whether to use RL to control diffusion hyperparameters')
    parser.add_argument('--rl_diff_noise_scale_cands', default='[0.005,0.01,0.02]', type=str,help='candidate noise_scale values')
    parser.add_argument('--rl_diff_scale_cands', default='[0.1,0.2,0.5]', type=str,help='candidate scale values')
    parser.add_argument('--rl_diff_entropy_reg', default=0.01, type=float,help='entropy regularization for diffusion RL')
    parser.add_argument('--rl_diff_lr', default=1e-3, type=float,help='learning rate for diffusion RL policy')
    parser.add_argument('--mask_ratio_cands', default='[0.10,0.20,0.30]', type=str,help='candidate mask_ratio values for RL')
    parser.add_argument('--walk_length_cands', default='[4,6,8]', type=str,help='candidate walk_length values for RL')
    parser.add_argument('--restart_prob_cands', default='[0.10,0.20,0.30]', type=str,help='candidate restart_prob values for RL')
    parser.add_argument('--rl_exploration_bonus', default=0.01, type=float,help='entropy bonus added to reward')
    parser.add_argument('--rl_action_smooth_penalty', default=0.05, type=float,help='penalty per unit change of continuous action (Euclidean)')
    parser.add_argument('--rl_alternate', default=True, type=str2bool,help='alternate update between mask and diff policy')
    parser.add_argument('--rl_hidden_dim', default=64, type=int,help='hidden dimension for policy networks')
    parser.add_argument('--do_tsne', default=False, type=str2bool, help='generate t-SNE after training')

    return parser.parse_args()


args = ParseArgs()