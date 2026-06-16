import os
# 先导入 args 以获取 gpu 参数（Params.py 不依赖 torch）
from Params import args

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TORCH_USE_CUDA_DSA'] = '1'
import torch
import torch.utils.data as dataloader
import numpy as np
import Utils.TimeLogger as logger
from Utils.TimeLogger import log
from Model import Model, Diffusion, Denoise_NN
from DataHandler import DataHandler
from DataHandler import TrnData
import pickle
from Utils.Utils import *
from Utils.Utils import contrast
import random
import setproctitle
from copy import deepcopy
from datetime import datetime
import time
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Available GPUs:", torch.cuda.device_count())
print("Current GPU:", torch.cuda.current_device())
print("GPU name:", torch.cuda.get_device_name())


class MaskPolicyNet(nn.Module):
    def __init__(self, state_dim=13, hidden_dim=64, action_dim=3):
        super(MaskPolicyNet, self).__init__()
        self.backbone = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.mu_head = nn.Linear(hidden_dim, action_dim)
        self.log_sigma_head = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        feat = self.backbone(state)
        mu = self.mu_head(feat)
        log_sigma = self.log_sigma_head(feat)
        log_sigma = torch.clamp(log_sigma, min=-5, max=2)
        return mu, log_sigma


class DiffPolicyNet(nn.Module):
    def __init__(self, state_dim=13, hidden_dim=64, action_dim=2):
        super(DiffPolicyNet, self).__init__()
        self.backbone = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.mu_head = nn.Linear(hidden_dim, action_dim)
        self.log_sigma_head = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        feat = self.backbone(state)
        mu = self.mu_head(feat)
        log_sigma = self.log_sigma_head(feat)
        log_sigma = torch.clamp(log_sigma, min=-5, max=2)
        return mu, log_sigma


class Coach:
    def __init__(self, handler):
        self.handler = handler

        print('USER', args.user, 'ITEM', args.item)
        print('NUM OF INTERACTIONS', self.handler.trnLoader.dataset.__len__())
        self.metrics = dict()
        mets = ['Loss', 'preLoss', 'Recall', 'NDCG']
        for met in mets:
            self.metrics['Train' + met] = list()
            self.metrics['Test' + met] = list()
        self.prev_train_loss = None
        self.prev_train_bpr = None
        self.prev_contrast = None
        self.reward_baseline = 0.0
        self._last_cont_action = None
        self.action_ranges = {
            'mask_ratio': (0.05, 0.35),
            'walk_length': (3, 7),
            'restart_prob': (0.1, 0.4),
            'noise_scale': (0.005, 0.02),
            'scale': (0.1, 0.4),
        }
        self.mask_policy = MaskPolicyNet(state_dim=13, hidden_dim=getattr(args, 'rl_hidden_dim', 64)).to(device)
        self.diff_policy = DiffPolicyNet(state_dim=13, hidden_dim=getattr(args, 'rl_hidden_dim', 64)).to(device)
        self.mask_opt = torch.optim.Adam(self.mask_policy.parameters(), lr=getattr(args, 'rl_lr', 1e-3))
        self.diff_opt = torch.optim.Adam(self.diff_policy.parameters(), lr=getattr(args, 'rl_diff_lr', 1e-3))


    def map_continuous_action(self, action_val, param_name):
        val = action_val if isinstance(action_val, float) else action_val.item()
        val = max(-1.0, min(1.0, val))
        low, high = self.action_ranges[param_name]
        if param_name == 'walk_length':
            raw = low + (val + 1) / 2 * (high - low)
            return int(round(raw))
        else:
            return low + (val + 1) / 2 * (high - low)

    def sample_continuous_action(self, state, policy_net, param_names, use_smoothing=True, smooth_prob=0.7):
        if use_smoothing and self._last_cont_action is not None and np.random.rand() < smooth_prob:
            action = {k: self._last_cont_action[k] for k in param_names}
            return action, torch.tensor(0.0, device=device), torch.tensor(0.0, device=device)

        mu, log_sigma = policy_net(state)
        sigma = log_sigma.exp()
        dist = Normal(mu, sigma)
        raw_action = dist.rsample()
        action_tanh = torch.tanh(raw_action)
        action = {}
        for i, pname in enumerate(param_names):
            action[pname] = self.map_continuous_action(action_tanh[0, i], pname)
        log_prob = dist.log_prob(raw_action) - torch.log(1 - action_tanh.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        if self._last_cont_action is None:
            self._last_cont_action = {}
        self._last_cont_action.update(action)

        return action, log_prob, entropy

    def sample_mask_action(self, state, use_smoothing=True):
        param_names = ['mask_ratio', 'walk_length', 'restart_prob']
        return self.sample_continuous_action(state, self.mask_policy, param_names, use_smoothing)

    def sample_diff_action(self, state, use_smoothing=True):
        param_names = ['noise_scale', 'scale']
        return self.sample_continuous_action(state, self.diff_policy, param_names, use_smoothing)

    def makePrint(self, name, ep, reses, save):
        ret = 'Epoch %d/%d, %s: ' % (ep, args.epoch, name)
        for metric in reses:
            val = reses[metric]
            ret += '%s = %.4f, ' % (metric, val)
            tem = name + metric
            if save and tem in self.metrics:
                self.metrics[tem].append(val)
        ret = ret[:-2] + '  '
        return ret

    def makePrintRes(self, name, ep, reses):
        ret = 'Epoch %d/%d, %s: ' % (ep, args.epoch, name)
        for metric in reses:
            val = reses[metric]
            ret += '%s = %.4f, ' % (metric, val)
        ret = ret[:-2] + '  '
        return ret

    def build_mask_state(self, ep):
        with torch.no_grad():
            usrEmbeds, itmEmbeds, _, _, _ = self.model(
                self.handler.torchBiAdj, self.handler.torchD_1Aadj
            )
            user_stats = torch.cat([usrEmbeds.mean(dim=0), usrEmbeds.std(dim=0)], dim=-1)
            item_stats = torch.cat([itmEmbeds.mean(dim=0), itmEmbeds.std(dim=0)], dim=-1)

        trn = self.handler.trnMat.tocsr()
        user_deg = np.array(trn.sum(axis=1)).reshape(-1)
        item_deg = np.array(trn.sum(axis=0)).reshape(-1)
        density = trn.nnz / float(args.user * args.item)
        graph_stats = torch.tensor([
            density,
            float(user_deg.mean()),
            float(item_deg.mean()),
            float((user_deg == 1).mean())
        ], dtype=torch.float32, device=device)

        mask_ratio = 0.0
        if hasattr(self.handler, 'masktorchBiAdj') and self.handler.masktorchBiAdj is not None:
            if self._last_cont_action is not None and 'mask_ratio' in self._last_cont_action:
                mask_ratio = self._last_cont_action['mask_ratio']

        last_loss = 0.0 if self.prev_train_loss is None else self.prev_train_loss

        user_mean_norm = torch.norm(user_stats[:args.latdim])
        user_std_norm = torch.norm(user_stats[args.latdim:])
        item_mean_norm = torch.norm(item_stats[:args.latdim])
        item_std_norm = torch.norm(item_stats[args.latdim:])

        cur_noise_scale = self.diffusion_model.noise_scale
        cur_scale = self.diffusion_model.scale
        noise_low, noise_high = self.action_ranges['noise_scale']
        norm_noise_scale = (cur_noise_scale - noise_low) / (noise_high - noise_low)
        norm_noise_scale = max(0.0, min(1.0, norm_noise_scale))
        scale_low, scale_high = self.action_ranges['scale']
        norm_scale = (cur_scale - scale_low) / (scale_high - scale_low)
        norm_scale = max(0.0, min(1.0, norm_scale))

        state = torch.tensor([
            ep / max(1, args.epoch),
            last_loss,
            user_mean_norm.item(),
            user_std_norm.item(),
            item_mean_norm.item(),
            item_std_norm.item(),
            density,
            float(user_deg.mean()),
            float(item_deg.mean()),
            float((user_deg == 1).mean()),
            mask_ratio,
            norm_noise_scale,
            norm_scale,
        ], dtype=torch.float32, device=device).unsqueeze(0)
        return state

    def run(self):
        self.prepareModel()
        log('Model Prepared')
        if args.load_model:
            self.loadModel()
            stloc = len(self.metrics['TrainLoss']) * args.tstEpoch - (args.tstEpoch - 1)
        else:
            stloc = 0
            log('Model Initialized')
        curTime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        fileName = f'trainingResult{args.data}-{curTime}'
        with open('./Result/' + fileName + '.txt', 'w') as f:
            hypeParameters = vars(args)
            f.write("HyperParameters:\n")
            for k, v in hypeParameters.items():
                f.write(f"{k}: {v}\n")

        bestRes = None
        for ep in range(stloc, args.epoch):
            tstFlag = (ep % args.tstEpoch == 0)

            reses = self.trainEpoch(ep)
            log(self.makePrint('Train', ep, reses, tstFlag))

            with open('./Result/' + fileName + '.txt', 'a') as f:
                f.write(f"{self.makePrintRes('Train', ep, reses)}\n")

            if tstFlag:
                with torch.no_grad():
                    reses = self.testEpoch()
                    with open('./Result/' + fileName + '.txt', 'a') as f:
                        f.write(f"{self.makePrintRes('Test', ep, reses)}\n")
                log(self.makePrint('Test', ep, reses, tstFlag))
                bestRes = reses if bestRes is None or reses['Recall10'] > bestRes['Recall10'] else bestRes
            print()

        with torch.no_grad():
            reses = self.testEpoch()
            self.saveRecord(reses, fileName)
            bestRes = reses if bestRes is None or reses['Recall10'] > bestRes['Recall10'] else bestRes

        log(self.makePrint('Test', args.epoch, reses, True))
        log(self.makePrint('Best Result', args.epoch, bestRes, True))
        with open('./Result/' + fileName + '.txt', 'a') as f:
            f.write(f"{self.makePrintRes('Test', ep, reses)}\n")
            f.write(f"{self.makePrintRes('Best Result', args.epoch, bestRes)}")

    def prepareModel(self):
        self.model = Model().to(device)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.decay)
        self.diffusion_model = Diffusion(args.noise_scale, args.noise_min, args.noise_max, args.time_step, scale=args.scale).to(device)
        mlp_out_dims = eval(args.mlp_dims) + [args.latdim]
        mlp_in_dims = mlp_out_dims[::-1]
        self.denoise_model = Denoise_NN(mlp_in_dims, mlp_out_dims, args.emb_size).to(device)
        self.opt1 = torch.optim.Adam(self.denoise_model.parameters(), lr=args.lr1, weight_decay=0)

    def trainEpoch(self, ep):
        epLoss, epPreLoss, epreg_loss, epunformity_loss, epelbo_loss = 0, 0, 0, 0, 0
        ep_contrast_total = 0.0
        state = self.build_mask_state(ep)

        if ep == getattr(args, 'rl_warmup', 5):
            self._last_cont_action = None
        use_mask_rl = getattr(args, 'use_rl_mask', False) and ep >= getattr(args, 'rl_warmup', 5)
        if use_mask_rl:
            mask_action, mask_log_prob, mask_entropy = self.sample_mask_action(state, use_smoothing=False)
            print(f"[DEBUG] Epoch {ep} Mask Action: {mask_action}")
        else:
            mask_action = {
                'mask_ratio': float(getattr(args, 'rw_mask_ratio', 0.2)),
                'walk_length': int(getattr(args, 'rw_walk_length', 6)),
                'restart_prob': float(getattr(args, 'rw_restart_prob', 0.15)),
                'num_walks': int(getattr(args, 'rw_num_walks', 4)),
                'seed_ratio': float(getattr(args, 'rw_seed_ratio', 0.1)),
                'bias_alpha': float(getattr(args, 'rw_bias_alpha', 0.0)),
            }
            mask_log_prob = torch.tensor(0.0, device=device)
            mask_entropy = torch.tensor(0.0, device=device)
        use_diff_rl = getattr(args, 'use_rl_diffusion', False) and ep >= getattr(args, 'rl_warmup', 5)
        if use_diff_rl:
            diff_action, diff_log_prob, diff_entropy = self.sample_diff_action(state, use_smoothing=False)
            print(f"[DEBUG] Epoch {ep} Diff Action: {diff_action}")
        else:
            diff_action = {
                'noise_scale': float(getattr(args, 'noise_scale', 0.01)),
                'scale': float(getattr(args, 'scale', 0.2)),
            }
            diff_log_prob = torch.tensor(0.0, device=device)
            diff_entropy = torch.tensor(0.0, device=device)

        action = {**mask_action, **diff_action}
        if use_diff_rl:
            self.diffusion_model.update_params(noise_scale=action['noise_scale'], scale=action['scale'])
        maskmat, visited_edges = self.handler.maskedge_random_walk(
            self.handler.trnMat,
            mask_ratio=action['mask_ratio'],
            walk_length=action['walk_length'],
            num_walks=getattr(args, 'rw_num_walks', 4),
            restart_prob=action['restart_prob'],
            seed_ratio=getattr(args, 'rw_seed_ratio', 0.1),
            bias_alpha=getattr(args, 'rw_bias_alpha', 0.0)
        )
        self.handler.masktorchBiAdj, self.handler.maskD_1Aadj = self.handler.makeTorchAdj(maskmat)

        with torch.no_grad():
            usrEmbeds_mask, itmEmbeds_mask, condition_embeds_mask, _, _ = self.model(
                self.handler.masktorchBiAdj, self.handler.maskD_1Aadj
            )
            embeds_mask = torch.cat([usrEmbeds_mask, itmEmbeds_mask], dim=0)
            embeds_mask_recon = self.diffusion_model.p_sample(
                self.denoise_model, embeds_mask, condition_embeds_mask,
                step=args.diffusion_contrast_step,
                noise_d=args.noiseDirection,
                sample_noise=False
            )
            usrEmbeds_mask_recon = embeds_mask_recon[:args.user]
            itmEmbeds_mask_recon = embeds_mask_recon[args.user:]

        trnLoader = self.handler.trnLoader
        trnLoader.dataset.negSampling()
        steps = trnLoader.dataset.__len__() // args.batch

        self.model.train()
        self.diffusion_model.train()
        self.denoise_model.train()
        self.mask_policy.train()
        self.diff_policy.train()

        for i, tem in enumerate(trnLoader):
            usrEmbeds_mask, itmEmbeds_mask, condition_embeds_mask, _, _ = self.model(
                self.handler.masktorchBiAdj, self.handler.maskD_1Aadj
            )

            ancs, poss, neg = tem
            ancs = ancs.long().to(device)
            poss = poss.long().to(device)
            neg = neg.long().to(device)

            ancEmbeds_mask = usrEmbeds_mask[ancs]
            posEmbeds_mask = itmEmbeds_mask[poss]
            negEmbeds_mask = itmEmbeds_mask[neg]

            ancEmbeds_mask_recon_batch = usrEmbeds_mask_recon[ancs].detach()
            posEmbeds_mask_recon_batch = itmEmbeds_mask_recon[poss].detach()
            contrast_loss_user = contrast_loss(ancEmbeds_mask_recon_batch, ancEmbeds_mask, temperature=args.cl_temperature)
            contrast_loss_item = contrast_loss(posEmbeds_mask_recon_batch, posEmbeds_mask, temperature=args.cl_temperature)
            contrast_total = (contrast_loss_user + contrast_loss_item) * args.contrast_loss_weight
            batch_idx = torch.cat([poss, neg], dim=0)
            batch_idx += args.user
            batch_idx = torch.cat([ancs, batch_idx], dim=0)
            batch_idx = torch.unique(batch_idx)

            embeds = torch.cat([usrEmbeds_mask, itmEmbeds_mask], dim=0)
            diff_loss, _, _ = self.diffusion_model.training_loss(
                self.denoise_model, embeds, condition_embeds_mask, args.noiseDirection, batch_idx
            )
            elbo = diff_loss.mean() * args.elbo
            ancs_unique = torch.unique(ancs)
            pos_unique = torch.unique(poss)
            uniformity_loss = (
                    Uniformity_loss2(usrEmbeds_mask[ancs_unique], usrEmbeds_mask[ancs_unique], args.temperature) +
                    Uniformity_loss2(itmEmbeds_mask[pos_unique], itmEmbeds_mask[pos_unique], args.temperature)
            ) * args.ssl_reg_uu_ii + Uniformity_loss2(
                usrEmbeds_mask[ancs_unique], itmEmbeds_mask[pos_unique], args.temperature1
            ) * args.ssl_reg_ui
            BPRloss = torch.mean(
                -torch.log(10e-6 + torch.sigmoid(pairPredict(ancEmbeds_mask, posEmbeds_mask, negEmbeds_mask))))
            regLoss = calcRegLoss_normal(ancEmbeds_mask, posEmbeds_mask, negEmbeds_mask) * args.reg

            loss = BPRloss + regLoss + elbo + uniformity_loss + contrast_total

            epLoss += loss.item()
            epPreLoss += BPRloss.item()
            epelbo_loss += elbo.item()
            epunformity_loss += uniformity_loss.item()
            epreg_loss += regLoss.item()
            ep_contrast_total += contrast_total.item()

            self.opt.zero_grad()
            self.opt1.zero_grad()
            loss.backward()
            self.opt.step()
            self.opt1.step()
        steps = max(1, steps)
        ret = dict()
        ret['Loss'] = epLoss / steps
        ret['bprLoss'] = epPreLoss / steps
        ret['elbo_loss'] = epelbo_loss / steps
        ret['epunformity_loss'] = epunformity_loss / steps
        ret['epreg_loss'] = epreg_loss / steps
        ret['contrast_loss'] = ep_contrast_total / steps
        ret['mask_ratio'] = action['mask_ratio']
        ret['walk_length'] = action['walk_length']
        ret['restart_prob'] = action['restart_prob']
        ret['noise_scale'] = action['noise_scale']
        ret['scale'] = action['scale']
        ret['masked_edges'] = len(visited_edges)

        cur_loss = ret['Loss']
        cur_bpr = ret['bprLoss']
        cur_contrast = ret['contrast_loss']

        if self.prev_train_bpr is None:
            reward = 0.0
        else:
            reward = (self.prev_train_bpr - cur_bpr) + 0.05 * (self.prev_train_loss - cur_loss) + 0.1 * (self.prev_contrast - cur_contrast)
        smooth_penalty = 0.0
        if self._last_cont_action is not None and use_mask_rl and use_diff_rl:
            delta_mask = (mask_action['mask_ratio'] - self._last_cont_action['mask_ratio'])**2 + \
                         ((mask_action['walk_length'] - self._last_cont_action['walk_length'])/10)**2 + \
                         (mask_action['restart_prob'] - self._last_cont_action['restart_prob'])**2
            delta_diff = (diff_action['noise_scale'] - self._last_cont_action['noise_scale'])**2 + \
                         (diff_action['scale'] - self._last_cont_action['scale'])**2
            delta = np.sqrt(delta_mask) + np.sqrt(delta_diff)
            smooth_penalty = args.rl_action_smooth_penalty * delta
        reward -= smooth_penalty

        entropy_bonus = (mask_entropy + diff_entropy) * args.rl_exploration_bonus
        reward += entropy_bonus.item()

        self.reward_baseline = 0.9 * self.reward_baseline + 0.1 * reward
        advantage = reward - self.reward_baseline
        update_mask = False
        update_diff = False
        if getattr(args, 'rl_alternate', True):
            if ep % 2 == 0:
                update_mask = use_mask_rl
            else:
                update_diff = use_diff_rl
        else:
            update_mask = use_mask_rl
            update_diff = use_diff_rl

        if update_mask and mask_log_prob.requires_grad:
            policy_loss = -mask_log_prob * advantage - args.rl_entropy_reg * mask_entropy
            self.mask_opt.zero_grad()
            policy_loss.backward()
            self.mask_opt.step()
            log('RL Update (Mask Cont): reward=%.6f, baseline=%.6f, advantage=%.6f' % (reward, self.reward_baseline, advantage), save=False)

        if update_diff and diff_log_prob.requires_grad:
            policy_loss = -diff_log_prob * advantage - getattr(args, 'rl_diff_entropy_reg', 1e-3) * diff_entropy
            self.diff_opt.zero_grad()
            policy_loss.backward()
            self.diff_opt.step()
            log('RL Update (Diff Cont): reward=%.6f, baseline=%.6f, advantage=%.6f' % (reward, self.reward_baseline, advantage), save=False)

        self.prev_train_loss = cur_loss
        self.prev_train_bpr = cur_bpr
        self.prev_contrast = cur_contrast

        return ret

    def testEpoch(self):
        tstLoader = self.handler.tstLoader
        epLoss, epRecall10, epNdcg10, epRecall20, epNdcg20 = [0] * 5
        i = 0
        num = tstLoader.dataset.__len__()
        steps = num // args.tstBat
        self.model.eval()
        self.diffusion_model.eval()
        self.denoise_model.eval()
        usrEmbeds, itmEmbeds, condition_embeds, _, _ = self.model(self.handler.torchBiAdj, self.handler.torchD_1Aadj)
        embeds = torch.cat([usrEmbeds, itmEmbeds], dim=0)
        embeds_recon = self.diffusion_model.p_sample(self.denoise_model, embeds, condition_embeds, args.sample_step,
                                                     args.noiseDirection)
        usrEmbeds = embeds_recon[:args.user]
        itmEmbeds = embeds_recon[args.user:]

        for usr, trnMask in tstLoader:
            i += 1
            usr = usr.long().to(device)
            trnMask = trnMask.to(device)
            allPreds = torch.mm(usrEmbeds[usr], torch.transpose(itmEmbeds, 1, 0)) * (1 - trnMask) - trnMask * 1e8
            _, topLocs10 = torch.topk(allPreds, 10)
            _, topLocs20 = torch.topk(allPreds, 20)
            recall10, ndcg10 = self.calcRes(topLocs10.cpu().numpy(), self.handler.tstLoader.dataset.tstLocs, usr, 10)
            recall20, ndcg20 = self.calcRes(topLocs20.cpu().numpy(), self.handler.tstLoader.dataset.tstLocs, usr, 20)
            epRecall10 += recall10
            epNdcg10 += ndcg10
            epRecall20 += recall20
            epNdcg20 += ndcg20
            log('Steps %d/%d: recall = %.1f, ndcg = %.1f          ' % (i, steps, recall10, ndcg10), save=False,
                oneline=True)
            log('Steps %d/%d: recall = %.1f, ndcg = %.1f          ' % (i, steps, recall20, ndcg20), save=False,
                oneline=True)

        ret = dict()
        ret['Recall10'] = epRecall10 / num
        ret['NDCG10'] = epNdcg10 / num
        ret['Recall20'] = epRecall20 / num
        ret['NDCG20'] = epNdcg20 / num
        return ret

    def calcRes(self, topLocs, tstLocs, batIds, topk):
        assert topLocs.shape[0] == len(batIds)
        allRecall = allNdcg = 0
        for i in range(len(batIds)):
            temTopLocs = list(topLocs[i])
            temTstLocs = tstLocs[batIds[i]]
            tstNum = len(temTstLocs)
            maxDcg = np.sum([np.reciprocal(np.log2(loc + 2)) for loc in range(min(tstNum, topk))])
            recall = dcg = 0
            for val in temTstLocs:
                if val in temTopLocs:
                    recall += 1
                    dcg += np.reciprocal(np.log2(temTopLocs.index(val) + 2))
            recall = recall / tstNum
            ndcg = dcg / maxDcg
            allRecall += recall
            allNdcg += ndcg
        return allRecall, allNdcg

    def loadModel(self):
        ckp = torch.load('./Models/' + args.load_model + '.mod')
        self.model = ckp['model']
        self.opt = torch.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=0)

        with open('./History/' + args.load_model + '.his', 'rb') as fs:
            self.metrics = pickle.load(fs)
        log('Model Loaded')

    def saveRecord(self, reses, fileName):
        pass


if __name__ == '__main__':
    logger.saveDefault = True
    log('Start')
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    handler = DataHandler()
    handler.LoadData()
    log('Load Data')
    coach = Coach(handler)
    coach.run()