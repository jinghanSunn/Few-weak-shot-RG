import os
from abc import abstractmethod
import json
import time
import torch
import pandas as pd
from numpy import inf
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


class BaseTrainer(object):
    def __init__(self, model, criterion, metric_ftns, args):
        self.args = args

        # setup GPU device if available, move model into configured device
        self.device, device_ids = self._prepare_device(args.n_gpu)
        # self.model = model.to(self.device)
        # print(self.device, device_ids)
        # if len(device_ids) > 1:
        
        self.model = model

        self.criterion = criterion
        self.metric_ftns = metric_ftns
        # self.optimizer = optimizer

        # self.epochs = self.args.epochs
        self.save_period = self.args.save_period

        self.mnt_mode = args.monitor_mode
        # self.mnt_metric = 'val_' + args.monitor_metric
        self.mnt_metric_test = 'test_' + args.monitor_metric
        assert self.mnt_mode in ['min', 'max']

        self.mnt_best = inf if self.mnt_mode == 'min' else -inf
        self.early_stop = getattr(self.args, 'early_stop', inf)

        self.start_epoch = 1
        self.checkpoint_dir = args.save_dir

        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        # if args.resume is not None:
        #     self._resume_checkpoint(args.resume)

        self.best_recorder = {
            # 'val': {self.mnt_metric: self.mnt_best},
                              'test': {self.mnt_metric_test: self.mnt_best}}

    @abstractmethod
    def _train_epoch(self, epoch):
        raise NotImplementedError

    def train(self, epoch, dataloader, fasterRCNN, model, args, test_semantic_embedding=None):
        if not os.path.exists(self.args.record_dir):
            os.mkdir(self.args.record_dir)
        ## record all the val
        record_json = {}
        not_improved_count = 0
        # for epoch in range(self.start_epoch, self.epochs + 1):
        result, test_gts, test_res = self._train_epoch(epoch, dataloader, fasterRCNN, model, args, test_semantic_embedding=None)

        # save outputs each epoch
        save_outputs = {'gts': test_gts, 'res': test_res}
        with open(os.path.join(self.args.record_dir, str(epoch)+'_token_results.json'), 'w') as f:
            json.dump(save_outputs, f)

        # save logged informations into log dict
        log = {'epoch': epoch}
        log.update(result)
        self._record_best(log)
        record_json[epoch] = log

        # print logged informations to the screen
        for key, value in log.items():
            print('\t{:15s}: {}'.format(str(key), value))

        # evaluate model performance according to configured metric, save best checkpoint as model_best
        best = False
        if self.mnt_mode != 'off':
            try:
                # check whether model performance improved or not, according to specified metric(mnt_metric)
                improved = (self.mnt_mode == 'min' and log[self.mnt_metric_test] <= self.mnt_best) or \
                            (self.mnt_mode == 'max' and log[self.mnt_metric_test] >= self.mnt_best)
            except KeyError:
                print("Warning: Metric '{}' is not found. " "Model performance monitoring is disabled.".format(
                    self.mnt_metric_test))
                self.mnt_mode = 'off'
                improved = False

            if improved:
                self.mnt_best = log[self.mnt_metric_test]
                not_improved_count = 0
                best = True
                save_outputs = {'gts': test_gts, 'res': test_res}
                with open(os.path.join(self.args.record_dir, 'best_word_results.json'), 'w') as f:
                    json.dump(save_outputs, f)
            else:
                not_improved_count += 1

            # if not_improved_count > self.early_stop:
            #     print("Validation performance didn\'t improve for {} epochs. " "Training stops.".format(
            #         self.early_stop))
            #     break

            # if epoch % self.save_period == 0:
            #     self._save_checkpoint(epoch, save_best=best)
        self._print_best()
        self._print_best_to_file()
        self._save_file(record_json)

    def _save_file(self, log):
        if not os.path.exists(self.args.record_dir):
            os.mkdir(self.args.record_dir)
        record_path = os.path.join(self.args.record_dir, self.args.dataset_name + '.json')
        with open(record_path, 'w') as f:
            json.dump(log, f)

    def _print_best_to_file(self):
        crt_time = time.asctime(time.localtime(time.time()))
        # self.best_recorder['val']['time'] = crt_time
        self.best_recorder['test']['time'] = crt_time
        # self.best_recorder['val']['seed'] = self.args.seed
        # self.best_recorder['test']['seed'] = self.args.seed
        # self.best_recorder['val']['best_model_from'] = 'val'
        self.best_recorder['test']['best_model_from'] = 'test'

        if not os.path.exists(self.args.record_dir):
            os.makedirs(self.args.record_dir)
        record_path = os.path.join(self.args.record_dir, self.args.dataset_name+'.csv')
        if not os.path.exists(record_path):
            record_table = pd.DataFrame()
        else:
            record_table = pd.read_csv(record_path)
        # record_table = record_table.append(self.best_recorder['val'], ignore_index=True)
        record_table = record_table.append(self.best_recorder['test'], ignore_index=True)
        record_table.to_csv(record_path, index=False)

    def _prepare_device(self, n_gpu_use):
        n_gpu = torch.cuda.device_count()
        if n_gpu_use > 0 and n_gpu == 0:
            print("Warning: There\'s no GPU available on this machine," "training will be performed on CPU.")
            n_gpu_use = 0
        if n_gpu_use > n_gpu:
            print(
                "Warning: The number of GPU\'s configured to use is {}, but only {} are available " "on this machine.".format(
                    n_gpu_use, n_gpu))
            n_gpu_use = n_gpu
        device = torch.device('cuda' if n_gpu_use > 0 else 'cpu')
        list_ids = list(range(n_gpu_use))
        return device, list_ids

    def _save_checkpoint(self, epoch, save_best=False):
        state = {
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            # 'optimizer': self.optimizer.state_dict(),
            'monitor_best': self.mnt_best
        }
        filename = os.path.join(self.checkpoint_dir, 'current_checkpoint.pth')
        torch.save(state, filename)
        print("Saving checkpoint: {} ...".format(filename))
        if save_best:
            best_path = os.path.join(self.checkpoint_dir, 'model_best.pth')
            torch.save(state, best_path)
            print("Saving current best: model_best.pth ...")

    def _resume_checkpoint(self, resume_path):
        resume_path = str(resume_path)
        print("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path)
        self.start_epoch = checkpoint['epoch'] + 1
        self.mnt_best = checkpoint['monitor_best']
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])

        print("Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch))

    def _record_best(self, log):
        # improved_val = (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.best_recorder['val'][
        #     self.mnt_metric]) or \
        #                (self.mnt_mode == 'max' and log[self.mnt_metric] >= self.best_recorder['val'][self.mnt_metric])
        # if improved_val:
        #     self.best_recorder['val'].update(log)

        improved_test = (self.mnt_mode == 'min' and log[self.mnt_metric_test] <= self.best_recorder['test'][
            self.mnt_metric_test]) or \
                        (self.mnt_mode == 'max' and log[self.mnt_metric_test] >= self.best_recorder['test'][
                            self.mnt_metric_test])
        if improved_test:
            self.best_recorder['test'].update(log)

    def _print_best(self):
        # print('Best results (w.r.t {}) in validation set:'.format(self.args.monitor_metric))
        # for key, value in self.best_recorder['val'].items():
        #     print('\t{:15s}: {}'.format(str(key), value))

        print('Best results (w.r.t {}) in test set:'.format(self.args.monitor_metric))
        for key, value in self.best_recorder['test'].items():
            print('\t{:15s}: {}'.format(str(key), value))


class Trainer(BaseTrainer):
    def __init__(self, model, criterion, metric_ftns, args, lr_scheduler,
                 test_dataloader):
        super(Trainer, self).__init__(model, criterion, metric_ftns, args)
        self.lr_scheduler = lr_scheduler
        self.test_dataloader = test_dataloader
        ## check the training
        self.writer = SummaryWriter()

    def _train_epoch(self, epoch, dataloader, fasterRCNN, model, args, test_semantic_embedding=None):

        # train_loss = 0
        # print_loss = 0

        # self.model.train()
        # for batch_idx, (images_id, images, reports_ids, reports_masks) in enumerate(tqdm(self.train_dataloader)):
        #     images, reports_ids, reports_masks = images.cuda(), reports_ids.cuda(), reports_masks.cuda()
        #     # print(images.shape)
        #     output = self.model(images, reports_ids, mode='train')
        #     loss = self.criterion(output, reports_ids, reports_masks)
        #     train_loss += loss.item()
        #     self.writer.add_scalar("data/Loss", loss.item(), batch_idx+len(self.train_dataloader)*(epoch-1))
        #     # To activate the tensorboard: tensorboard --logdir=runs --bind_all
        #     print_loss += loss.item()
        #     self.optimizer.zero_grad()
        #     loss.backward()
        #     torch.nn.utils.clip_grad_value_(self.model.parameters(), 0.1)
        #     self.optimizer.step()
        #     if batch_idx %5 == 0:
        #         print('Epoch: {}, Training Loss: {:.4f}'.format(epoch, print_loss/5))
        #         print_loss = 0
        # log = {'train_loss': train_loss / len(self.train_dataloader)}
        # print("Finish Epoch {} Training, Start Eval...".format(epoch))
        log = {}
        self.model.eval()
        # with torch.no_grad():
        #     val_gts, val_res = [], []
        #     for batch_idx, (images_id, images, reports_ids, reports_masks) in enumerate(self.val_dataloader):
        #         images, reports_ids, reports_masks = images.to(self.device), reports_ids.to(
        #             self.device), reports_masks.to(self.device)
        #         output = self.model(images, mode='sample')
        #         reports = self.model.module.tokenizer.decode_batch(output.cpu().numpy())
        #         ground_truths = self.model.module.tokenizer.decode_batch(reports_ids[:, 1:].cpu().numpy())
        #         val_res.extend(reports)
        #         val_gts.extend(ground_truths)
        #     val_met = self.metric_ftns({i: [gt] for i, gt in enumerate(val_gts)},
        #                                {i: [re] for i, re in enumerate(val_res)})
        #     log.update(**{'val_' + k: v for k, v in val_met.items()})
        # self.writer.add_scalar("data/b1/val", val_met['BLEU_1'], epoch)
        # self.writer.add_scalar("data/b2/val", val_met['BLEU_2'], epoch)
        # self.writer.add_scalar("data/b3/val", val_met['BLEU_3'], epoch)
        # self.writer.add_scalar("data/b4/val", val_met['BLEU_4'], epoch)
        # self.writer.add_scalar("data/met/val", val_met['METEOR'], epoch)
        # self.writer.add_scalar("data/rou/val", val_met['ROUGE_L'], epoch)
        # self.writer.add_scalar("data/cid/val", val_met['CIDER'], epoch)

        self.model.eval()
        with torch.no_grad():
            test_gts, test_res = [], []
            for data in tqdm(dataloader):
                loss = 0
                im_data_all, im_info_all, gt_boxes_all, num_boxes_all, targets, targets_masks = data
                im_data_all = im_data_all.squeeze()
                # gt_boxes = gt_boxes.squeeze(0)
                gt_boxes_all = gt_boxes_all[0]
                num_boxes_all = num_boxes_all.squeeze()
                
                # 筛选出有bbox的图像
                index = np.array(num_boxes_all!=0).flatten()
                # print(index)
                im_data_all = im_data_all[index]
                gt_boxes_all = gt_boxes_all[index] # [x, 1, 5]
                num_boxes_all = num_boxes_all[index]
                # print(im_data.shape)
                # print(gt_boxes)
                # print(num_boxes.shape)

                if args.cuda:
                    im_data_batch = im_data_all.cuda() # [b,...]
                    im_info_batch = im_info_all.cuda() # [1, 3]
                    num_boxes_batch = num_boxes_all.cuda()
                    gt_boxes_batch = gt_boxes_all.cuda()
                
                fasterRCNN.zero_grad()
                for k in range(len(im_data_batch)):
                    im_data = im_data_batch[k].unsqueeze(0)
                    im_info = im_info_batch # [1, 3]
                    gt_boxes = gt_boxes_batch[k].unsqueeze(0)
                    num_boxes = num_boxes_batch[k].unsqueeze(0)
                    
                    rois, cls_prob, bbox_pred, \
                    rpn_loss_cls, rpn_loss_box, \
                    RCNN_loss_cls, RCNN_loss_bbox, \
                    rois_label, att_feats, fc_feats = fasterRCNN(im_data, im_info, gt_boxes, num_boxes, return_feature_and_loss=True, Class_Semantic_embedding=test_semantic_embedding)
                
                    if k == 0: # 一个case只有一个融合了所有图像的特征
                        case_att_feats = att_feats.unsqueeze(0) # [1, 16, 2048]
                        case_fc_feats = fc_feats.unsqueeze(0) # [1, 2048]
                    else:
                        case_att_feats = torch.cat([att_feats.unsqueeze(0), case_att_feats],0)
                        case_fc_feats = torch.cat([fc_feats.unsqueeze(0), case_fc_feats],0)
                    
                # Report Generate ########################################

                reports_ids, reports_masks = targets.squeeze().unsqueeze(0).cuda(), targets_masks.squeeze().unsqueeze(0).cuda()
                output = model(case_att_feats.mean(0).unsqueeze(0),case_fc_feats.mean(0).unsqueeze(0), reports_ids, mode='sample')

                reports = self.model.tokenizer.decode_batch(output.cpu().numpy())
                print(reports)
                ground_truths = self.model.tokenizer.decode_batch(reports_ids[:, 1:].cpu().numpy())
                test_res.extend(reports)
                test_gts.extend(ground_truths)


            test_met = self.metric_ftns({i: [gt] for i, gt in enumerate(test_gts)},
                                        {i: [re] for i, re in enumerate(test_res)})
            log.update(**{'test_' + k: v for k, v in test_met.items()})
        self.writer.add_scalar("data/b1/test", test_met['BLEU_1'], epoch)
        self.writer.add_scalar("data/b2/test", test_met['BLEU_2'], epoch)
        self.writer.add_scalar("data/b3/test", test_met['BLEU_3'], epoch)
        self.writer.add_scalar("data/b4/test", test_met['BLEU_4'], epoch)
        self.writer.add_scalar("data/met/test", test_met['METEOR'], epoch)
        self.writer.add_scalar("data/rou/test", test_met['ROUGE_L'], epoch)
        self.writer.add_scalar("data/cid/test", test_met['CIDER'], epoch)

        self.lr_scheduler.step()
        self.writer.close()

        return log, test_gts, test_res
