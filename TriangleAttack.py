import json
import torch
import os
import argparse
import random
from foolbox.distances import l2
import numpy as np
from PIL import Image
from foolbox.attacks.blended_noise import LinearSearchBlendedUniformNoiseAttack
from foolbox.attacks.base import MinimizationAttack, get_criterion
import sys
import torch_dct
import time

class TA:
    def __init__(self, model, input_device, side_length=256, seed=20, max_queries=1000, ratio_mask=0.1, dim_num=1, max_iter_num_in_2d=2, init_theta=2, 
                 init_alpha=np.pi/2, plus_learning_rate=0.1, minus_learning_rate=0.005, half_range=0.1):
        self.net = model
        self.device = input_device
        self.side_length = side_length
        self.seed = seed
        self.max_queries = max_queries
        self.ratio_mask = ratio_mask
        self.dim_num = dim_num
        self.max_iter_num_in_2d = max_iter_num_in_2d,
        self.init_theta = init_theta
        self.init_alpha = init_alpha
        self.plus_learning_rate = plus_learning_rate
        self.minus_learning_rate = minus_learning_rate
        self.half_range = half_range


    def get_label(self, logit):
        return torch.argmax(logit, dim=1)

        
    # initialize an adversarial example with uniform noise
    def get_x_adv(self, x_o: torch.Tensor, label: torch.Tensor, model) -> torch.Tensor:
        criterion = get_criterion(label)
        init_attack: MinimizationAttack = LinearSearchBlendedUniformNoiseAttack(steps=100)
        x_adv = init_attack.run(model, x_o, criterion)
        return x_adv


    # coompute the difference
    def get_difference(self, x_o: torch.Tensor, x_adv: torch.Tensor) -> torch.Tensor:
        difference = x_adv - x_o
        if torch.norm(difference, p=2) == 0:
            raise Exception('difference is zero vector!')
            return difference
        return difference


    def rotate_in_2d(self, x_o2x_adv: torch.Tensor, direction: torch.Tensor, theta: float = np.pi / 8) -> torch.Tensor:
        alpha = torch.sum(x_o2x_adv * direction) / torch.sum(x_o2x_adv * x_o2x_adv)
        orthogonal = direction - alpha * x_o2x_adv
        direction_theta = x_o2x_adv * np.cos(theta) + torch.norm(x_o2x_adv, p=2) / torch.norm(orthogonal,
                                                                                            p=2) * orthogonal * np.sin(
            theta)
        direction_theta = direction_theta / torch.norm(direction_theta) * torch.norm(x_o2x_adv)
        return direction_theta


    # obtain the mask in the low frequency
    def get_orthogonal_1d_in_subspace(self,x_o2x_adv: torch.Tensor, n, ratio_size_mask=0.3, if_left=1) -> torch.Tensor:
        random.seed(time.time())
        zero_mask = torch.zeros(size=[self.side_length, self.side_length], device=self.device)
        size_mask = int(self.side_length * ratio_size_mask)
        if if_left:
            zero_mask[:size_mask, :size_mask] = 1

        else:
            zero_mask[-size_mask:, -size_mask:] = 1

        to_choose = torch.where(zero_mask == 1)
        x = to_choose[0]
        y = to_choose[1]

        select = np.random.choice(len(x), size=n, replace=False)
        mask1 = torch.zeros_like(zero_mask)
        mask1[x[select], y[select]] = 1
        mask1 = mask1.reshape(-1, self.side_length, self.side_length)

        select = np.random.choice(len(x), size=n, replace=False)
        mask2 = torch.zeros_like(zero_mask)
        mask2[x[select], y[select]] = 1
        mask2 = mask2.reshape(-1, self.side_length, self.side_length)

        select = np.random.choice(len(x), size=n, replace=False)
        mask3 = torch.zeros_like(zero_mask)
        mask3[x[select], y[select]] = 1
        mask3 = mask3.reshape(-1, self.side_length, self.side_length)

        mask = torch.cat([mask1, mask2, mask3], dim=0).expand([1, 3, self.side_length, self.side_length])
        mask *= torch.randn_like(mask, device=self.device)
        direction = self.rotate_in_2d(x_o2x_adv, mask, theta=np.pi / 2)
        return direction / torch.norm(direction, p=2) * torch.norm(x_o2x_adv, p=2), mask


    # compute the best adversarial example in the surface
    def get_x_hat_in_2d(self, x_o: torch.Tensor, x_adv: torch.Tensor, axis_unit1: torch.Tensor, axis_unit2: torch.Tensor,
                        queries, original_label, plus_learning_rate=0.01,minus_learning_rate=0.0005,half_range=0.1, init_alpha = np.pi/2):
        if not hasattr(self.get_x_hat_in_2d, 'alpha'):
            self.get_x_hat_in_2d_alpha = init_alpha
        upper = np.pi / 2 + half_range
        lower = np.pi / 2 - half_range

        d = torch.norm(x_adv - x_o, p=2)

        theta = max(np.pi - 2 * self.get_x_hat_in_2d_alpha, 0) + min(np.pi / 16, self.get_x_hat_in_2d_alpha / 2)
        x_hat = torch_dct.idct_2d(x_adv)
        right_theta = np.pi - self.get_x_hat_in_2d_alpha
        x = x_o + d * (axis_unit1 * np.cos(theta) + axis_unit2 * np.sin(theta)) / np.sin(self.get_x_hat_in_2d_alpha) * np.sin(
            self.get_x_hat_in_2d_alpha + theta)
        x = torch_dct.idct_2d(x)
        self.get_x_hat_in_2d_total += 1
        self.get_x_hat_in_2d_clamp += torch.sum(x > 1) + torch.sum(x < 0)
        x = torch.clamp(x, 0, 1)
        label = self.get_label(self.net(x))
        queries += 1
        if label != original_label:
            x_hat = x
            left_theta = theta
            flag = 1
        else:

            self.get_x_hat_in_2d_alpha -= minus_learning_rate
            self.get_x_hat_in_2d_alpha = max(lower, self.get_x_hat_in_2d_alpha)
            theta = max(theta, np.pi - 2 * self.get_x_hat_in_2d_alpha + np.pi / 64)

            x = x_o + d * (axis_unit1 * np.cos(theta) - axis_unit2 * np.sin(theta)) / np.sin(
                self.get_x_hat_in_2d_alpha) * np.sin(
                self.get_x_hat_in_2d_alpha + theta)  # * mask
            x = torch_dct.idct_2d(x)
            self.get_x_hat_in_2d_total += 1
            self.get_x_hat_in_2d_clamp += torch.sum(x > 1) + torch.sum(x < 0)
            x = torch.clamp(x, 0, 1)
            label = self.get_label(self.net(x))
            queries += 1
            if label != original_label:
                x_hat = x
                left_theta = theta
                flag = -1
            else:
                self.get_x_hat_in_2d_alpha -= minus_learning_rate
                self.get_x_hat_in_2d_alpha = max(self.get_x_hat_in_2d_alpha, lower)
                return x_hat, queries, False

        # binary search for beta
        theta = (left_theta + right_theta) / 2
        for i in range(self.max_iter_num_in_2d[0]):
            x = x_o + d * (axis_unit1 * np.cos(theta) + flag * axis_unit2 * np.sin(theta)) / np.sin(
                self.get_x_hat_in_2d_alpha) * np.sin(
                self.get_x_hat_in_2d_alpha + theta)
            x = torch_dct.idct_2d(x)
            self.get_x_hat_in_2d_total += 1
            self.get_x_hat_in_2d_clamp += torch.sum(x > 1) + torch.sum(x < 0)
            x = torch.clamp(x, 0, 1)
            label = self.get_label(self.net(x))
            queries += 1
            if label != original_label:
                left_theta = theta
                x_hat = x
                self.get_x_hat_in_2d_alpha += plus_learning_rate
                return x_hat, queries, True
            else:

                self.get_x_hat_in_2d_alpha -= minus_learning_rate
                self.get_x_hat_in_2d_alpha = max(lower, self.get_x_hat_in_2d_alpha)
                theta = max(theta, np.pi - 2 * self.get_x_hat_in_2d_alpha + np.pi / 64)

                flag = -flag
                x = x_o + d * (axis_unit1 * np.cos(theta) + flag * axis_unit2 * np.sin(theta)) / np.sin(
                    self.get_x_hat_in_2d_alpha) * np.sin(
                    self.get_x_hat_in_2d_alpha + theta)
                x = torch_dct.idct_2d(x)
                self.get_x_hat_in_2d_total += 1
                self.get_x_hat_in_2d_clamp += torch.sum(x > 1) + torch.sum(x < 0)
                x = torch.clamp(x, 0, 1)
                label = self.get_label(self.net(x))
                queries += 1
                if label != original_label:
                    left_theta = theta
                    x_hat = x
                    self.get_x_hat_in_2d_alpha += plus_learning_rate
                    self.get_x_hat_in_2d_alpha = min(upper, self.get_x_hat_in_2d_alpha)
                    return x_hat, queries, True
                else:
                    self.get_x_hat_in_2d_alpha -= minus_learning_rate
                    self.get_x_hat_in_2d_alpha = max(lower, self.get_x_hat_in_2d_alpha)
                    left_theta = max(np.pi - 2 * self.get_x_hat_in_2d_alpha, 0) + min(np.pi / 16, self.get_x_hat_in_2d_alpha / 2)
                    right_theta = theta
            theta = (left_theta + right_theta) / 2
        self.get_x_hat_in_2d_alpha += plus_learning_rate
        self.get_x_hat_in_2d_alpha = min(upper, self.get_x_hat_in_2d_alpha)
        return x_hat, queries, True


    def get_x_hat_arbitary(self, x_o: torch.Tensor, original_label, init_x=None,dim_num=5):
        if self.get_label(self.net(x_o)) != original_label:
            return x_o, 1001, [[0, 0.], [1001, 0.]]
        if init_x is None:
            x_adv = self.get_x_adv(x_o, original_label, self.net)
        else:
            x_adv = init_x
        x_hat = x_adv
        queries = 0.
        dist = torch.norm(x_o - x_adv)
        intermediate = []
        intermediate.append([0, dist.item(), self.get_x_hat_in_2d_alpha])

        while queries < self.max_queries :

            x_o2x_adv = torch_dct.dct_2d(self.get_difference(x_o, x_adv))
            axis_unit1 = x_o2x_adv / torch.norm(x_o2x_adv)
            direction, mask = self.get_orthogonal_1d_in_subspace(x_o2x_adv, dim_num, self.ratio_mask, self.dim_num)
            axis_unit2 = direction / torch.norm(direction)
            x_hat, queries, changed = self.get_x_hat_in_2d(torch_dct.dct_2d(x_o), torch_dct.dct_2d(x_adv), axis_unit1,
                                                    axis_unit2, queries, original_label ,plus_learning_rate=self.plus_learning_rate,minus_learning_rate=self.minus_learning_rate,half_range=self.half_range, init_alpha=self.init_alpha)
            x_adv = x_hat

            dist = torch.norm(x_hat - x_o)
            intermediate.append([queries, dist.item(), self.get_x_hat_in_2d_alpha])
            if queries >= self.max_queries:
                break
        return x_hat, queries, intermediate

    def attack(self, dataloader):
        images = dataloader.dataset.__getimages__()
        labels = dataloader.dataset.__getlabels__()

        labels = torch.argmax(labels, dim=1)
        
        images = images.to(self.device)
        labels = labels.to(self.device)
        # self.net = self.net.to(self.device)

        print(labels.shape)
        # Assuming images is your input tensor
        if len(images.shape) == 3:  # If it's [channels, height, width]
            images = images.unsqueeze(0)  # Add a batch dimension

        if images.shape[1] == 1:  # If there's only one channel
            images = images.repeat(1, 3, 1, 1)  # Repeat the channel dimension

        # Now pass images to the model
        # output = self.net(images)

        # print(output.shape)

        x_adv_list = torch.zeros_like(images, device=self.device)
        queries = []
        intermediates = []
        init_attack: MinimizationAttack = LinearSearchBlendedUniformNoiseAttack(steps=50)
        criterion = get_criterion(labels.long())

        if os.path.isfile('best_advs.pt'):
            print("loading best_advs")
            best_advs = torch.load('best_advs.pt')
        else:
            print("calculating best_advs")
            best_advs = init_attack.run(self.net, images, criterion, early_stop=None)
            torch.save(best_advs, 'best_advs.pt')

        print("\n", best_advs[0], "\n")

        best_advs = best_advs.to(self.device)

        max_length = 0
        acc = [0., 0., 0.]
        for i, [input, label] in enumerate(dataloader):
            input = input.to(self.device)
            label = label.to(self.device)

            print('[{}/{}]:'.format(i + 1, len(dataloader.dataset)), end='')
            global probability
            probability = np.ones(input.shape[1] * input.shape[2])
            global is_visited_1d
            is_visited_1d = torch.zeros(input.shape[0] * input.shape[1] * input.shape[2])
            global selected_h
            global selected_w
            selected_h = input.shape[1]
            selected_w = input.shape[2]
            self.get_x_hat_in_2d_alpha = np.pi / 2

            self.get_x_hat_in_2d_total = 0
            self.get_x_hat_in_2d_clamp = 0

            print(input.shape)
            print(best_advs[i][np.newaxis, :, :, :].shape)

            # ran until here [np.newaxis, :, :, :]
            print(self.max_iter_num_in_2d)
            x_adv, q, intermediate = self.get_x_hat_arbitary(input, torch.argmax(label).to(self.device),
                                                        init_x=best_advs[i].unsqueeze(0), dim_num=self.dim_num)
            x_adv_list[i] = x_adv[0].to(self.device)
            diff = torch.norm(x_adv[0].to(self.device) - input, p=2) / (self.side_length * np.sqrt(3))
            if diff <= 0.1:
                acc[0] += 1
            if diff <= 0.05:
                acc[1] += 1
            if diff <= 0.01:
                acc[2] += 1
            print("Top-1 Acc:{} Top-2 Acc:{} Top-3 Acc:{}".format(acc[0] / (i + 1), acc[1] / (i + 1),
                                                                                     acc[2] / (i + 1)))
            queries.append(q)
            intermediates.append(intermediate)
            if max_length < len(intermediate):
                max_length = len(intermediate)
        queries = np.array(queries)
        return x_adv_list, queries, intermediates, max_length
    

    
#  __name__ == "__main__":

#      # images, labels = samples(fmodel, dataset="imagenet", batchsize=args.n_images)
#     images, labels, selected_paths = read_imagenet_data_specify(args, device)
#     print("{} images loaded with the following labels: {}".format(len(images), labels))

#     ###############################
#     print("Attack !")
#     time_start = time.time()

#     ta_model = attack.TA(fmodel, input_device=device)
#     my_advs, q_list, my_intermediates, max_length = ta_model.attack(args,images, labels)
#     print('TA Attack Done')
#     print("{:.2f} s to run".format(time.time() - time_start))
#     print("Results")

#     my_labels_advs = fmodel(my_advs).argmax(1)
#     my_advs_l2 = l2(images, my_advs)

#     for image_i in range(len(images)):
#         print("My Adversarial Image {}:".format(image_i))
#         label_o = int(labels[image_i])
#         label_adv = int(my_labels_advs[image_i])
#         print("\t- l2 = {}".format(my_advs_l2[image_i]))
#         print("\t- {} queries\n".format(q_list[image_i]))
#     save_results(args,my_intermediates, len(images))

    