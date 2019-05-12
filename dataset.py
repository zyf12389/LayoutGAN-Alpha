from torch.utils import data
# zc:把from ./data改成from data
import sys,os
p = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(p)
sys.path.insert(0,p)
print(sys.path)
from data import ObjectCategories, RenderedScene, RenderedComposite

import random
import math
import torch
import cv2
import _pickle as pickle
import color_index


class Dataset():
    """
    Dataset for training/testing the layoutGAN network
    """

    def __init__(self, data_root_dir, data_dir, scene_indices=(0, 6400), num_per_epoch=1, seed=None):
        """
        Parameters
        ----------
        data_root_dir (String): root dir where all data lives
        data_dir (String): directory where this dataset lives (relative to data_root_dir)
        scene_indices (tuple[int, int]): list of indices of scenes (in data_dir) that are considered part of this set
        num_per_epoch (int): number of random variants of each scene that will be used per training epoch
        """
        self.data_root_dir = data_root_dir
        # self.data_dir = data_root_dir + '/' + data_dir
        self.data_dir = data_dir
        self.scene_indices = scene_indices
        self.num_per_epoch = num_per_epoch

        # Load up the map between SUNCG model IDs and category names
        # self.category_map = ObjectCategories(data_root_dir + '/suncg_data/ModelCategoryMapping.csv')
        # Also load up the list of coarse categories used in this particular dataset
        # self.categories = self.get_coarse_categories()
        # Build a reverse map from category to index
        # self.cat_to_index = {self.categories[i]:i for i in range(len(self.categories))}
        self.seed = seed

    def __len__(self):
        return (self.scene_indices[1] - self.scene_indices[0]) * self.num_per_epoch

    def __getitem__(self, index):
        if self.seed:
            random.seed(self.seed)

        i = int(index + self.scene_indices[0] / self.num_per_epoch)
        scene = RenderedScene(i, self.data_dir, self.data_root_dir)
        composite = scene.create_composite()

        num_categories = len(scene.categories)
        # Flip a coin for whether we're going remove objects or treat this as a complete scene

        num_objects = len(scene.object_nodes)
        object_nodes = scene.object_nodes

        # 理解：p_existing是输入的p
        # 一个场景的num_categories数量固定，每种标签至少有一个物体，但可能不止一个
        # 因此，列数是num_categories，而行数暂时先多填了5个
        # 疑问：1.每个场景的num_categories不一样，形成的one-hot vector长度不一致，是否
        # 需要改成固定长度的num_categories
        p_existing = torch.zeros(num_objects, num_categories)

        for i in range(num_objects):
            existing_categories = torch.zeros(num_categories)
            node = scene.object_nodes[i]
            composite.add_node(node)

            existing_categories[node["category"]] = 1
            p_existing[i] = existing_categories

        coordinates_existing = torch.zeros(num_objects, 4)

        wall = scene.wall
        wall_mask = wall.clone()
        index_nonzero = torch.nonzero(wall_mask)
        xmin_scene, ymin_scene = index_nonzero[0][0], index_nonzero[0][1]
        xmax_scene, ymax_scene = index_nonzero[index_nonzero.shape[0] - 1][0], \
                                 index_nonzero[index_nonzero.shape[0] - 1][1]

        for i in range(num_objects):
            #existing_coordinates = torch.zeros(4)
            node = object_nodes[i]
            xmin, _, ymin, _ = node["bbox_min"]
            xmax, _, ymax, _ = node["bbox_max"]

            # TO DO
            # 1 scale coordinates(need to pre-define the height and width of map)
            # 获取房间俯视图的xmin_scene,xmax_scene,ymin_scene,ymax_scene
            # 将坐标归一化到0-1之间(房间的边缘是0和1)
            xmin = (xmin - xmin_scene) / (xmax_scene - xmin_scene).double()
            xmax = (xmax - xmin_scene) / (xmax_scene - xmin_scene).double()
            ymin = (ymin - ymin_scene) / (ymax_scene - ymin_scene).double()
            ymax = (ymax - ymin_scene) / (ymax_scene - ymin_scene).double()
            existing_coordinates = torch.Tensor((xmin, ymin, xmax, ymax))

            coordinates_existing[i] = existing_coordinates
        existing_object = torch.cat((p_existing, coordinates_existing), 1)
        non_existing = torch.zeros(num_categories + 5 - num_objects, num_categories + 4)
        output = torch.cat((existing_object, non_existing), 0)

        #print("output shape=",output.shape)
        return output
