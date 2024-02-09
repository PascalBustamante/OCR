import torch 
import torch.nn as nn
import torch.nn.functional as F
from bilinear_interpolation import bilinear_sampler

class SpatialTransformerNetwork:
    def __init__(self, input_shape, theta, num_steps):
        """
        :param input_shape: Input feature shape (channels, height, width)
        :param theta: Localization network output with shape [batch_size, num_steps, 6]
        :param num_steps: Num steps, it refers to word or lines we wish to localize
        """
        self.input_shape = input_shape
        self.theta = theta
        self.num_steps = num_steps

    def grid_generator(self):
        batch_sz = self.input_shape[0]
        H = self.input_shape[1]
        W = self.input_shape[2]

        # Reshape theta
        theta = self.theta.view(batch_sz * self.num_steps, 2, 3)

        # Generate grid
        grid_h = H
        grid_w = W
        X = torch.linspace(1.0, -1.0, grid_w)
        Y = torch.linspace(1.0, -1.0, grid_h)
        grid_xt, grid_yt = torch.meshgrid(X, Y)
        grid_xt = grid_xt.view(-1)
        grid_yt = grid_yt.view(-1)
        ones = torch.ones_like(grid_xt)
        samplig_grid = torch.stack([grid_xt, grid_yt, ones])
        samplig_grid = samplig_grid.unsqueeze(0).repeat(theta.size(0), 1, 1)
        samplig_grid = samplig_grid.unsqueeze(0)
        gen_grid = torch.matmul(theta, samplig_grid)
        gen_grid = gen_grid.view(batch_sz, self.num_steps, 2, grid_h, grid_w)
        gen_grid_x = gen_grid[:, :, 0, :, :]
        gen_grid_y = gen_grid[:, :, 1, :, :]

        return gen_grid_x, gen_grid_y

    def image_sampling(self, input):
        gen_grid_x, gen_grid_y = self.grid_generator()

        # Tile input
        input = input.unsqueeze(1).repeat(1, self.num_steps, 1, 1, 1)

        # Sample using bilinear interpolation
        output_feature_list = []
        for i in range(self.num_steps):
            output_feature_list.append(bilinear_sampler(input[:, i, :, :, :], gen_grid_x[:, i, :, :], gen_grid_y[:, i, :, :]))
        output_feature = torch.cat(output_feature_list, dim=1)
        return output_feature


if __name__ == "__main__":
    input_shape = (32, 600, 150)  # Example input shape
    num_steps = 1
    theta = torch.ones((32, num_steps, 6), dtype=torch.float32)  # Example theta tensor
    stn_obj = SpatialTransformerNetwork(input_shape, theta, num_steps)

    # Example input tensor
    input_data = torch.randn(32, 600, 150)  # Batch, Height, Width

    # Perform image sampling
    output_feature = stn_obj.image_sampling(input_data)
    print("Output feature shape:", output_feature.shape)