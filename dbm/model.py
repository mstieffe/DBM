import torch
import torch.nn as nn
from dbm.util import compute_same_padding

class NoScaleDropout(nn.Module):
    """
        Dropout without rescaling and variable dropout rates.
    """

    def __init__(self, rate_max) -> None:
        super().__init__()
        self.rate_max = rate_max

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.rate_max == 0:
            return x
        else:
            rate = torch.empty(1, device=x.device).uniform_(0, self.rate_max)
            N, n_atoms, *rest = x.shape
            mask = torch.empty((N, n_atoms), device=x.device).bernoulli_(1 - rate)
            return x * mask[:, :, None, None, None]

def _facify(n, fac):
    """
    Function to divide n by fac and cast the result to an integer.

    Args:
        n (float): the numerator.
        fac (float): the denominator.

    Returns:
        int: the result of n divided by fac, cast to an integer.
    """
    return int(n // fac)

def _sn_to_specnorm(sn: int):
    """
    Function to create a Spectral Normalization layer with the specified number of power
        iterations (sn).

    Args:
        sn (int): the number of power iterations to use for computing the spectral norm.

    Returns:
        function: a function that applies Spectral Normalization with the specified number of
            power iterations to a given module.
    """
    if sn > 0:

        def specnorm(module):
            return nn.utils.spectral_norm(module, n_power_iterations=sn)

    else:

        def specnorm(module, **kw):
            return module

    return specnorm

class AtomGen_tiny(nn.Module):

    """
    A class to define a generator network for generating 3D atoms.

    Attributes:
    - z_dim (int): dimension of the noise vector z
    - in_channels (int): number of input channels
    - start_channels (int): number of channels in the first layer
    - fac (float): factor by which to increase the number of channels in each layer
    - sn (int): spectral normalization, if sn > 0, will be applied
    - device (str): device on which to run the model

    Public Methods:
    - forward(z, l, c): runs a forward pass of the generator network with the given input tensors
    """

    def __init__(
        self,
        z_dim,
        in_channels,
        start_channels,
        fac=1,
        sn: int = 0,
        device=None,
    ):
        """
         Initializes an instance of the AtomGen_tiny class.

         Args:
         - z_dim (int): dimension of the noise vector z
         - in_channels (int): number of input channels
         - start_channels (int): number of channels in the first layer
         - fac (int): factor by which to increase the number of channels in each layer
         - sn (int): spectral normalization, if sn > 0, will be applied
         - device (str): device on which to run the model
         """

        super().__init__()
        specnorm = _sn_to_specnorm(sn)
        embed_condition_blocks = [
            specnorm(
                nn.Conv3d(
                    in_channels=in_channels,
                    out_channels=_facify(start_channels, fac),
                    kernel_size=5,
                    stride=1,
                    padding=2,
                )
            ),
            nn.GroupNorm(1, num_channels=_facify(start_channels, fac)),
            nn.LeakyReLU(),
            specnorm(
                nn.Conv3d(
                    in_channels=start_channels,
                    out_channels=_facify(start_channels, fac),
                    kernel_size=3,
                    stride=1,
                    padding=compute_same_padding(3, 1, 1)
                )
            ),
            nn.GroupNorm(1, num_channels=_facify(start_channels, fac)),
            nn.LeakyReLU(),

        ]
        self.embed_condition = nn.Sequential(*tuple(embed_condition_blocks)).to(device=device)

        downsample_cond_block = [
            specnorm(
                nn.Conv3d(
                    in_channels=start_channels,
                    out_channels=_facify(start_channels*2, fac),
                    kernel_size=3,
                    stride=2,
                    padding=compute_same_padding(3, 2, 1)
                )
            ),
            nn.GroupNorm(1, num_channels=_facify(start_channels*2, fac)),
            nn.LeakyReLU(),

        ]
        self.downsample_cond = nn.Sequential(*tuple(downsample_cond_block)).to(device=device)

        self.embed_noise_label = EmbedNoise(z_dim, _facify(start_channels*2, fac), sn=sn)

        combined_block = [
            specnorm(
                nn.Conv3d(
                    in_channels=start_channels*4,
                    out_channels=_facify(start_channels*2, fac),
                    kernel_size=3,
                    stride=1,
                    padding=compute_same_padding(3, 1, 1),
                )
            ),
            nn.GroupNorm(1, num_channels=_facify(start_channels*2, fac)),
            nn.LeakyReLU(),

        ]
        self.combined = nn.Sequential(*tuple(combined_block)).to(device=device)

        deconv_block = [
            specnorm(
                nn.Conv3d(
                    in_channels=start_channels*2,
                    out_channels=_facify(start_channels, fac),
                    kernel_size=3,
                    stride=1,
                    padding=compute_same_padding(3, 1, 1),
                )
            ),
            nn.GroupNorm(1, num_channels=_facify(start_channels, fac)),
            nn.LeakyReLU(),

        ]
        self.deconv = nn.Sequential(*tuple(deconv_block)).to(device=device)

        to_image_blocks = [
            specnorm(
                nn.Conv3d(
                    in_channels=start_channels*2,
                    out_channels=_facify(start_channels, fac),
                    kernel_size=3,
                    stride=1,
                    padding=compute_same_padding(3, 1, 1),
                )
            ),
            nn.GroupNorm(1, num_channels=_facify(start_channels, fac)),
            nn.LeakyReLU(),
            specnorm(
                nn.Conv3d(
                    in_channels=start_channels,
                    out_channels=_facify(start_channels / 2, fac),
                    kernel_size=3,
                    stride=1,
                    padding=compute_same_padding(3, 1, 1),
                )
            ),
            nn.GroupNorm(1, num_channels=_facify(start_channels/2, fac)),
            nn.LeakyReLU(),
            specnorm(nn.Conv3d(_facify(start_channels/2, fac), 1, kernel_size=1, stride=1)),
            nn.Sigmoid(),
        ]
        self.to_image = nn.Sequential(*tuple(to_image_blocks)).to(device=device)

    def forward(self, z, l, c):
        z_l = torch.cat((z, l), dim=1)
        embedded_c = self.embed_condition(c)
        down = self.downsample_cond(embedded_c)
        embedded_z_l = self.embed_noise_label(z_l)
        out = torch.cat((embedded_z_l, down), dim=1)
        out = self.combined(out)
        out = out.repeat(1, 1, 2, 2, 2)
        out = self.deconv(out)
        out = torch.cat((out, embedded_c), dim=1)
        out = self.to_image(out)

        return out

class AtomGen(nn.Module):
    """
    A class to define a generator network for generating 3D atoms. (more layers included than AtomGen_tiny

    Attributes:
    - z_dim (int): dimension of the noise vector z
    - in_channels (int): number of input channels
    - start_channels (int): number of channels in the first layer
    - fac (float): factor by which to increase the number of channels in each layer
    - sn (int): spectral normalization, if sn > 0, will be applied
    - device (str): device on which to run the model

    Public Methods:
    - forward(z, l, c): runs a forward pass of the generator network with the given input tensors
    """
    def __init__(
        self,
        z_dim,
        in_channels,
        start_channels,
        fac=1,
        sn: int = 0,
        device=None,
    ):
        super().__init__()
        specnorm = _sn_to_specnorm(sn)
        embed_condition_blocks = [
            specnorm(
                nn.Conv3d(
                    in_channels=in_channels,
                    out_channels=_facify(start_channels, fac),
                    kernel_size=5,
                    stride=1,
                    padding=2,
                )
            ),
            nn.GroupNorm(1, num_channels=_facify(start_channels, fac)),
            nn.LeakyReLU(),
            Residual3DConvBlock(
                _facify(start_channels, fac), _facify(start_channels, fac), kernel_size=3, stride=1, sn=sn
            ),
        ]
        self.embed_condition = nn.Sequential(*tuple(embed_condition_blocks)).to(
            device=device
        )
        self.downsample_cond = Residual3DConvBlock(
            _facify(start_channels, fac),
            n_filters=_facify(start_channels, fac),
            kernel_size=3,
            stride=2,
            sn=sn,
        )

        self.embed_noise_label = EmbedNoise(z_dim, _facify(start_channels, fac), sn=sn)
        self.combined = GeneratorCombined3Block(
            _facify(start_channels*2, fac), _facify(start_channels, fac), sn=sn
        )

        to_image_blocks = [
            Residual3DConvBlock(
                _facify(start_channels*2, fac), _facify(start_channels/2, fac), 3, 1, trans=True, sn=sn
            ),
            specnorm(nn.Conv3d(_facify(start_channels/2, fac), 1, kernel_size=1, stride=1)),
            nn.Sigmoid(),
        ]
        self.to_image = nn.Sequential(*tuple(to_image_blocks)).to(device=device)

    def forward(self, z, l, c):
        z_l = torch.cat((z, l), dim=1)

        embedded_c = self.embed_condition(c)
        down = self.downsample_cond(embedded_c)

        embedded_z_l = self.embed_noise_label(z_l)

        out = self.combined(embedded_z_l, down)

        out = torch.cat((out, embedded_c), dim=1)
        out = self.to_image(out)

        return out

class AtomCrit_tiny(nn.Module):
    def __init__(
        self, in_channels, start_channels, fac=1, sn: int = 0, device=None
    ):
        """
        Constructor for AtomCrit_tiny class.

        Arguments:
        in_channels -- Number of input channels.
        start_channels -- Number of output channels for first layer.
        fac -- Factor to multiply the number of output channels of each layer. (default 1)
        sn -- Spectral normalization factor. (default 0)
        device -- Device to run the network on. (default None)

        Attributes:
        step1 -- First convolutional layer with LeakyReLU activation.
        step2 -- Second convolutional layer with GroupNorm and LeakyReLU activation.
        step3 -- Third convolutional layer with GroupNorm and LeakyReLU activation.
        step4 -- Fourth convolutional layer with GroupNorm and LeakyReLU activation.
        step5 -- Fifth convolutional layer with GroupNorm and LeakyReLU activation.
        step6 -- Sixth convolutional layer with GroupNorm and LeakyReLU activation.
        step7 -- Seventh convolutional layer with GroupNorm and LeakyReLU activation.
        step8 -- Eighth convolutional layer with GroupNorm and LeakyReLU activation.
        step9 -- Ninth convolutional layer with GroupNorm and LeakyReLU activation.
        step10 -- Tenth convolutional layer with GroupNorm and LeakyReLU activation.
        step11 -- Eleventh convolutional layer with GroupNorm and LeakyReLU activation.
        to_critic_value -- Final linear layer to get the output.
        """
        super().__init__()
        specnorm = _sn_to_specnorm(sn)
        self.step1 = specnorm(
            nn.Conv3d(
                in_channels=in_channels,
                out_channels=_facify(start_channels, fac),
                kernel_size=5,
                stride=1,
                padding=compute_same_padding(5, 1, 1),
            )
        )
        self.step2 = nn.LeakyReLU()

        self.step3 = specnorm(
            nn.Conv3d(
                in_channels=_facify(start_channels, fac),
                out_channels=_facify(start_channels*2, fac),
                kernel_size=3,
                stride=2,
                padding=compute_same_padding(3, 1, 1),
            )
        )
        self.step4 = nn.GroupNorm(1, _facify(start_channels*2, fac))
        self.step5 = nn.LeakyReLU()

        self.step6 = specnorm(
            nn.Conv3d(
                in_channels=_facify(start_channels*2, fac),
                out_channels=_facify(start_channels*2, fac),
                kernel_size=3,
                stride=1,
                padding=compute_same_padding(3, 1, 1),
            )
        )
        self.step7 = nn.GroupNorm(1, _facify(start_channels*2, fac))
        self.step8 = nn.LeakyReLU()

        self.step9 = specnorm(
            nn.Conv3d(
                in_channels=_facify(start_channels*2, fac),
                out_channels=_facify(start_channels, fac),
                kernel_size=4,
                stride=1,
                padding=0,
            )
        )
        self.step10 = nn.GroupNorm(1, _facify(start_channels, fac))
        self.step11 = nn.LeakyReLU()

        self.to_critic_value = specnorm(
            nn.Linear(
                in_features=_facify(start_channels, fac), out_features=1
            )
        )

    def forward(self, inputs):
        out = self.step1(inputs)
        out = self.step2(out)
        out = self.step3(out)
        out = self.step4(out)
        out = self.step5(out)
        out = self.step6(out)
        out = self.step7(out)
        out = self.step8(out)
        out = self.step9(out)
        out = self.step10(out)
        out = self.step11(out)
        out = torch.flatten(out, start_dim=1, end_dim=-1)
        out = self.to_critic_value(out)
        return out

class AtomCrit(nn.Module):
    def __init__(
        self, in_channels, start_channels, fac=1, sn: int = 0, device=None
    ):
        """
        Constructor for AtomCrit class.

        Arguments:
        in_channels -- Number of input channels.
        start_channels -- Number of output channels for first layer.
        fac -- Factor to multiply the number of output channels of each layer. (default 1)
        sn -- Spectral normalization factor. (default 0)
        device -- Device to run the network on. (default None)

        Attributes:
        step1 -- First convolutional layer with LeakyReLU activation.
        step2 -- Second convolutional layer with GroupNorm and LeakyReLU activation.
        step3 -- Third convolutional layer with GroupNorm and LeakyReLU activation.
        step4 -- Fourth convolutional layer with GroupNorm and LeakyReLU activation.
        step5 -- Fifth convolutional layer with GroupNorm and LeakyReLU activation.
        step6 -- Sixth convolutional layer with GroupNorm and LeakyReLU activation.
        step7 -- Seventh convolutional layer with GroupNorm and LeakyReLU activation.
        step8 -- Eighth convolutional layer with GroupNorm and LeakyReLU activation.
        step9 -- Ninth convolutional layer with GroupNorm and LeakyReLU activation.
        step10 -- Tenth convolutional layer with GroupNorm and LeakyReLU activation.
        step11 -- Eleventh convolutional layer with GroupNorm and LeakyReLU activation.
        to_critic_value -- Final linear layer to get the output.
        """
        super().__init__()
        specnorm = _sn_to_specnorm(sn)
        self.step1 = specnorm(
            nn.Conv3d(
                in_channels=in_channels,
                out_channels=_facify(start_channels, fac),
                kernel_size=5,
                stride=1,
                padding=compute_same_padding(5, 1, 1),
            )
        )
        self.step2 = nn.LeakyReLU()

        self.step4 = Residual3DConvBlock(
            in_channels=_facify(start_channels, fac),
            n_filters=_facify(start_channels*2, fac),
            kernel_size=3,
            stride=2,
            sn=sn,
            device=device,
        )

        self.step6 = specnorm(
            nn.Conv3d(
                in_channels=_facify(start_channels*2, fac),
                out_channels=_facify(start_channels*2, fac),
                kernel_size=3,
                stride=1,
                padding=compute_same_padding(3, 1, 1),
            )
        )
        self.step7 = nn.GroupNorm(1, _facify(start_channels*2, fac))
        self.step8 = nn.LeakyReLU()

        self.step9 = specnorm(
            nn.Conv3d(
                in_channels=_facify(start_channels*2, fac),
                out_channels=_facify(start_channels, fac),
                kernel_size=4,
                stride=1,
                padding=0,
            )
        )
        self.step10 = nn.GroupNorm(1, _facify(start_channels, fac))
        self.step11 = nn.LeakyReLU()

        self.to_critic_value = specnorm(
            nn.Linear(
                in_features=_facify(start_channels, fac), out_features=1
            )
        )

    def forward(self, inputs):
        out = self.step1(inputs)
        out = self.step2(out)
        #out = self.step3(out)
        out = self.step4(out)
        #out = self.step5(out)
        out = self.step6(out)
        out = self.step7(out)
        out = self.step8(out)
        out = self.step9(out)
        out = self.step10(out)
        out = self.step11(out)
        out = torch.flatten(out, start_dim=1, end_dim=-1)
        out = self.to_critic_value(out)
        return out


class Residual3DConvBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        n_filters,
        kernel_size,
        stride,
        trans=False,
        sn: int = 0,
        device=None,
    ):
        super(Residual3DConvBlock, self).__init__()
        specnorm = _sn_to_specnorm(sn)
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.trans = trans

        if self.stride != 1:
            self.downsample = nn.Sequential(
                nn.MaxPool3d(kernel_size=2, stride=2),
                specnorm(
                    nn.Conv3d(
                        in_channels,
                        out_channels=self.n_filters,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                    )
                ),
            )
        elif self.trans:
            self.downsample = nn.Sequential(
                specnorm(
                    nn.Conv3d(
                        in_channels,
                        out_channels=self.n_filters,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                    )
                )
            )
        else:
            self.downsample = nn.Identity()
        self.downsample = self.downsample.to(device=device)

        same_padding = compute_same_padding(self.kernel_size, self.stride, dilation=1)
        block_elements = [
            specnorm(
                nn.Conv3d(
                    in_channels=in_channels,
                    out_channels=self.n_filters,
                    kernel_size=self.kernel_size,
                    stride=self.stride,
                    padding=same_padding,
                )
            ),
            nn.GroupNorm(1, num_channels=self.n_filters),
            nn.LeakyReLU(),
            specnorm(
                nn.Conv3d(
                    in_channels=self.n_filters,
                    out_channels=self.n_filters,
                    kernel_size=self.kernel_size,
                    stride=1,
                    padding=same_padding,
                )
            ),
            nn.GroupNorm(1, num_channels=self.n_filters),
        ]
        self.block = nn.Sequential(*tuple(block_elements)).to(device=device)
        self.nonlin = nn.LeakyReLU()

    def forward(self, inputs):
        out = self.block(inputs)
        downsampled = self.downsample(inputs)
        result = 0.5 * (out + downsampled)
        result = self.nonlin(result)
        return result



class Residual3DConvBlock_drop(nn.Module):
    def __init__(
        self,
        in_channels,
        n_filters,
        kernel_size,
        stride,
        trans=False,
        sn: int = 0,
        device=None,
    ):
        super(Residual3DConvBlock_drop, self).__init__()
        specnorm = _sn_to_specnorm(sn)
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.trans = trans

        if self.stride != 1:
            self.downsample = nn.Sequential(
                nn.MaxPool3d(kernel_size=2, stride=2),
                specnorm(
                    nn.Conv3d(
                        in_channels,
                        out_channels=self.n_filters,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                    )
                ),
            )
        elif self.trans:
            self.downsample = nn.Sequential(
                specnorm(
                    nn.Conv3d(
                        in_channels,
                        out_channels=self.n_filters,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                    )
                )
            )
        else:
            self.downsample = nn.Identity()
        self.downsample = self.downsample.to(device=device)

        same_padding = compute_same_padding(self.kernel_size, self.stride, dilation=1)
        block_elements = [
            specnorm(
                nn.Conv3d(
                    in_channels=in_channels,
                    out_channels=self.n_filters,
                    kernel_size=self.kernel_size,
                    stride=self.stride,
                    padding=same_padding,
                )
            ),
            nn.Dropout3d(),
            nn.GroupNorm(1, num_channels=self.n_filters),
            nn.LeakyReLU(),
            specnorm(
                nn.Conv3d(
                    in_channels=self.n_filters,
                    out_channels=self.n_filters,
                    kernel_size=self.kernel_size,
                    stride=1,
                    padding=same_padding,
                )
            ),
            nn.Dropout3d(),
            nn.GroupNorm(1, num_channels=self.n_filters),
        ]
        self.block = nn.Sequential(*tuple(block_elements)).to(device=device)
        self.nonlin = nn.LeakyReLU()

    def forward(self, inputs):
        out = self.block(inputs)
        downsampled = self.downsample(inputs)
        result = 0.5 * (out + downsampled)
        result = self.nonlin(result)
        return result


class Residual3DDeconvBlock(nn.Module):
    def __init__(self, n_filters_in, n_filters, kernel_size, sn=0):
        super(Residual3DDeconvBlock, self).__init__()

        specnorm = _sn_to_specnorm(sn)

        self.n_filters_in = n_filters_in
        self.n_filters = n_filters
        self.kernel_size = kernel_size

        self.conv = specnorm(
            nn.Conv3d(n_filters_in, n_filters, kernel_size=1, stride=1)
        )
        same_padding = compute_same_padding(self.kernel_size, 1, dilation=1)
        block_blocks = [
            specnorm(
                nn.Conv3d(
                    n_filters,
                    n_filters,
                    kernel_size=self.kernel_size,
                    stride=1,
                    padding=same_padding,
                )
            ),
            nn.GroupNorm(1, num_channels=n_filters),
            nn.LeakyReLU(),
            specnorm(
                nn.Conv3d(
                    n_filters,
                    n_filters,
                    kernel_size=self.kernel_size,
                    stride=1,
                    padding=same_padding,
                )
            ),
            nn.GroupNorm(1, num_channels=n_filters),
        ]
        self.block = nn.Sequential(*tuple(block_blocks))
        self.nonlin = nn.LeakyReLU()

    def forward(self, inputs: torch.Tensor):
        # print("SHAPE", inputs.shape)
        inputs = inputs.repeat(1, 1, 2, 2, 2)  # B, C, dims repeat factor 2
        inputs = self.conv(inputs)
        out = self.block(inputs)
        out = 0.5 * (out + inputs)
        out = self.nonlin(out)
        return out


class EmbedNoise(nn.Module):
    def __init__(self, z_dim, channels, sn=0):
        super().__init__()
        specnorm = _sn_to_specnorm(sn)
        self.pad = nn.Linear(z_dim, channels * 4 * 4 * 4)
        self.pad = specnorm(self.pad)
        # self.pad = nn.ConstantPad3d(padding=(3, 3, 3, 3, 3, 3), value=0.)  # -> (B, z_dim, 7, 7, 7)
        # self.conv = nn.Conv3d(z_dim, channels, kernel_size=4, stride=1, padding=0)  # -> (B, channels, 4, 4, 4)
        self.nonlin = nn.LeakyReLU()
        self.z_dim = z_dim
        self.channels = channels

    def forward(self, z):
        # batch_size = z.shape[0]
        out = self.pad(z)
        # out = self.conv(out.view((-1, self.z_dim, 7, 7, 7)))
        out = self.nonlin(out)
        out = out.view((-1, self.channels, 4, 4, 4))
        return out


class GeneratorCombined1Block(nn.Module):
    def __init__(self, in_channels, out_channels, sn=0):
        super().__init__()
        specnorm = _sn_to_specnorm(sn)
        self.conv = specnorm(
            nn.Conv3d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
            )
        )
        self.group_norm = nn.GroupNorm(1, num_channels=out_channels)
        self.nonlin = nn.LeakyReLU()
        self.res1 = Residual3DConvBlock(out_channels, out_channels, 3, 1, sn=sn)
        self.res2 = Residual3DConvBlock(out_channels, out_channels, 3, 1, sn=sn)
        self.res_deconv = Residual3DDeconvBlock(
            n_filters_in=out_channels, n_filters=out_channels, kernel_size=3, sn=sn
        )

    def forward(self, embedded_z, down2):
        out = torch.cat((embedded_z, down2), dim=1)
        out = self.conv(out)
        out = self.group_norm(out)
        out = self.nonlin(out)
        out = self.res1(out)
        out = self.res2(out)
        out = self.res_deconv(out)
        return out


class GeneratorCombined2Block(nn.Module):
    def __init__(self, channels, sn=0):
        super().__init__()
        self.conv = Residual3DConvBlock(
            channels, n_filters=channels, kernel_size=3, stride=1, sn=sn
        )
        self.deconv = Residual3DDeconvBlock(
            n_filters_in=channels, n_filters=channels, kernel_size=3, sn=sn
        )

    def forward(self, inputs):
        out = self.conv(inputs)
        out = self.deconv(out)
        return out

class GeneratorCombined3Block(nn.Module):
    def __init__(self, in_channels, out_channels, sn=0):
        super().__init__()
        specnorm = _sn_to_specnorm(sn)
        self.conv = specnorm(
            nn.Conv3d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
            )
        )
        self.group_norm = nn.GroupNorm(1, num_channels=out_channels)
        self.nonlin = nn.LeakyReLU()
        self.res_deconv = Residual3DDeconvBlock(
            n_filters_in=out_channels, n_filters=out_channels, kernel_size=3, sn=sn
        )

    def forward(self, embedded_z, down2):
        out = torch.cat((embedded_z, down2), dim=1)
        out = self.conv(out)
        out = self.group_norm(out)
        out = self.nonlin(out)
        out = self.res_deconv(out)
        return out


