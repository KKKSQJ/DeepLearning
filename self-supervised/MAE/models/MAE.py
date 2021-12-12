import torch
import torch.nn as nn
from models.VIT import ViT, Transformer
import torch.nn.functional as F


class MAE(nn.Module):
    def __init__(self,
                 encoder, decoder_dim,
                 mask_ratio=0.75, decoder_depth=1,
                 num_decoder_heads=8, decoder_dim_per_head=64):
        super(MAE, self).__init__()
        assert 0.0 < mask_ratio < 1.0, f"mask ratio must be kept between 0 and 1, got :{mask_ratio}"

        # Encoder:使用 vit 实现
        self.encoder = encoder
        # 16 * 16 (patch_Size)
        self.patch_h, self.patch_w = encoder.patch_h, encoder.patch_w

        # 由于原生的Vit有cls_token,因此其position embedding的倒数第2个维度是：
        # 实际划分的patch数量 + 1个cls_token

        # 197=image_size/patch_Size **2 + 1(cls_token), 768
        num_patches_plus_cls_token, encoder_dim = encoder.pos_embed.shape[-2:]

        # encoder patch embedding的输入通道：patch_size ** 2 * 3
        # 预测头的输出通道，从而能够对patch中的所有通道像素值进行预测
        # 这里其实有问题，这里的weight.shape[0]应该是encoder_dim=768更好等于16*16*3.
        num_pixels_per_patch = encoder.patch_embed.weight.size(1)

        # encoder-decoder:encoder 输出的维度可能和decoder要求的输入维度不一致，因此需要转换
        self.enc_to_dec = nn.Linear(encoder_dim, decoder_dim) if encoder_dim != decoder_dim else nn.Identity()

        # mask token
        self.mask_ratio = mask_ratio
        # mask token的本质是：一个可学习的共享向量
        self.mask_embed = nn.Parameter(torch.randn(decoder_dim))

        # decoder:多层堆叠的transformer block
        self.decoder = Transformer(
            decoder_dim,
            decoder_dim * 4,
            depth=decoder_depth,
            num_heads=num_decoder_heads,
            dim_per_head=decoder_dim_per_head,
        )

        # 在decoder 中用作对mask tokens 的 position embedding
        # 过滤掉cls_token ，因此去掉第一个维度的cls_token
        self.decoder_pos_embed = nn.Embedding(num_embeddings=num_patches_plus_cls_token - 1, embedding_dim=decoder_dim)

        # 预测头输出的维度数量等于1个patch的像素值数量
        self.head = nn.Linear(decoder_dim, num_pixels_per_patch)
        self.apply(self.init_weights)

    def init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Conv2d):
            # NOTE conv was left to pytorch default in my original init
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)

    def forward(self, x):
        b, c, h, w = x.shape
        device = x.device
        # 得到多少个patch块
        num_patches = (h // self.patch_h) * (w // self.patch_w)
        # (b, c=3, h, w)->(b, n_patches, patch_size**2*c)
        patches = x.view(
            b, c,
            h // self.patch_h, self.patch_h,
            w // self.patch_w, self.patch_w
        ).permute(0, 2, 4, 3, 5, 1).reshape(b, num_patches, -1)

        # 获取每张图片需要mask的索引
        num_masked = int(self.mask_ratio * num_patches)
        # shuffle, 排序，取前百分之1-mask_ratio的patches参与训练
        # Shuffle:生成对应 patch 的随机索引
        # torch.rand() 服从均匀分布(normal distribution)
        # torch.rand() 只是生成随机数，argsort() 是为了获得成索引
        # (b, num_patches)
        shuffle_indices = torch.rand(b, num_patches, device=device).argsort()
        mask_indices, unmask_indices = shuffle_indices[:, :num_masked], shuffle_indices[:, num_masked:]

        # 利用先前生成的索引对 patches 进行采样，分为 mask 和 unmasked 两组
        # (b, 1)
        batch_indices = torch.arange(b, device=device).unsqueeze(-1)
        mask_patches, unmask_patches = patches[batch_indices, mask_indices], patches[batch_indices, unmask_indices]

        # encoder
        # 将 patches 通过 emebdding 转换成 tokens
        unmask_tokens = self.encoder.patch_embed(unmask_patches)

        # 为 tokens 加入 position embeddings
        # 注意这里索引加1是因为索引0对应 ViT 的 cls_token
        unmask_tokens += self.encoder.pos_embed.repeat(b, 1, 1)[batch_indices, unmask_indices + 1]

        # 真正的编码过程
        encoded_tokens = self.encoder.transformer(unmask_tokens)

        # decoder
        # 对编码后的 tokens 维度进行转换，从而符合 Decoder 要求的输入维度
        enc_to_dec_tokens = self.enc_to_dec(encoded_tokens)

        # 由于 mask token 实质上只有1个，因此要对其进行扩展，从而和 masked patches 一一对应
        # (decoder_dim)->(b, n_masked, decoder_dim)
        mask_tokens = self.mask_embed[None, None, :].repeat(b, num_masked, 1)
        # 为 mask tokens 加入位置信息
        mask_tokens += self.decoder_pos_embed(mask_indices)

        # 将 mask tokens 与 编码后的 tokens 拼接起来
        # (b, n_patches, decoder_dim)
        concat_tokens = torch.cat([mask_tokens, enc_to_dec_tokens], dim=1)
        # Un-shuffle：恢复原先 patches 的次序
        dec_input_tokens = torch.empty_like(concat_tokens, device=device)
        dec_input_tokens[batch_indices, shuffle_indices] = concat_tokens
        # 将全量 tokens 喂给 Decoder 解码
        decoded_tokens = self.decoder(dec_input_tokens)

        # Loss Computation
        """
        取出解码后的 mask tokens 送入头部进行像素值预测，然后将预测结果和 masked patches 比较，计算 MSE loss：
        """
        # 取出解码后的 mask tokens
        dec_mask_tokens = decoded_tokens[batch_indices, mask_indices, :]
        # 预测 masked patches 的像素值
        # (b, n_masked, n_pixels_per_patch=patch_size**2 x c)
        pred_mask_pixel_values = self.head(dec_mask_tokens)
        return pred_mask_pixel_values, mask_patches

        # loss = F.mse_loss(pred_mask_pixel_values, mask_patches)
        # return loss

    @torch.no_grad()
    def predict(self, x):
        self.eval()

        device = x.device
        b, c, h, w = x.shape

        '''i. Patch partition'''

        num_patches = (h // self.patch_h) * (w // self.patch_w)
        # (b, c=3, h, w)->(b, n_patches, patch_size**2*c)
        patches = x.view(
            b, c,
            h // self.patch_h, self.patch_h,
            w // self.patch_w, self.patch_w
        ).permute(0, 2, 4, 3, 5, 1).reshape(b, num_patches, -1)

        '''ii. Divide into masked & un-masked groups'''

        num_masked = int(self.mask_ratio * num_patches)

        # Shuffle
        # (b, n_patches)
        shuffle_indices = torch.rand(b, num_patches, device=device).argsort()
        mask_indices, unmask_indices = shuffle_indices[:, :num_masked], shuffle_indices[:, num_masked:]

        # (b, 1)
        batch_indices = torch.arange(b, device=device).unsqueeze(-1)
        mask_patches, unmask_patches = patches[batch_indices, mask_indices], patches[batch_indices, unmask_indices]

        '''iii. Encode'''

        unmask_tokens = self.encoder.patch_embed(unmask_patches)
        # Add position embeddings
        unmask_tokens += self.encoder.pos_embed.repeat(b, 1, 1)[batch_indices, unmask_indices + 1]
        encoded_tokens = self.encoder.transformer(unmask_tokens)

        '''iv. Decode'''

        enc_to_dec_tokens = self.enc_to_dec(encoded_tokens)

        # (decoder_dim)->(b, n_masked, decoder_dim)
        mask_tokens = self.mask_embed[None, None, :].repeat(b, num_masked, 1)
        # Add position embeddings
        mask_tokens += self.decoder_pos_embed(mask_indices)

        # (b, n_patches, decoder_dim)
        concat_tokens = torch.cat([mask_tokens, enc_to_dec_tokens], dim=1)
        # dec_input_tokens = concat_tokens
        dec_input_tokens = torch.empty_like(concat_tokens, device=device)
        # Un-shuffle
        dec_input_tokens[batch_indices, shuffle_indices] = concat_tokens
        decoded_tokens = self.decoder(dec_input_tokens)

        '''v. Mask pixel Prediction'''

        dec_mask_tokens = decoded_tokens[batch_indices, mask_indices, :]
        # (b, n_masked, n_pixels_per_patch=patch_size**2 x c)
        pred_mask_pixel_values = self.head(dec_mask_tokens)

        # 比较下预测值和真实值
        mse_per_patch = (pred_mask_pixel_values - mask_patches).abs().mean(dim=-1)
        mse_all_patches = mse_per_patch.mean()

        print(
            f'mse per (masked)patch: {mse_per_patch} mse all (masked)patches: {mse_all_patches} total {num_masked} masked patches')
        print(f'all close: {torch.allclose(pred_mask_pixel_values, mask_patches, rtol=1e-1, atol=1e-1)}')

        '''vi. Reconstruction'''

        recons_patches = patches.detach()
        # Un-shuffle (b, n_patches, patch_size**2 * c)
        recons_patches[batch_indices, mask_indices] = pred_mask_pixel_values
        # 模型重建的效果图
        # Reshape back to image
        # (b, n_patches, patch_size**2 * c)->(b, c, h, w)
        recons_img = recons_patches.view(
            b, h // self.patch_h, w // self.patch_w,
            self.patch_h, self.patch_w, c
        ).permute(0, 5, 1, 3, 2, 4).reshape(b, c, h, w)

        mask_patches = torch.randn_like(mask_patches, device=mask_patches.device)
        # mask 效果图
        patches[batch_indices, mask_indices] = mask_patches
        patches_to_img = patches.view(
            b, h // self.patch_h, w // self.patch_w,
            self.patch_h, self.patch_w, c
        ).permute(0, 5, 1, 3, 2, 4).reshape(b, c, h, w)

        return recons_img, patches_to_img


def MAEVisonTransformer(
        image_size=224,
        patch_size=16,
        encoer_dim=512,
        mlp_dim=1024,
        encoder_depth=6,
        num_encoder_head=8,
        dim_per_head=64,
        decoder_dim=512,
        decoder_depth=6,
        num_decoder_head=8,
        mask_ratio=0.75):
    encoder = ViT(image_size=image_size, patch_size=patch_size, dim=encoer_dim, mlp_dim=mlp_dim,
                  dim_per_head=dim_per_head, depth=encoder_depth, num_heads=num_encoder_head)
    mae = MAE(encoder=encoder, decoder_dim=decoder_dim, decoder_depth=decoder_depth, mask_ratio=mask_ratio,
              num_decoder_heads=num_decoder_head)
    return mae


if __name__ == '__main__':
    from torchsummary import summary

    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    encoder = ViT(image_size=224, patch_size=16, dim=512, mlp_dim=1024, dim_per_head=64)
    mae = MAE(encoder=encoder, decoder_dim=512, decoder_depth=6).to(device)
    summary(mae, input_size=(3, 224, 224))
    # print(mae)
