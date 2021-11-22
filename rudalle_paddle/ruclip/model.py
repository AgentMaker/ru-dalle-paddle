# -*- coding: utf-8 -*-
import os
import json
import paddle

from clip import CLIP


class CLIPModel(CLIP):
    def encode_text(self, text):
        x = self.token_embedding(text)
        if x.shape[1] != self.context_length:
            x = paddle.concat([
                x,
                paddle.zeros(
                    [x.shape[0], self.context_length - x.shape[1], x.shape[2]],
                    dtype=x.dtype
                )
            ], 1)
        x = x + self.positional_embedding
        x = self.transformer(x)
        x = self.ln_final(x)[:, :text.shape[1]]

        select = []
        index = zip(
            paddle.arange(x.shape[0]).numpy(),
            text.argmax(axis=-1).numpy()
        )
        for i, j in index:
            select.append(x[int(i), int(j)])

        x = paddle.stack(select) @ self.text_projection

        return x

    def forward(self, **kwargs):
        logits_per_image, logits_per_text = super(CLIPModel, self).forward(
            kwargs.get('pixel_values'), kwargs.get('input_ids'))
        outputs = type('LamdaCls', (), {
            'logits_per_image': logits_per_image,
            'logits_per_text': logits_per_text
        })
        return outputs

    @classmethod
    def from_pretrained(cls, folder):
        with open(os.path.join(folder, 'config.json'), 'r', encoding='utf-8') as f:
            src_conf = json.load(f)
        dst_conf = {
            'embed_dim': src_conf['projection_dim'],
            # vision
            'image_resolution': src_conf['vision_config']['image_size'],
            'vision_layers': src_conf['vision_config']['num_hidden_layers'],
            'vision_width': src_conf['vision_config']['hidden_size'],
            'vision_patch_size': src_conf['vision_config_dict']['patch_size'],
            # text
            'context_length': src_conf['text_config']['max_position_embeddings'],
            'vocab_size': src_conf['text_config']['vocab_size'],
            'transformer_width': src_conf['text_config']['hidden_size'],
            'transformer_heads': src_conf['text_config']['num_attention_heads'],
            'transformer_layers': src_conf['text_config']['num_hidden_layers'],
        }
        obj = cls(**dst_conf)
        paddle_weights = os.path.join(folder, 'ruclip_paddle.pdparams')
        if not os.path.exists(paddle_weights):
            cls.convert(folder)
        obj.set_state_dict(paddle.load(paddle_weights))
        return obj

    @staticmethod
    def convert(folder):
        import os
        import torch
        torch_weights = os.path.join(folder, 'pytorch_model.bin')
        target_model_path = os.path.join(folder, 'ruclip_paddle.pdparams')
        if os.path.exists(target_model_path):
            return
        state_dict = torch.load(torch_weights, map_location='cpu')

        name_pairs = [
            ('text_model.embeddings.position_embedding.weight', 'positional_embedding', False),
            ('visual_projection.weight', 'visual.proj', True),
            ('text_projection.weight', 'text_projection', True),
            ('text_model.embeddings.token_embedding.weight', 'token_embedding.weight', False),
            ('logit_scale', 'logit_scale', False),
            ('vision_model.embeddings.class_embedding', 'visual.class_embedding', False),
            ('vision_model.embeddings.patch_embedding.weight', 'visual.conv1.weight', False),
            ('vision_model.embeddings.position_embedding.weight', 'visual.positional_embedding', False),
            ('vision_model.pre_layrnorm', 'visual.ln_pre', False),
            ('vision_model.encoder.layers', 'visual.transformer.resblocks', True),
            ('text_model.encoder.layers', 'transformer.resblocks', True),
            ('self_attn.k_proj', 'attn.k_proj', True),
            ('self_attn.v_proj', 'attn.v_proj', True),
            ('self_attn.q_proj', 'attn.q_proj', True),
            ('self_attn.out_proj', 'attn.out_proj', True),
            ('layer_norm1', 'ln_1', False),
            ('layer_norm2', 'ln_2', False),
            ('mlp.fc1', 'mlp.c_fc', True),
            ('mlp.fc2', 'mlp.c_proj', True),
            ('vision_model.post_layernorm', 'visual.ln_post', False),
            ('text_model.final_layer_norm', 'ln_final', False)
        ]
        exclude_names = [
            'text_model.embeddings.position_ids',
            'vision_model.embeddings.position_ids'
        ]
        paddle_state_dict = {}
        for name, param in state_dict.items():
            is_pair = False
            no_need_transpose = True
            if name in exclude_names:
                continue
            for pre_name, post_name, do_transpose in name_pairs:
                if pre_name in name:
                    is_pair = True
                    name = name.replace(pre_name, post_name)
                    no_need_transpose = not do_transpose if no_need_transpose else False
            assert is_pair, f'Weight of {name} need to be converted.'
            if not no_need_transpose and param.ndim == 2:
                param = param.transpose(1, 0)
            if param.ndim == 0:
                param = param.unsqueeze(0)
            param = param.cpu().detach().numpy()
            paddle_state_dict[name] = param

        paddle.save(paddle_state_dict, target_model_path)
