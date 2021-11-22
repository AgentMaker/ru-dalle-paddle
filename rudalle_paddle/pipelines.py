# -*- coding: utf-8 -*-
import paddle
import transformers
import more_itertools
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

from . import utils


def generate_images(text, tokenizer, dalle, vae, top_k, top_p, images_num, image_prompts=None, temperature=1.0, bs=8,
                    seed=None, use_cache=True):
    # TODO docstring
    if seed is not None:
        utils.seed_everything(seed)

    vocab_size = dalle.get_param('vocab_size')
    text_seq_length = dalle.get_param('text_seq_length')
    image_seq_length = dalle.get_param('image_seq_length')
    total_seq_length = dalle.get_param('total_seq_length')
    device = dalle.get_param('device')

    text = text.lower().strip()
    input_ids = tokenizer.encode_text(text, text_seq_length=text_seq_length)
    pil_images, scores = [], []
    for chunk in more_itertools.chunked(range(images_num), bs):
        chunk_bs = len(chunk)
        with paddle.no_grad():
            attention_mask = paddle.tril(paddle.ones((chunk_bs, 1, total_seq_length, total_seq_length)).to(device))
            out = input_ids.unsqueeze(0).tile([chunk_bs, 1]).to(device)
            has_cache = False
            sample_scores = []
            if image_prompts is not None:
                prompts_idx, prompts = image_prompts.image_prompts_idx, image_prompts.image_prompts
                prompts = prompts.tile([images_num, 1])
                if use_cache:
                    use_cache = False
                    print('Warning: use_cache changed to False')
            for idx in tqdm(range(out.shape[1], total_seq_length)):
                idx -= text_seq_length
                if image_prompts is not None and idx in prompts_idx:
                    out = paddle.concat((out, prompts[:, idx].unsqueeze(1)), axis=-1)
                else:
                    logits, has_cache = dalle(out, attention_mask,
                                              has_cache=has_cache, use_cache=use_cache, return_loss=False)
                    logits = logits[:, -1, vocab_size:]
                    logits /= temperature
                    filtered_logits = transformers.top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
                    probs = paddle.nn.functional.softmax(filtered_logits, axis=-1)
                    probs = paddle.nn.functional.softmax(logits, axis=-1)
                    sample = paddle.multinomial(probs, 1)
                    sample_scores.append(probs[paddle.arange(probs.shape[0]), sample.swapaxes(0, 1)])
                    out = paddle.concat((out, sample), axis=-1)
            codebooks = out[:, -image_seq_length:]
            images = vae.decode(codebooks)
            pil_images += utils.paddle_tensors_to_pil_list(images)
            scores += paddle.concat(sample_scores).sum(0).detach().cpu().numpy().tolist()
    return pil_images, scores


def super_resolution(pil_images, realesrgan):
    result = []
    for pil_image in pil_images:
        with paddle.no_grad():
            sr_image = realesrgan.predict(np.array(pil_image))
        result.append(sr_image)
    return result


def cherry_pick_by_clip(pil_images, text, ruclip, ruclip_processor, device='cpu', count=4):
    with paddle.no_grad():
        inputs = ruclip_processor(text=text, images=pil_images)
        for key in inputs.keys():
            inputs[key] = inputs[key].to(device)
        outputs = ruclip(**inputs)
        sims = paddle.nn.functional.softmax(outputs.logits_per_image.reshape([-1]), axis=0)
        items = []
        for index, sim in enumerate(sims.cpu().numpy()):
            items.append({'img_index': index, 'cosine': sim})
    items = sorted(items, key=lambda x: x['cosine'], reverse=True)[:count]
    top_pil_images = [pil_images[x['img_index']] for x in items]
    top_scores = [x['cosine'] for x in items]
    return top_pil_images, top_scores


def show(pil_images, nrow=4):
    imgs = utils.pil_list_to_paddle_tensors(pil_images)
    imgs = paddle.nn.functional.pad(imgs.astype(paddle.int64), [1, 1, 1, 1], value=0)
    imgs = paddle.concat(paddle.concat(imgs.split(imgs.shape[0]//nrow, 0), 2).split(nrow, 0), 3)
    if not isinstance(imgs, list):
        imgs = [imgs.cpu()]
    fix, axs = plt.subplots(ncols=len(imgs), squeeze=False, figsize=(14, 14))
    for i, img in enumerate(imgs):
        img = img.detach().numpy().astype(np.uint8)[0].transpose([1, 2, 0])
        axs[0, i].imshow(img)
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    fix.show()
    plt.show()
