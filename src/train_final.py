import torch
import json
import os
from accelerate import Accelerator
from omegaconf import OmegaConf
from tqdm import  tqdm
import time
from collections import Counter
import multiprocessing
from datetime import datetime
import random
import torch.nn.functional as F
from transformers import AutoModel
from safetensors import safe_open
import wandb
from PIL import Image
from safetensors.torch import load_file
import open_clip
from transformers import AutoTokenizer
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
import matplotlib.pyplot as plt
from safetensors.torch import save_file
from safetensors.torch import save_model
from joblib import Parallel, delayed, effective_n_jobs
from args import get_args
import models.utils as utils
from data.dataset import get_nsd_data, MultiSubjectBatchIterator, collate_fn
from models.brain_omni import BrainOmni
from joblib import Parallel, delayed
import numpy as np
from transformers import get_cosine_schedule_with_warmup

def blue(text: str) -> str:
    return f"\033[34m{text}\033[0m"

@torch.no_grad()
def fmri2img(model, subj_idx, test_loader, save_dir, accelerator):
    src_dir = os.path.join(save_dir, 'src')
    gen_dir = os.path.join(save_dir, 'gen')
    concat_path = os.path.join(save_dir, 'concat')
    os.makedirs(src_dir, exist_ok=True)
    os.makedirs(gen_dir, exist_ok=True)
    os.makedirs(concat_path, exist_ok=True)

    idx_lists = []
    src_lists = []
    gen_lists = []

    for i, batch in enumerate(tqdm(test_loader)):
        voxel = batch["voxels"]
        # 这里的 batch["idx"] 和 batch["images"] 是当前进程分配到的局部数据
        curr_idx = batch["idx"]
        curr_images = batch["images"]
        
        # 模型推理
        generated_imgs = model.infer_fmri2image(voxel, subj_idx, cfg_weight=5.0)

        # 1. 先 Gather: 将所有进程中的当前 batch 数据汇总
        # gather_for_metrics 会自动处理分布式采样导致的重复样本剔除
        reduce_indices = accelerator.gather_for_metrics(curr_idx)
        reduce_srcs = accelerator.gather_for_metrics(curr_images)
        reduce_gens = accelerator.gather_for_metrics(generated_imgs)

        # 2. 再 Append: 将汇总后的数据转到 CPU 并存入列表
        # 建议在 append 之前 detach() 并转到 cpu，防止显存溢出
        idx_lists.append(reduce_indices.detach().cpu().to(torch.int64))
        
        # 转换维度 (B, C, H, W) -> (B, H, W, C) 方便后续可视化或保存
        src_lists.append(reduce_srcs.permute(0, 2, 3, 1).detach().cpu())
        gen_lists.append(reduce_gens.permute(0, 2, 3, 1).detach().cpu())

    # 3. 最后 Concat: 循环结束后，将列表中所有的 batch 合并成一个大的 Tensor
    all_indices = torch.cat(idx_lists, dim=0)
    all_srcs = torch.cat(src_lists, dim=0)
    all_gens = torch.cat(gen_lists, dim=0)

    if accelerator.is_main_process:

        indices_np = all_indices.numpy()
        src_np = all_srcs.numpy()
        gen_np = all_gens.numpy()
        
        save_data = {
            "indices": all_indices,
            "src": all_srcs,  # 原始图像 (B, C, H, W)
            "gen": all_gens,  # 生成图像 (B, C, H, W)
        }
        torch.save(save_data, os.path.join(save_dir, f"fmri2img.pt"))

        def fast_save(idx_val, s_arr, g_arr):
            Image.fromarray(s_arr).save(f"{save_dir}/src/{idx_val:04d}.png")
            Image.fromarray(g_arr).save(f"{save_dir}/gen/{idx_val:04d}.png")
            combined_arr = np.concatenate([s_arr, g_arr], axis=1)
            Image.fromarray(combined_arr).save(f"{concat_path}/{idx_val:04d}.png")
        num_workers = len(os.sched_getaffinity(0))
        Parallel(n_jobs=num_workers, prefer="threads")(
            delayed(fast_save)(indices_np[i], src_np[i], gen_np[i])
            for i in range(len(indices_np))
        )

@torch.no_grad()
def fmri2text(model, subj_idx, test_loader, save_dir, accelerator):
    all_src, all_gen = [], []
    for i, batch in enumerate(tqdm(test_loader)):
        voxel     = batch["voxels"]
        captions  = batch["captions"]

        generated_texts = model.infer_fmri2text(voxel, subj_idx)

        gathered_src_lists = accelerator.gather_for_metrics(captions)
        gathered_gen_lists = accelerator.gather_for_metrics(generated_texts)

        all_src.extend(gathered_src_lists)
        all_gen.extend(gathered_gen_lists)

    save_path = os.path.join(save_dir, f"fmri2text.json")

    if accelerator.is_main_process:
        data = { "src": all_src, "gen": all_gen }
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

@torch.no_grad()
def img2fmri(model, subj_idx, test_loader, save_dir, accelerator):
    src_voxel = []
    gen_voxel = []
    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_loader)):
            img       = batch["images"]
            voxel     = batch["voxels"]
            mask = batch["masks"]
            L = int(mask.sum(dim=1)[0])
            recon, ce_loss = model.infer_image2fmri(voxel, img, subj_idx, L,'VQ')
            voxel = voxel[:, :L]
            recon = recon[:, :L]

            src_voxel_gather = accelerator.gather_for_metrics(voxel)
            gen_voxel_gather = accelerator.gather_for_metrics(recon)

            src_voxel.append(src_voxel_gather.cpu())
            gen_voxel.append(gen_voxel_gather.cpu())

    if accelerator.is_main_process:
        src_voxel = torch.cat(src_voxel, dim=0)
        gen_voxel = torch.cat(gen_voxel, dim=0)
        save_file({ "src_voxel": src_voxel, "gen_voxel": gen_voxel}, os.path.join(save_dir, "img2fmri.safetensors") )


@torch.no_grad()
def text2fmri(model, subj_idx, test_loader, save_dir, accelerator):
    src_voxel = []
    gen_voxel = []
    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_loader)):
            img       = batch["images"]
            voxel     = batch["voxels"]
            mask = batch["masks"]
            captions   = batch["captions"]
            
            L = int(mask.sum(dim=1)[0])
            recon, ce_loss = model.infer_text2fmri(voxel, captions, subj_idx, L)
            voxel = voxel[:, :L]
            recon = recon[:, :L]

            src_voxel_gather = accelerator.gather_for_metrics(voxel)
            gen_voxel_gather = accelerator.gather_for_metrics(recon)

            src_voxel.append(src_voxel_gather.cpu())
            gen_voxel.append(gen_voxel_gather.cpu())

    if accelerator.is_main_process:
        src_voxel = torch.cat(src_voxel, dim=0)
        gen_voxel = torch.cat(gen_voxel, dim=0)
        save_file({ "src_voxel": src_voxel, "gen_voxel": gen_voxel}, os.path.join(save_dir, "text2fmri.safetensors") )

if __name__ == "__main__":
    # from models.janus import VLChatProcessor
    # janus_vl_chat_processor = VLChatProcessor.from_pretrained('deepseek-ai/Janus-Pro-7B')

    # print(janus_vl_chat_processor)

    # exit(1)

    # start = time.time()
    accelerator = Accelerator()


    args, unknown = get_args()
    cfg = OmegaConf.load(args.config)
    unknown_cfg = OmegaConf.from_cli(unknown)
    unknown_cfg = {k.lstrip('-'): v for k, v in unknown_cfg.items()}
    cfg = OmegaConf.merge(cfg, vars(args), unknown_cfg)

    output_dir = cfg.output_dir

    # time_str = datetime.now().strftime("%Y%m%d_%H%M")
    # name = f"{cfg.name}_{time_str}"
    name = cfg.name
    
    save_dir = os.path.join(output_dir, name)
    
    os.makedirs(save_dir, exist_ok=True)
    log_dir = output_dir
    if accelerator.is_main_process:
        wandb.init(project="BrainRosetta", name=name, dir = log_dir)

    cfg, train_data, test_data = get_nsd_data(cfg)


    pad_S = 2**(len(cfg.encoder_ch_mult)-1)
    train_loaders = {k: accelerator.prepare(torch.utils.data.DataLoader(v, batch_size=cfg.batch_size, collate_fn=lambda batch: collate_fn(batch,pad_S), shuffle=True, drop_last=True, pin_memory=True, num_workers=0))
        for k,v in train_data.items()
    }
    train_iterator = MultiSubjectBatchIterator(train_loaders, accelerator)

    test_loaders = {k: accelerator.prepare(torch.utils.data.DataLoader(v, batch_size=cfg.test_batch_size, shuffle=False, collate_fn=lambda batch: collate_fn(batch,pad_S), drop_last=False, pin_memory=True, num_workers=0))
        for k,v in test_data.items()
    }

    device = accelerator.device

    cfg.voxel_num = {k: v.voxel_num for k,v in train_data.items()}

    
    model = BrainOmni(cfg, janus_model_name=cfg.model_type, accelerator=accelerator)
    
    if cfg.ckpt_dir != None:
        ckpt_dir = cfg.ckpt_dir
        # ckpt_path = f"{ckpt_dir}/model.safetensors"
        # state_dict = {}
        # with safe_open(ckpt_path, framework="pt", device="cpu") as f:
        #     for k in f.keys():
        #         state_dict[k] = f.get_tensor(k)
        # model.vq_fmri_encoder.load_state_dict(state_dict, strict=True)
        # model.vq_fmri_encoder.requires_grad_(False)

        ckpt_path = f"{ckpt_dir}/model.pt"
        state_dict = torch.load(ckpt_path, map_location="cpu")
        model.vq_fmri_encoder.load_state_dict(state_dict, strict=True)
        model.vq_fmri_encoder.requires_grad_(False)

        # model.vq_fmri_encoder.quantize.embedding.requires_grad_(False)
        # model.vq_fmri_encoder.post_quant_conv.requires_grad_(True)
        # model.vq_fmri_encoder.decoder.requires_grad_(True)
    # optimizer = torch.optim.Adam(brain_omni.parameters(), lr=1e-4)


    model.print_trainable_params(accelerator=accelerator)


    optimizer = torch.optim.AdamW(model.parameters(),  lr=1e-4, betas=(0.9, 0.95)) #4e-5

    total_steps_per_epoch = sum(len(loader) for loader in train_loaders.values())
    num_training_steps = total_steps_per_epoch * cfg.epoch * accelerator.num_processes
    num_warmup_steps = int(0.05 * num_training_steps)

    lr_scheduler = get_cosine_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)
    # print("Trainable parameters:")
    # for name, param in model.named_parameters(): # 或者 model.named_parameters()
    #     if param.requires_grad:
    #         print(f" - {name}")
    _ = torch.utils.data.DataLoader(next(iter(test_data.values())), batch_size=cfg.test_batch_size, shuffle=False, collate_fn=lambda batch: collate_fn(batch,pad_S), drop_last=False, pin_memory=True, num_workers=0)
    model, optimizer, _, lr_scheduler = accelerator.prepare( model, optimizer, _, lr_scheduler)


    if accelerator.is_main_process:
        config_save_path = os.path.join(save_dir, "config.yaml")
        OmegaConf.save(config=cfg, f=config_save_path, resolve=True)
        accelerator.print(f"Config saved to: {config_save_path}")

    accelerator.print("Running baseline test before training...")
    model.eval()
    epoch_save_dir = os.path.join(save_dir, f"epoch_-1")
    for sid, test_loader in test_loaders.items():
        test_save_dir = os.path.join(epoch_save_dir,"test",f"subj_{sid:02d}")
        if accelerator.is_main_process:
            os.makedirs(test_save_dir, exist_ok=True)
        # fmri2img(model, sid, test_loader, test_save_dir, accelerator)
        # fmri2text(model, sid, test_loader, test_save_dir, accelerator)
        # img2fmri(model, sid, test_loader, test_save_dir, accelerator)
        # text2fmri(model, sid, test_loader, test_save_dir, accelerator)
    torch.cuda.empty_cache()
                                                 
    global_step = 0
    for epoch in range(cfg.epoch):
        model.train()
        epoch_save_dir = os.path.join(save_dir, f"epoch_{epoch}")
        

        pbar = tqdm(train_iterator, disable=not accelerator.is_main_process)
        for i, (sid, task, batch) in enumerate(pbar):
            task = cfg.task if cfg.task else task

            
            
            fmri_tokens, loss_dict, info = model.train_final(batch, sid, task)
            loss =  loss_dict['ce_loss']

            if torch.isnan(loss):
                accelerator.print("Detected NaN loss!")

            accelerator.backward(loss)
            optimizer.step()

            if lr_scheduler is not None:
                lr_scheduler.step()

            optimizer.zero_grad()
            
            

            if epoch==0 and i==0:
                accelerator.print(f"{fmri_tokens.shape=}")


            # accelerator.print(blue(f"Epoch {epoch}, Iter {i} Loss: {avg_loss}"))
            if accelerator.is_main_process:

                tqdm.write(
                    blue(f"Epoch {epoch} Iter {i} Loss {loss.item():.3f}, PPL {info['ppl']:.3f}")
                )

            if accelerator.is_main_process:
                wandb.log({"Train/Loss": loss.item(), "Train/PPL": info['ppl'], "Train/LR": optimizer.param_groups[0]['lr'], "epoch": epoch}, step=global_step)
                global_step += 1

            # torch.cuda.empty_cache()

        if (epoch+1)%3 == 0 or epoch == cfg.epoch-1 :
            os.makedirs(epoch_save_dir, exist_ok=True)
            model.eval()
            test_dict = {}
            for subj_idx, test_loader in test_loaders.items():
                test_save_dir = os.path.join(epoch_save_dir,"test",f"subj_{subj_idx:02d}")
                if accelerator.is_main_process:
                    os.makedirs(test_save_dir, exist_ok=True)
                info = {}
                info['fmri2img'] = fmri2img(model, subj_idx, test_loader, test_save_dir, accelerator)
                # info['fmri2text'] = fmri2text(model, subj_idx, test_loader, test_save_dir, accelerator)
                # info['img2fmri'] = img2fmri(model, subj_idx, test_loader, test_save_dir, accelerator)
                # info['text2fmri'] = text2fmri(model, subj_idx, test_loader, test_save_dir, accelerator)

                test_dict[f"subj_{subj_idx:02d}"] = info

            if accelerator.is_main_process:
                    wandb.log({"Test": test_dict}, step=global_step)

        if ((epoch + 1) % 15 == 0 or epoch == cfg.epoch-1) and accelerator.is_main_process:
            os.makedirs(epoch_save_dir, exist_ok=True)
            unwrapped = accelerator.unwrap_model(model)
            torch.save(unwrapped.state_dict(), os.path.join(epoch_save_dir, "model.pt") )
            accelerator.print(f"Saving checkpoint for epoch {epoch} to {epoch_save_dir}/model.pt ...")
            
        
    # checkpoint_path = os.path.join(output_dir, name, f"AAA_Last")
    # accelerator.print(f"Saving checkpoint for last epoch to {checkpoint_path}...")
    # accelerator.save_state(checkpoint_path)