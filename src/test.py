import json
import os
import argparse
import time
import re

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from PIL import Image, ImageFile
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from sklearn.metrics import accuracy_score, f1_score
from accelerate import PartialState
import jieba
from nltk.translate.bleu_score import sentence_bleu,SmoothingFunction

from model.vlm import LinearVLM
from model.create import create_model
from judger import compute_vqa_accuracy

punct = [";", r"/", "[", "]", '"', "{", "}", "(", ")", "=", "+", "\\", "_", "-", ">", "<", "@", "`", ",", "?", "!", ':']
ImageFile.LOAD_TRUNCATED_IMAGES = True


path_to_pope = ""
path_to_gqa = ""
path_to_vqav2 = ""
path_to_okvqa = ""
path_to_coco_val_2014 = ""
path_to_robovqa = ""
path_to_sapien = ""
path_to_vizwiz = ""


def world_info_from_env():
    local_rank = 0
    for v in (
            "LOCAL_RANK",
            "MPI_LOCALRANKID",
            "SLURM_LOCALID",
            "OMPI_COMM_WORLD_LOCAL_RANK",
    ):
        if v in os.environ:
            local_rank = int(os.environ[v])
            break
    global_rank = 0
    for v in ("RANK", "PMI_RANK", "SLURM_PROCID", "OMPI_COMM_WORLD_RANK"):
        if v in os.environ:
            global_rank = int(os.environ[v])
            break
    world_size = 1
    for v in ("WORLD_SIZE", "PMI_SIZE", "SLURM_NTASKS", "OMPI_COMM_WORLD_SIZE"):
        if v in os.environ:
            world_size = int(os.environ[v])
            break

    return local_rank, global_rank, world_size


class VQADataset(Dataset):
    def __init__(self, data, transform, prefix, new_token, input_lambda=None, img_lambda=None):
        self.data = data
        self.prefix = prefix
        self.new_token = new_token
        self.transform = transform
        self.input_lambda = input_lambda
        self.img_lambda = img_lambda
        self.empty_img = torch.zeros_like(self.transform(Image.new("RGB", (1, 1))))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_text = self.input_lambda(self.data[idx])
        input_text = f"<image>Question:{input_text} Answer:"
        img_idx = self.img_lambda(self.data[idx])
        if os.path.exists(img_idx):
            return {
                "output_id": self.data[idx]['question_id'],
                "input_ids": input_text,
                "pixel_values": self.transform(Image.open(img_idx).convert("RGB")),
            }
        return {
            "output_id": self.data[idx]['question_id'],
            "input_ids": input_text,
            "pixel_values": self.empty_img.clone(),
        }


def load_dataset_pope(model: LinearVLM):
    data = []
    for fx in os.listdir(f"{path_to_pope}"):
        js = json.loads(open(f"{path_to_pope}/{fx}").read())
        for i in js:
            i['question_id'] = f"{fx}-{i['question_id']}"
            data.append(i)
    transform = model.transform
    return VQADataset(data, transform, 'pope', 2, lambda x: x['text'],
                      lambda x: f"{path_to_coco_val_2014}/{x['image']}")


def load_dataset_vqa(model):
    js = json.loads(open(f"{path_to_vqav2}/v2_OpenEnded_mscoco_val2014_questions.json").read())[
        "questions"]
    return VQADataset(js, model.transform, 'vqav2', 16,
                      lambda x: x['question'],
                      lambda x: f"{path_to_coco_val_2014}/COCO_val2014_{x['image_id']:012d}.jpg")


def load_dataset_gqa(model):
    data = []
    js = json.loads(open(f"{path_to_gqa}/testdev_balanced_questions.json").read())
    for k, v in js.items():
        data.append(
            {
                "question_id": k,
                "text": v["question"],
                "image": f"{path_to_gqa}/images/{v['imageId']}.jpg"
            }
        )
    return VQADataset(data, model.transform, 'gqa', 256,
                      lambda x: x['text'],
                      lambda x: x['image'])


def load_dataset_manip(model):
    data = []
    js = json.loads(open(f"{path_to_sapien}/maniptest.json").read())
    for v in js:
        data.append(
            {
                "question_id": v['image'].replace('robot_pics/','').replace('.png',''),
                "text": v["conversations"][0]['value'],
                "image": f"{path_to_sapien}/robot_pics/{v['image']}"
            }
        )
    return VQADataset(data, model.transform, 'manip', 32, lambda x: x['text'], lambda x: x['image'])

def load_dataset_vizwiz(model):
    js = json.loads(open(f"{path_to_vizwiz}/val.json").read())
    data = []
    for idx, i in enumerate(js):
        i['question_id'] = idx
        data.append(i)
    return VQADataset(data, model.transform, 'vizwiz', 64, 
                      lambda x: x['question'] + "When the provided information is insuffcient, respond with 'Unanswerable'. Answer the question using a single word or phrase. " ,
                       lambda x: f"{path_to_vizwiz}/val/{x['image']}")


def load_dataset_okvqa(model):
    js = json.loads(open(f"{path_to_okvqa}/OpenEnded_mscoco_val2014_questions.json").read())['questions']
    return VQADataset(js, model.transform, 'okvqa', 64,
                      lambda x: x['question'],
                      lambda x: f"{path_to_coco_val_2014}/COCO_val2014_{x['image_id']:012d}.jpg")

def load_dataset_robovqa(model):
    data = []
    js = json.loads(open(f"{path_to_robovqa}/robovqa.json").read())
    for v in js:
        data.append(
            {
                "question_id": v['id'],
                "text": v["conversations"][0]['value'],
                "image": f"{path_to_robovqa}/{v['image']}"
            }
        )
    return VQADataset(data, model.transform, 'robovqa', 128, lambda x: x['text'], lambda x: x['image'])



def judge_pope(args):
    prefix = os.path.join(args.save_prefix, "pope")
    d = []
    d2 = {}
    for f in os.listdir(prefix):
        for line in open(prefix + '/' + f):
            sp = line.strip().split(',', 1)
            d.append({"question_id": sp[0], "answer": sp[1]})
            d2[sp[0]] = sp[1]
    json.dump(d, open(prefix + '.json', 'w'), indent=4)
    pred, gt = {}, {}
    all_p, all_g = [], []
    for ix in os.listdir(path_to_pope):
        print(ix)
        pred[ix], gt[ix] = [], []
        with open(f'{path_to_pope}/' + ix) as f:
            js = json.load(f)
            for i in js:
                question_id = f"{ix}-{i['question_id']}"
                ans = d2[question_id][:3].lower() == 'yes'
                pred[ix].append(int(ans))
                gt[ix].append(int(i['label'] == 'yes'))
                all_p.append(int(ans))
                all_g.append(int(i['label'] == 'yes'))
            print("f1", f1_score(gt[ix], pred[ix]))
            print("acc", accuracy_score(gt[ix], pred[ix]))
    print("all")
    print("f1", f1_score(all_g, all_p))
    print("acc", accuracy_score(all_g, all_p))
    print("yes rate", sum(all_p) / len(all_p))


def judge_vqa(args):
    save_prefix = os.path.join(args.save_prefix, "vqav2")
    d = []
    d2 = {}
    for f in os.listdir(save_prefix):
        for line in open(os.path.join(save_prefix, f)):
            sp = line.strip().split(',', 1)
            if sp[0].isdigit():
                d.append({"question_id": int(sp[0]), "answer": sp[1].split("Answer:")[0]})
                d2[int(sp[0])] = sp[1].split("Answer:")[0]
    json.dump(d, open(save_prefix + '.json', 'w'), indent=4)

    print(compute_vqa_accuracy(save_prefix + '.json',
                               f"{path_to_vqav2}/v2_OpenEnded_mscoco_val2014_questions.json",
                               f"{path_to_vqav2}/v2_mscoco_val2014_annotations.json"))


def judge_gqa(args):
    prefix = os.path.join(args.save_prefix, "gqa")
    d = []
    d2 = {}
    for f in os.listdir(prefix):
        for line in open(os.path.join(prefix, f)):
            sp = line.strip().split(',', 1)
            d.append({"question_id": sp[0], "answer": sp[1]})
            d2[sp[0]] = sp[1]
    json.dump(d, open(f'{prefix}.json', 'w'), indent=4)
    js = json.loads(open(f"{path_to_gqa}/testdev_balanced_questions.json").read())
    result = []
    for k, v in js.items():
        x1 = v["answer"].lower().replace("men", "man").replace("drapes", "curtains").replace("girl", "woman")
        x2 = d2[k].lower().replace("men", "man").replace("drapes", "curtains").replace("girl", "woman")
        result.append(x1 == x2[:len(x1)] or all(u in x2.split() for u in x1.split()))
    print("acc", sum(result) / len(result))


def judge_viz(args):
    save_prefix = os.path.join(args.save_prefix, "vizwiz")
    d = []
    d2 = {}
    for f in os.listdir(save_prefix):
        for line in open(os.path.join(save_prefix, f)):
            sp = line.strip().split(',', 1)
            if sp[0].isdigit():
                d.append({"question_id": int(sp[0]), "answer": sp[1].split("Answer:")[0]})
                d2[int(sp[0])] = sp[1].split("Answer:")[0]
    json.dump(d, open(save_prefix + '.json', 'w'), indent=4)
    results = []
    with open(f"{path_to_vizwiz}/val.json") as f:
        js = json.load(f)
        for idx,i in enumerate(js):
            ans = i['answers']
            res = d2[idx].lower()
            
            for p in punct:
                res = res.replace(p,' ')

            res = res.replace("\n", " ")
            res = res.replace("\t", " ")
            res = res.strip()

            for l in range(len(ans)):
                for p in punct:
                    ans[l]['answer'] = ans[l]['answer'].replace(p,' ')
                ans[l]['answer'] = ans[l]['answer'].replace("\n", " ")
                ans[l]['answer'] = ans[l]['answer'].replace("\t", " ")
                ans[l]['answer'] = ans[l]['answer'].strip()
            gtAcc = []
            for gt in ans:
                otherGTAns = [
                    item['answer'] for item in ans if item != gt
                ]
                matchingAns = [item for item in otherGTAns if (res[:len(item)]==item or all(it.lower() in res  for it in item.split()))]
                acc = min(1, float(len(matchingAns)) / 3)
                gtAcc.append(acc)
            results.append(sum(gtAcc)/len(gtAcc))
    print(sum(results)/len(results))


def judge_okvqa(args):
    save_prefix = os.path.join(args.save_prefix, "okvqa")
    d = []
    d2 = {}
    for f in os.listdir(save_prefix):
        for line in open(os.path.join(save_prefix, f)):
            sp = line.strip().split(',', 1)
            if sp[0].isdigit():
                d.append({"question_id": int(sp[0]), "answer": sp[1].split("Answer:")[0]})
                d2[int(sp[0])] = sp[1].split("Answer:")[0]
    json.dump(d, open(save_prefix + '.json', 'w'), indent=4)

    print(compute_vqa_accuracy(save_prefix + '.json',
                               f"{path_to_okvqa}/OpenEnded_mscoco_val2014_questions.json",
                               f"{path_to_okvqa}/mscoco_val2014_annotations.json"))



def judge_manip(args):
    prefix = os.path.join(args.save_prefix, "manip")
    d = []
    d2 = {}
    for f in os.listdir(prefix):
        for line in open(prefix + '/' + f):
            sp = line.strip().split(',', 1)
            d.append({"image_id": sp[0], "detection": eval(sp[1])})
            d2[sp[0]] = sp[1]
    json.dump(d, open(prefix + '.json', 'w'), indent=4)


def judge_robovqa(args):
    prefix = os.path.join(args.save_prefix, "robovqa")
    d = []
    d2 = {}
    for f in os.listdir(prefix):
        for line in open(prefix + '/' + f):
            sp = line.strip().split(',', 1)
            d.append({"question_id": sp[0], "answer": sp[1]})
            d2[sp[0]] = sp[1]
    json.dump(d, open(prefix + '.json', 'w'), indent=4)
    js = json.load(open(f'{path_to_robovqa}/robovqa.json'))
    val = []
    for i in js:
        result = re.sub(r'<\|.*?\|>', ' ', d2[i['id']])
        result = re.sub(r'<.*?>', ' ', result)
        result = result.replace('A:',' ').replace(':',' ').replace('(',' ').replace(')',' ')
        result = result.strip()
        ans = "".join(jieba.cut(result))
        gt = ''.join(jieba.cut(i['conversations'][1]['value'].split(":")[1])).strip()

        val.append([
            max(sentence_bleu([ans[:len(gt)-lenth].split()], gt.split(), smoothing_function=SmoothingFunction().method4, weights=(1,0,0,0)) for lenth in range(-5,5)),
            max(sentence_bleu([ans[:len(gt)-lenth].split()], gt.split(), smoothing_function=SmoothingFunction().method4, weights=(0,1,0,0)) for lenth in range(-5,5)),
            max(sentence_bleu([ans[:len(gt)-lenth].split()], gt.split(), smoothing_function=SmoothingFunction().method4, weights=(0,0,1,0)) for lenth in range(-5,5)),
            max(sentence_bleu([ans[:len(gt)-lenth].split()], gt.split(), smoothing_function=SmoothingFunction().method4, weights=(0,0,0,1)) for lenth in range(-5,5)),
        ])
    print('bleu1', sum([i[0] for i in val])/len(val))
    print('bleu2', sum([i[1] for i in val])/len(val))
    print('bleu3', sum([i[2] for i in val])/len(val))
    print('bleu4', sum([i[3] for i in val])/len(val))


def run_test(dataset, model, args):
    local_rank, global_rank, world_size = world_info_from_env()
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=local_rank,
        shuffle=True,
        seed=args.seed,
    )
    # the batch_size and num_workers are per-GPU !
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=0,
        sampler=sampler,
    )

    if local_rank == 0:
        tq = tqdm(dataloader)
    else:
        tq = dataloader
    save_prefix = os.path.join(args.save_prefix, dataset.prefix)
    os.makedirs(save_prefix, exist_ok=True)

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    
    with torch.autocast("cuda", dtype=dtype, enabled=True):
        with open(f"{save_prefix}/result-train-test-{global_rank}.txt", "a+") as f:
            for batch in tq:
                sequence = model.tokenizer(batch["input_ids"], return_tensors='pt', padding=True, truncation=True)
                input_ids = sequence["input_ids"].to(global_rank)

                if args.run_type == 'VLM':
                    text = model.generate_batch(
                        batch["pixel_values"].to(global_rank, dtype=dtype),
                        input_ids,
                        max_new_tokens=dataset.new_token,
                        do_sample=True,
                        temperature=0.4,
                    )
                    for idx, answer in zip(batch['output_id'], text):
                        answer = answer.strip().replace('\n', ' ')
                        f.write(f"{idx},{answer}\n")
                else:
                    res = model(batch["pixel_values"].to(global_rank, dtype=dtype), input_ids)
                    for idx, answer in zip(batch['output_id'], res):
                        answer = answer.detach().float().cpu().tolist()
                        f.write(f"{idx},{answer}\n")

def run(args):
    local_rank, global_rank, world_size = world_info_from_env()
    device = torch.device(f"cuda:{global_rank}") if torch.cuda.is_available() else torch.device("cpu")
    partial_state = PartialState(device=device)
    # In case your GPU does not support bf16
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    model = create_model(llm_type=args.llm_name, vision_encoder=args.vision_encoder_name, types=args.run_type)
    model.load_state_dict(torch.load(args.checkpoint, map_location='cpu'), strict=False)

    if args.lora_checkpoint:
        model.lora()
        model.load_state_dict(torch.load(args.lora_checkpoint, map_location='cpu'))

    model = model.to(device=partial_state.device, dtype=dtype)
    model.tokenizer.pad_token = model.tokenizer.eos_token
    model.eval()

    datasets = {
        'pope': {
            'load': load_dataset_pope,
            'judge': judge_pope
        },
        'vqav2': {
            'load': load_dataset_vqa,
            'judge': judge_vqa
        },
        'gqa': {
            'load': load_dataset_gqa,
            'judge': judge_gqa
        },
        'vizwiz': {
            'load': load_dataset_vizwiz,
            'judge': judge_viz
        },
        'okvqa': {
            'load': load_dataset_okvqa,
            'judge': judge_okvqa
        },
        'manip': {  
            'load': load_dataset_manip,
            'judge': judge_manip
        },
        'robovqa':{
            'load': load_dataset_robovqa,
            'judge': judge_robovqa
        }
    }

    device_idx = 0
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=True):
        for test_dataset in args.dataset:
            if test_dataset in datasets:
                run_test(datasets[test_dataset]['load'](model), model, args)
                torch.distributed.barrier()
                if local_rank == device_idx % world_size:
                    print(f"Testing on {test_dataset}")
                    datasets[test_dataset]['judge'](args)
                device_idx += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # model configuration args
    parser.add_argument("--vision_encoder_name", default='SIGLIP', type=str)
    parser.add_argument("--llm_name", default='790m', type=str)
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="path to checkpoint to test",
        default=None,
    )
    parser.add_argument("--lora_checkpoint", type=str, default="")
    parser.add_argument("--run_type", default='VLM', type=str)
    # test args
    parser.add_argument("--save_prefix", type=str, default="result")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset", nargs='+', default=['pope', 'gqa', 'vizwiz', 'okvqa', 'vqav2'])
    # distributed training args
    parser.add_argument("--dist-url", default="env://")
    parser.add_argument("--dist-backend", default="nccl", type=str)
    

    args = parser.parse_args()
    args.local_rank, _, _ = world_info_from_env()
    torch.distributed.init_process_group(
        backend=args.dist_backend, init_method=args.dist_url
    )
    args.world_size = torch.distributed.get_world_size()
    args.rank = torch.distributed.get_rank()

    with torch.no_grad():
        run(args)
