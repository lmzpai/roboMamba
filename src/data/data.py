import json
import os
import re
import pickle
import pathlib

from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
from torch.nn.utils.rnn import pad_sequence

IGNORED = -100


class JsonDatset(Dataset):
    def __init__(self, json_path, transform=None, tokenizer=None):
        super(JsonDatset, self).__init__()
        print(json_path)
        self.data = json.load(open(json_path))
        self.len = len(self.data)
        self.transform = transform
        self.tokenizer = tokenizer
        self.img_path = os.path.split(json_path)[0]
        #self.tokenizer.model_max_length = 384
        

    def __getitem__(self, idx):
        js = self.data[idx]
        content = js["conversations"]
        image = Image.open(os.path.join(self.img_path,js["image"])) if 'image' in js else Image.new('RGB', (1, 1))
        if image.mode != 'RGB':
            image = image.convert('RGB')
        text = []
        target = []
        img = self.transform(image)

        if image.size[0] == 1:
            img = torch.zeros_like(img)

        for i in range(len(content)):
            # if len(content[i]['value']) >=200:
            #     continue
            if i % 2 == 0:
                text_now = self.tokenizer.encode(f"<|user|> {content[i]['value']}\n", add_special_tokens=(i==0))
                text.extend(text_now)
                target.extend(len(text_now) * [IGNORED])
                #target += text_now
            else:
                text_now = self.tokenizer.encode(f"<|assistant|> {content[i]['value']}\n",add_special_tokens=False)
                text.extend(text_now)
                target.extend(text_now)

        target.extend(self.tokenizer.encode("<|endoftext|>"))
        text.extend(self.tokenizer.encode("<|endoftext|>"))

        target = torch.tensor(target[: self.tokenizer.model_max_length-258])
        text = torch.tensor(text[: self.tokenizer.model_max_length-258])

        assert len(text) == len(target)
        return img, text, target

    def __len__(self):
        return self.len
    
class ManipulationDataset(Dataset):
    def __init__(self, json_path, transform=None, tokenizer=None) -> None:
        super(ManipulationDataset, self).__init__()
        self.data = json.load(open(json_path))
        self.len = len(self.data)
        self.transform = transform
        self.tokenizer = tokenizer
        self.img_path = os.path.split(json_path)[0]

    def __getitem__(self, idx):
        js = self.data[idx]
        content = js["conversations"]
        image = Image.open(os.path.join(self.img_path,js["image"])) if 'image' in js else Image.new('RGB', (1, 1))
        if image.mode != 'RGB':
            image = image.convert('RGB')
        text = self.tokenizer.encode(f"<|user|> {content[0]['value']}\n", add_special_tokens=True)
        label = content[1]['value']
        target = list(map(lambda x:float(x), label.split('(')[1].split(')')[0].split(', '))) \
         + list(map(lambda x:float(x)/50, label.split('[')[1].split(']')[0].split(', '))) \
         + list(map(lambda x:float(x)/50, label.split('[')[2].split(']')[0].split(', ')))
        assert len(target) == 8, f"{target}"
        img = self.transform(image)

        if image.size[0] == 1:
            img = torch.zeros_like(img)
    
        return img, torch.tensor(text), torch.FloatTensor(target), js["image"]
    
    def __len__(self):
        return self.len

LANGUAGE_DESCRIPTION = {
    'assembly': 'Pick up a nut and place it onto a peg.',
    'basketball': 'Dunk the basketball into the basket.',
    'bin-picking': 'Grasp the puck from one bin and place it into another bin.',
    'box-close': 'Grasp the cover and close the box with it.',
    'button-press-topdown': 'Press a button from the top.',
    'button-press-topdown-wall': 'Bypass a wall and press a button from the top.',
    'button-press': 'Press a button.',
    'button-press-wall': 'Bypass a wall and press a button.',
    'coffee-button': 'Push a button on the coffee machine.',
    'coffee-pull': 'Pull a mug from a coffee machine.',
    'coffee-push': 'Push a mug into a coffee machine.',
    'dial-turn': 'Rotate a dial 180 degrees.',
    'disassemble': 'pick a nut out of the a peg.',
    'door-close': 'Close a door with a revolving joint.',
    'door-lock': 'Lock the door by rotating the lock clockwise.',
    'door-open': 'Open a door with a revolving joint.',
    'door-unlock': 'Unlock the door by rotating the lock counter-clockwise.',
    'drawer-close': 'Push and close a drawer.',
    'drawer-open': 'Open a drawer.',
    'faucet-close': 'Rotate the faucet clockwise.',
    'faucet-open': 'Rotate the faucet counter-clockwise.',
    'hammer': 'Hammer a screw on the wall.',
    'hand-insert': 'Insert the gripper into a hole.',
    'handle-press-side': 'Press a handle down sideways.',
    'handle-press': 'Press a handle down.',
    'handle-pull-side': 'Pull a handle up sideways.',
    'handle-pull': 'Pull a handle up.',
    'lever-pull': 'Pull a lever down 90 degrees.',
    'peg-insert-side': 'Insert a peg sideways.',
    'peg-unplug-side': 'Unplug a peg sideways.',
    'pick-out-of-hole': 'Pick up a puck from a hole.',
    'pick-place': 'Pick and place a puck to a goal.',
    'pick-place-wall': 'Pick a puck, bypass a wall and place the puck.',
    'plate-slide-back-side': 'Get a plate from the cabinet sideways.',
    'plate-slide-back': 'Get a plate from the cabinet.',
    'plate-slide-side': 'Slide a plate into a cabinet sideways.',
    'plate-slide': 'Slide a plate into a cabinet.',
    'push-back': 'Pull a puck to a goal.',
    'push': 'Push a puck to a goal.',
    'push-wall': 'Bypass a wall and push a puck to a goal.',
    'reach': 'reach a goal position.',
    'reach-wall': 'Bypass a wall and reach a goal.',
    'shelf-place': 'pick and place a puck onto a shelf.',
    'soccer': 'Kick a soccer into the goal.',
    'stick-pull': 'Grasp a stick and pull a box with the stick.',
    'stick-push': 'Grasp a stick and push a box with the stick.',
    'sweep-info': 'Sweep a puck into a hole.',
    'sweep': 'Sweep a puck off the table.',
    'window-close': 'Close a door with a revolving joint.',
    'window-open': 'Open a window with a revolving joint.',
}


class MetaworldDataset(torch.utils.data.Dataset):
    """
    Dataset for Metaworld Benchmark.
    
    Images range: [0, 255]
    Robot states range: [-1.0, 1.0]
    Raw states range: [-1.0, 1.0]
    Actions range: [-7.0, 13.0]
    """
    def __init__(self, data_dir, train: bool, transform=None, tokenizer=None):
        file_name = 'train.pkl' if train else 'validation.pkl'
        data_path = pathlib.Path(data_dir) / file_name
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
            # (T, H, W, C) -> (T, C, H, W)
            self._images = data['images']
            self._robot_states = data['robot_states']
            self._raw_states = data['raw_states']
            self._actions = data['actions']
            self._episode_ends = data['episode_ends']
        assert len(self._images) == len(self._robot_states) == len(self._raw_states) == len(self._actions)
        self._dataset_size = len(self._actions)
        self._text = LANGUAGE_DESCRIPTION[os.path.basename(data_dir).split('.')[0]]
        self.transform = transform
        self.tokenizer = tokenizer

    def __getitem__(self, idx):
        image = self.transform(Image.fromarray(self._images[idx].astype('uint8')))
        text = torch.LongTensor(self.tokenizer.encode(f"<|user|> <image>\n{self._text}\n<|assistant|>", add_special_tokens=True))
        robot_state = torch.from_numpy(self._robot_states[idx]).float()
        robot_state[-1] /= 0.6
        #raw_state = torch.from_numpy(self._raw_states[idx]).float()
        action = torch.from_numpy(self._actions[idx]).float()
        return image, text, robot_state, action

    def __len__(self):
        return self._dataset_size

    
def json_collate_fn(batch):
    return (
        default_collate([i[0] for i in batch]),
        pad_sequence([i[1] for i in batch], batch_first=True, padding_value=0),
        pad_sequence([i[2] for i in batch], batch_first=True, padding_value=IGNORED),
    )
            
def manip_collate_fn(batch):
    return (
        default_collate([i[0] for i in batch]),
        pad_sequence([i[1] for i in batch], batch_first=True, padding_value=0),
        default_collate([i[2] for i in batch]),
        [i[3] for i in batch]
    )
            
def create_dataset(json_path, transform, tokenizer, types):
    if types == 'VLM':
        return JsonDatset(json_path, transform, tokenizer)
    elif types == 'MANIP':
        return ManipulationDataset(json_path, transform, tokenizer)
    elif types == 'METAWORLD':
        return MetaworldDataset(json_path, True, transform, tokenizer)
    assert False, f"Unknown dataset type: {types}"

# def create_dataloader(json_path, batch_size, transform, tokenizer):
#     return DataLoader(JsonDatset(json_path, transform, tokenizer), batch_size=batch_size, shuffle=True, collate_fn=json_collate_fn)

if __name__ == "__main__":
    for i in MetaworldDataset('/sata/ljm/csx/Codes/Robotic-Eval/data/metaworld-336/assembly', True):
        print(i)