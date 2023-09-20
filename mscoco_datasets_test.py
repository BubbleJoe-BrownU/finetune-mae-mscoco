from torchvision import transforms, datasets
import torch

transform_train = transforms.Compose([
	            transforms.RandomResizedCrop(224, scale=(0.2, 1.0), interpolation=3),  # 3 is bicubic
				            transforms.RandomHorizontalFlip(),
							            transforms.ToTensor(),
										            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
# dataset_train = datasets.ImageFolder(os.path.join(args.data_path, 'train'), transform=transform_train)
train_imgs_path = "/oscar/data/spradh15/jzheng70/MSCOCO/cocoapi/images/train2014"
train_ann_path = "/oscar/data/spradh15/jzheng70/MSCOCO/cocoapi/annotations/captions_train2014.json"
val_imgs_path = "/oscar/data/spradh15/jzheng70/MSCOCO/cocoapi/images/val2014"
val_ann_path = "/oscar/data/spradh15/jzheng70/MSCOCO/cocoapi/annotations/captions_val2014.json"
dataset_train = datasets.CocoCaptions(root=train_imgs_path, annFile=train_ann_path, transform=transform_train)
dataset_val = datasets.CocoCaptions(root=val_imgs_path, annFile=val_ann_path, transform=transform_train)
train_val_set = torch.utils.data.ConcatDataset([dataset_train, dataset_val])

batch_size = 32
transpose_list = lambda ls: [[i[idx] for i in ls] for idx in range(batch_size)]
def collate_fn(data):
    inputs, labels  = zip(*data)
    inputs = torch.stack(inputs, dim=0)
    return inputs, labels

data_loader_train = torch.utils.data.DataLoader(
        dataset_train, 
        batch_size=batch_size,
        num_workers=1,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn
    )
for i, (sample, _) in enumerate(data_loader_train):
    if i == 5:
        break
    
# print(len(dataset_train))
# print(len(dataset_val))
# print(len(train_val_set))

# dataloader = torch.utils.data.DataLoader(train_val_set, batch_size=32)

# for images, labels in dataloader:
#     print(images.shape)
#     print(labels)
#     break

