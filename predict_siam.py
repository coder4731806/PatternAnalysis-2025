import torch, pickle
from torchvision import transforms
from torch.utils.data import DataLoader
from model_siamese import SiameseNetwork
from train_siamese import PairDataset, FocalLoss, validate

def main():
    model=SiameseNetwork()
    model.load_state_dict(torch.load('siamese_best.pth',map_location='cpu'))
    model.eval()
    test_pkl='siamese_test.pkl'
    transform=transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor()
    ])
    ds=PairDataset(test_pkl,transform)
    loader=DataLoader(ds,batch_size=16)
    loss_fn=FocalLoss()
    loss,acc=validate(model,loader,loss_fn,torch.device('cpu'))
    print(f"Test Loss={loss:.4f}, Test Accuracy={acc:.4f}")

if __name__=="__main__":
    main()
