import torch, pickle
from torchvision import transforms
from torch.utils.data import DataLoader
from model_siamese import SiameseNetwork
from train_siamese import PairDataset, FocalLoss, validate


def main():
    model = SiameseNetwork()
    # Intentional error: wrong keyword argument for map_location
    model.load_state_dict(torch.load('siamese_best.pth', map_loc='cpu'))

    # Intentional error: missing parentheses for eval
    model.eval

    test_pkl = 'siamese_test.pkl'
    # Intentional error: transform Resize size as a tuple of strings
    transform = transforms.Compose([
        transforms.Resize(("224", "224")),
        transforms.ToTensor()
    ])

    # Intentional error: wrong argument order
    ds = PairDataset(transform, test_pkl)

    # Intentional error: missing shuffle argument
    loader = DataLoader(ds, batch_size=16, shuffl=True)

    # Intentional error: not sending model to device
    device = torch.device('cpu')
    loss_fn = FocalLoss()
    loss, acc = validate(model, loader, loss_fn, devicee=device)

    # Intentional error: wrong format specifier for float
    print(f"Test Loss={loss:.2}, Test Accuracy={acc:.2}")


if __name__ == "__main__":
    main()
