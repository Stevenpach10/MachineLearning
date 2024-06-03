import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pytorch_lightning as pl

class BirdsDataset(Dataset):
    def __init__(self, data_dir, selected_classes=None, transform=None):
        self.data_dir = data_dir
        self.transform = transform

        if selected_classes is None:
            self.class_counts = self._count_images_per_class()
            self.selected_classes = self._select_top_classes(30)
        else:
            self.selected_classes = selected_classes

        self.class_to_idx = {cls: i for i, cls in enumerate(self.selected_classes)}
        self.samples = self._load_samples()

    def _count_images_per_class(self):
        """

        """
        class_counts = {}
        # get the classes names and count each file jpg and png and fill a dictionary
        for cls in os.listdir(self.data_dir):
            class_dir = os.path.join(self.data_dir, cls)
            if os.path.isdir(class_dir):
                count = sum([1 for _, _, files in os.walk(class_dir) for file in files if file.endswith('.jpg') or file.endswith('.png')])
                class_counts[cls] = count
        return class_counts

    def _select_top_classes(self, top_n):
        # Ordenar las clases por número de imágenes y seleccionar las top_n
        sorted_classes = sorted(self.class_counts.items(), key=lambda item: item[1], reverse=True)
        top_classes = [cls for cls, _ in sorted_classes[:top_n]]
        return top_classes

    def _load_samples(self):
        samples = []
        for cls in self.selected_classes:
            class_dir = os.path.join(self.data_dir, cls)
            for root, _, files in os.walk(class_dir):
                for file in files:
                    if file.endswith('.jpg') or file.endswith('.png'):
                        sample_path = os.path.join(root, file)
                        samples.append((sample_path, self.class_to_idx[cls]))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')  # Asegurarse de que las imágenes sean en RGB
        if self.transform:
            image = self.transform(image)
        return image, label

class ImageDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, num_workers, batch_size=32):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        # Transformaciones para las imágenes
        transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Cambiar tamaño a 64x64
            transforms.ToTensor(),         # Convertir la imagen a tensor
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalizar
        ])
        
        # Crear el dataset de entrenamiento y obtener las clases seleccionadas
        self.train_dataset = BirdsDataset(data_dir=os.path.join(self.data_dir, 'train'), transform=transform)
        self.selected_classes = self.train_dataset.selected_classes

        # Crear datasets de validación y prueba usando las clases seleccionadas
        self.val_dataset = BirdsDataset(data_dir=os.path.join(self.data_dir, 'val'), selected_classes=self.selected_classes, transform=transform)
        self.test_dataset = BirdsDataset(data_dir=os.path.join(self.data_dir, 'test'), selected_classes=self.selected_classes, transform=transform)


    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers = self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers = self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers = self.num_workers)